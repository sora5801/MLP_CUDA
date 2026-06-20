// ============================================================================
// src/matrix.cu
// ----------------------------------------------------------------------------
// Role: Implementation of the small `Matrix` abstraction declared in
//       include/matrix.cuh. A `Matrix` is just a thin, non-owning-by-default
//       bundle of a *device* (GPU global memory) pointer plus its row/col
//       dimensions; the functions here are the only place in the whole repo
//       that talk directly to the CUDA memory-management runtime API
//       (cudaMalloc / cudaFree / cudaMemset / cudaMemcpy). Everything above
//       this layer (kernels, mlp, dataset) just manipulates `Matrix` handles.
//
// Why a wrapper at all? CUDA's C-style API is easy to misuse: byte sizes are
// computed by hand, copy directions are passed as enums, and every call can
// silently fail by returning a cudaError_t. Centralizing those concerns here
// (and routing every call through CUDA_CHECK) means the rest of the codebase
// reads in terms of matrices, not bytes and error codes.
//
// Memory-model reminder for the CUDA newcomer:
//   * Host memory  = ordinary CPU RAM, reachable by normal C++ pointers.
//   * Device memory= the GPU's own global memory (VRAM). A device pointer is
//     a *number that is only meaningful to the GPU*; you must NOT dereference
//     it on the host. You move data across the PCIe bus with cudaMemcpy.
//   * `Matrix::data` always points into device global memory.
// ============================================================================

#include "matrix.cuh"   // struct Matrix + the declarations we implement here
#include "common.cuh"   // CUDA_CHECK(...) wrapper around the runtime API

// ----------------------------------------------------------------------------
// matrix_bytes
// ----------------------------------------------------------------------------
// What:  Returns the number of *bytes* one needs to store this matrix's
//        elements contiguously: rows * cols * sizeof(float).
// Why:   Every cudaMalloc / cudaMemset / cudaMemcpy below is expressed in
//        bytes, never in element counts. Computing the byte size in one place
//        avoids the classic bug of passing an element count where a byte count
//        is expected (which would under-allocate/under-copy by 4x for floats).
// Params:
//   m  : the matrix whose logical size we want. Only m.rows and m.cols are
//        read; m.data is not touched, so this is valid even when data==nullptr.
// Shape/units: result is in bytes. The matrix is conceptually [rows, cols]
//        row-major, so the flat element count is rows*cols.
// Note on types: rows/cols are int, but we deliberately cast to size_t BEFORE
//        multiplying. size_t is the unsigned, pointer-width type the CUDA API
//        expects for sizes; doing the multiply in size_t avoids 32-bit int
//        overflow for large matrices (rows*cols could exceed 2^31).
size_t matrix_bytes(const Matrix& m) {
    return static_cast<size_t>(m.rows) *
           static_cast<size_t>(m.cols) *
           sizeof(float);
}

// ----------------------------------------------------------------------------
// matrix_alloc
// ----------------------------------------------------------------------------
// What:  Allocates a [rows, cols] float matrix in GPU global memory and
//        returns a Matrix handle pointing at it. The contents are UNINITIALIZED
//        (cudaMalloc does not zero memory, exactly like C's malloc).
// Why:   This is the single chokepoint for obtaining device storage in the
//        repo. Callers that need zeros should follow up with matrix_zero().
// Params:
//   rows : number of logical rows    (>= 0)            [count, unitless]
//   cols : number of logical columns (>= 0)            [count, unitless]
// Returns: a Matrix with .rows/.cols set and .data pointing into device global
//          memory (or nullptr if rows*cols == 0; see below).
// Memory layout: the returned buffer is a single contiguous, row-major block of
//          rows*cols floats: element (r,c) lives at flat index r*cols + c.
// CUDA detail: cudaMalloc(void** ptr, size_t bytes) writes a *device* address
//          into *ptr. We pass &m.data cast to void**, because m.data is float*
//          and cudaMalloc is type-agnostic. The returned address is only valid
//          on the GPU — never read/write m.data from host code.
Matrix matrix_alloc(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = nullptr;

    size_t bytes = matrix_bytes(m);   // rows*cols*sizeof(float)

    // Guard the degenerate zero-size case. cudaMalloc(.., 0) is allowed but its
    // behavior (may return nullptr or a non-dereferenceable pointer) is not
    // useful to us, so we simply leave data == nullptr and return.
    if (bytes == 0) {
        return m;
    }

    // Request `bytes` of device global memory. On success cudaMalloc stores the
    // device pointer into m.data; on failure CUDA_CHECK prints file/line and the
    // human-readable error (e.g. "out of memory") and aborts the program.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m.data), bytes));

    return m;
}

// ----------------------------------------------------------------------------
// matrix_free
// ----------------------------------------------------------------------------
// What:  Releases the device buffer owned by `m` and resets the handle so it
//        can't be accidentally reused (defensive "poisoning" to nullptr/0).
// Why:   GPU memory is a scarce, manually managed resource — every successful
//        matrix_alloc must be paired with exactly one matrix_free, or VRAM
//        leaks for the lifetime of the process/context.
// Params:
//   m : matrix to free, passed by reference so we can null out its fields.
//       Safe to call when m.data == nullptr (cudaFree(nullptr) is a no-op,
//       mirroring free(NULL)), which makes double-free / free-of-empty benign.
// Post-conditions: m.data == nullptr, m.rows == 0, m.cols == 0.
void matrix_free(Matrix& m) {
    if (m.data != nullptr) {
        // cudaFree returns the device allocation to the driver. Wrapped in
        // CUDA_CHECK so a freed-twice / invalid-pointer mistake surfaces loudly
        // instead of corrupting later allocations.
        CUDA_CHECK(cudaFree(m.data));
    }
    // Reset the handle. After this the Matrix describes an empty 0x0 matrix.
    m.data = nullptr;
    m.rows = 0;
    m.cols = 0;
}

// ----------------------------------------------------------------------------
// matrix_zero
// ----------------------------------------------------------------------------
// What:  Sets every float in the matrix to 0.0f, in place on the device.
// Why:   Gradients and accumulators must start at a known zero state; cudaMalloc
//        leaves memory uninitialized, so callers that need zeros call this.
// Params:
//   m : an allocated matrix (m.data must be valid device memory).
// CUDA detail: cudaMemset(ptr, value, bytes) writes `value` to each *byte* of
//        the buffer — it is a byte-wise memset, NOT an element-wise fill. We
//        pass value 0, and because the IEEE-754 bit pattern of +0.0f is all
//        zero bytes (0x00000000), zeroing the bytes correctly yields float 0.0.
//        (This trick does NOT generalize: cudaMemset(.., 1, ..) would NOT make
//        the floats 1.0f — it would set every byte to 0x01.)
// Edge case: if data == nullptr (empty matrix) there is nothing to zero.
void matrix_zero(Matrix& m) {
    if (m.data == nullptr) {
        return;
    }
    CUDA_CHECK(cudaMemset(m.data, 0, matrix_bytes(m)));
}

// ----------------------------------------------------------------------------
// matrix_copy_to_device  (Host -> Device, "H2D")
// ----------------------------------------------------------------------------
// What:  Copies rows*cols floats from a host array into this matrix's device
//        buffer, overwriting it entirely.
// Why:   This is how CPU-side data (e.g. a batch of input features built on the
//        host) gets onto the GPU so kernels can operate on it.
// Params:
//   m    : destination matrix; m.data is the device pointer we write into.
//   host : source pointer in *host* (CPU) memory. MUST point to at least
//          rows*cols contiguous floats laid out row-major in the SAME order as
//          the device buffer — element (r,c) at host[r*cols + c]. Marked const
//          because the host side is read-only here.
// Direction enum: the 4th argument cudaMemcpyHostToDevice tells the runtime the
//          source is host memory and the destination is device memory. The enum
//          is how the driver knows which way across the PCIe bus to move bytes;
//          getting it wrong is a common bug (it would copy from the wrong space
//          or error out). Mnemonic: arguments are (dst, src, ...), and the enum
//          reads in the same order: Host(=src)To Device(=dst).
// Sync note: cudaMemcpy (the non-Async form) is synchronous with respect to the
//          host — it returns only after the copy completes, so `host` may be
//          modified or freed immediately afterward.
void matrix_copy_to_device(Matrix& m, const float* host) {
    CUDA_CHECK(cudaMemcpy(m.data,                 // dst: device global memory
                          host,                   // src: host (CPU) memory
                          matrix_bytes(m),        // number of BYTES to copy
                          cudaMemcpyHostToDevice));// direction: H2D
}

// ----------------------------------------------------------------------------
// matrix_copy_to_host  (Device -> Host, "D2H")
// ----------------------------------------------------------------------------
// What:  Copies rows*cols floats from this matrix's device buffer back into a
//        host array, so the CPU can inspect results (loss, predictions, etc.).
// Why:   Kernels leave their outputs in device memory; to print or post-process
//        them on the CPU we must pull them back across the bus.
// Params:
//   m    : source matrix (const &) — we only read its device data and dims.
//   host : destination pointer in *host* memory. MUST have room for at least
//          rows*cols floats; receives them row-major (element (r,c) lands at
//          host[r*cols + c], matching the device layout exactly).
// Direction enum: cudaMemcpyDeviceToHost — source is device, destination is
//          host. Same (dst, src) argument order; the enum names src-To-dst.
// Sync note: like above, the plain cudaMemcpy blocks the host until the copy is
//          done. Conveniently this also means any kernels that wrote `m.data`
//          must have finished (the copy waits on the same default stream), so
//          the data read here reflects completed kernel output.
void matrix_copy_to_host(const Matrix& m, float* host) {
    CUDA_CHECK(cudaMemcpy(host,                    // dst: host (CPU) memory
                          m.data,                  // src: device global memory
                          matrix_bytes(m),         // number of BYTES to copy
                          cudaMemcpyDeviceToHost)); // direction: D2H
}
