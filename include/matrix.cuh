// ============================================================================
// include/matrix.cuh
// ----------------------------------------------------------------------------
// ROLE IN THE PROJECT
//   This header declares the `Matrix` type that every other part of the MLP
//   uses to talk about a 2-D block of `float`s that lives in GPU global memory,
//   plus the small set of host-callable helpers for allocating, freeing,
//   zeroing, and copying that memory. Think of `Matrix` as a thin, didactic
//   wrapper around a raw `cudaMalloc`'d pointer + its dimensions -- it carries
//   no methods and does no automatic memory management on purpose, so that the
//   CUDA memory model (explicit alloc / free / copy) stays visible to the
//   reader. The implementations live in `src/matrix.cu`.
//
// MEMORY LAYOUT (the single most important convention in this repo)
//   * Every matrix is stored ROW-MAJOR in a single flat `float*` array.
//   * The pointer `data` is a *device* pointer: it indexes GPU global memory,
//     NOT host (CPU) memory. You may not dereference it on the host; you must
//     `cudaMemcpy` to/from it. Passing it to a `printf` or `host[i]` is a bug.
//   * For a matrix of shape [rows, cols], the element at logical (r, c) lives at
//     flat index `r * cols + c`. Walking one column to the right is +1 element;
//     walking one row down is +cols elements. (This matches the global math
//     convention in the build spec: W is [in,out] => W[i*out + o]; activations
//     are [batch,features] => A[r*features + c].)
// ============================================================================

#pragma once

// We deliberately route through common.cuh so that anyone including matrix.cuh
// also gets the CUDA runtime header (<cuda_runtime.h>) and the CUDA_CHECK error
// macros that the implementations in matrix.cu rely on. Keeping all the "system
// + error-checking" includes in one place is the project convention.
#include "common.cuh"

// ----------------------------------------------------------------------------
// struct Matrix
// ----------------------------------------------------------------------------
// A plain-old-data (POD) descriptor for a 2-D, row-major float array stored in
// GPU global memory. It is intentionally a "dumb" struct:
//   * No constructor / destructor: ownership is explicit. You create storage
//     with matrix_alloc() and release it with matrix_free(). Copying a Matrix
//     by value copies the *pointer* (a shallow copy / alias), NOT the data --
//     two Matrix values can therefore point at the same device buffer, so be
//     careful never to free the same `data` twice.
//   * Trivially copyable, which is exactly what we want when we pass it around
//     by value as a lightweight handle.
//
// Fields:
//   data : device pointer into GPU global memory, row-major, holding exactly
//          rows*cols floats. May be nullptr (e.g. before allocation, or after
//          matrix_free has reset it). It is a DEVICE address -- only kernels and
//          cudaMemcpy may touch the bytes it points to.
//   rows : number of logical rows    (units: count; >= 0).
//   cols : number of logical columns (units: count; >= 0).
//
// The total element count is rows*cols and the total byte size is
// rows*cols*sizeof(float) (see matrix_bytes()).
struct Matrix {
    float* data;   // device pointer, row-major, length rows*cols (may be nullptr)
    int rows;      // number of rows    (logical dimension 0)
    int cols;      // number of columns (logical dimension 1)
};

// ----------------------------------------------------------------------------
// matrix_alloc
// ----------------------------------------------------------------------------
// Allocate an UNINITIALIZED rows*cols float buffer in GPU global memory and
// return a Matrix handle describing it.
//
// What happens under the hood (in matrix.cu): a single cudaMalloc reserves
// rows*cols*sizeof(float) contiguous bytes on the device and hands back a
// device pointer stored in the returned Matrix's `data` field. cudaMalloc does
// NOT zero the memory -- the contents are garbage until you write them (via a
// kernel, matrix_zero, or matrix_copy_to_device). Allocation failures are
// surfaced by the CUDA_CHECK wrapper, which aborts the program.
//
// Parameters:
//   rows : desired number of rows    (count, >= 0).
//   cols : desired number of columns (count, >= 0).
// Returns:
//   A Matrix with .rows/.cols set and .data pointing to freshly allocated
//   (uninitialized) device memory.
Matrix matrix_alloc(int rows, int cols);

// ----------------------------------------------------------------------------
// matrix_free
// ----------------------------------------------------------------------------
// Release the device buffer owned by `m` and reset the handle so it can no
// longer be used by accident.
//
// Implementation (matrix.cu): calls cudaFree(m.data) to return the GPU global
// memory to the allocator, then sets m.data = nullptr and m.rows = m.cols = 0.
// Resetting the fields turns a dangling pointer into an obviously-empty handle,
// which makes double-free / use-after-free bugs easier to catch. Because Matrix
// copies are shallow aliases, you must free a given device buffer exactly once
// (free the "owning" handle; do not also free aliases of it).
//
// Parameters:
//   m : the Matrix to free, taken by reference so its fields can be cleared.
void matrix_free(Matrix& m);

// ----------------------------------------------------------------------------
// matrix_zero
// ----------------------------------------------------------------------------
// Set every element of `m` to 0.0f.
//
// Implementation (matrix.cu): cudaMemset(m.data, 0, matrix_bytes(m)). NOTE the
// subtlety that makes this work for floats: cudaMemset writes BYTES, and the
// IEEE-754 bit pattern of all-zero bytes is exactly +0.0f. (This trick is valid
// for zeroing floats but would NOT produce, say, 1.0f -- memset is byte-wise.)
// Used to clear gradient accumulators and to initialize bias vectors to 0.
//
// Parameters:
//   m : the matrix whose rows*cols floats are overwritten with 0.0f. Passed by
//       reference for consistency with the other mutating helpers (its fields
//       are not changed, only the device bytes it points to).
void matrix_zero(Matrix& m);

// ----------------------------------------------------------------------------
// matrix_copy_to_device
// ----------------------------------------------------------------------------
// Host-to-Device (H2D) copy: upload rows*cols floats from a host array into the
// device buffer of `m`.
//
// Implementation (matrix.cu): cudaMemcpy(m.data, host, matrix_bytes(m),
// cudaMemcpyHostToDevice). The direction enum cudaMemcpyHostToDevice tells the
// runtime that the source pointer is CPU memory and the destination is GPU
// global memory; getting this enum wrong is a classic CUDA bug. The caller must
// guarantee that `host` points to at least rows*cols valid floats laid out in
// the SAME row-major order the device side expects.
//
// Parameters:
//   m    : destination matrix; m.data must already be allocated (non-null) and
//          large enough for rows*cols floats.
//   host : source pointer in HOST memory, length rows*cols floats, row-major.
void matrix_copy_to_device(Matrix& m, const float* host);

// ----------------------------------------------------------------------------
// matrix_copy_to_host
// ----------------------------------------------------------------------------
// Device-to-Host (D2H) copy: download rows*cols floats from the device buffer
// of `m` into a host array. This is how the CPU "reads back" results computed
// on the GPU (e.g. probabilities for accuracy, per-row losses for reporting).
//
// Implementation (matrix.cu): cudaMemcpy(host, m.data, matrix_bytes(m),
// cudaMemcpyDeviceToHost). The cudaMemcpyDeviceToHost enum marks the source as
// GPU memory and the destination as CPU memory. A plain cudaMemcpy like this is
// SYNCHRONOUS with respect to the host, so once it returns the host array is
// safe to read.
//
// Parameters:
//   m    : source matrix; taken by const reference (we only read it). m.data
//          must point to rows*cols valid device floats.
//   host : destination pointer in HOST memory, must hold >= rows*cols floats.
void matrix_copy_to_host(const Matrix& m, float* host);

// ----------------------------------------------------------------------------
// matrix_bytes
// ----------------------------------------------------------------------------
// Return the size of the matrix's backing buffer in bytes: the count the other
// helpers feed to cudaMalloc / cudaMemset / cudaMemcpy.
//
// Computed as static_cast<size_t>(rows) * cols * sizeof(float). Doing the
// multiplication in size_t (not int) avoids 32-bit overflow for large matrices.
//
// Parameters:
//   m : the matrix to measure (const reference; not modified).
// Returns:
//   rows*cols*sizeof(float) as a size_t (units: bytes).
size_t matrix_bytes(const Matrix& m);
