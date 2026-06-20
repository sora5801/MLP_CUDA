// =============================================================================
// src/main.cu  —  Entry point: the whole pipeline wired together end to end.
// -----------------------------------------------------------------------------
// ROLE OF THIS FILE
//   This is the runnable program that ties every other piece of the repo into a
//   single, observable demonstration of "train an MLP on the GPU from scratch":
//
//     dataset.cu  -> synthetic data on the HOST (CPU)
//     matrix.cu   -> device (GPU global memory) buffers + H2D/D2H copies
//     kernels.cu  -> the actual CUDA work (GEMM, ReLU, softmax, gradients, SGD)
//     mlp.cu      -> orchestrates kernels into forward / backward / update,
//                    plus the loss, accuracy, and finite-difference grad-check
//
//   main() reads like the narrative of a training run:
//     1. print the GPU we are running on (so the numbers below have context),
//     2. microbenchmark naive vs tiled GEMM with cudaEvents (a CUDA lesson),
//     3. run a gradient check to PROVE the backward pass is correct,
//     4. run the epoch loop (shuffle -> batch -> H2D -> fwd -> bwd -> SGD),
//     5. report loss/accuracy, then free everything and cudaDeviceReset().
//
//   Every step is commented to explain how it connects the kernels and layers.
//   The reader is assumed to know C++ but to be learning CUDA, so CUDA-specific
//   ideas (events, host vs device memory, async launches) are spelled out.
// =============================================================================

#include <cstdio>      // printf
#include <cstdlib>     // EXIT_SUCCESS
#include <vector>      // std::vector for small host-side scratch buffers
#include <cmath>       // (not strictly required, but handy/illustrative)

#include "common.cuh"  // CUDA_CHECK / CUDA_CHECK_LAST, kBlockSize, kTileDim, ceil_div
#include "matrix.cuh"  // Matrix struct + device-memory helpers
#include "kernels.cuh" // launch_gemm / launch_gemm_tiled (used by the benchmark)
#include "mlp.cuh"     // MLP struct + create/forward/backward/step/loss/acc/gradcheck
#include "dataset.cuh" // make_blobs / dataset_standardize / dataset_shuffle / dataset_free

// -----------------------------------------------------------------------------
// CONFIGURATION CONSTANTS (spec §2 src/main.cu, item 1)
// -----------------------------------------------------------------------------
// These are compile-time constants so the whole experiment is reproducible and
// the shapes that flow through the kernels are easy to trace by hand.
//
// IMPORTANT divisibility note: we drop the last partial batch each epoch (see the
// training loop), so we choose n_per_class such that the TOTAL sample count is an
// exact multiple of kBatchSize. With 3 classes * 256 = 768 samples and batch 64
// that is exactly 12 full batches and zero leftover. Dropping a partial batch is
// the standard simplification: a smaller final batch would change the 1/batch
// scale baked into the gradients (see cross_entropy_grad) and complicate the
// fixed-size reusable device batch buffer below.
static constexpr unsigned long long kSeed        = 1234567ULL; // master RNG seed
static constexpr int   kNPerClass   = 256;   // samples per class -> 768 total
static constexpr int   kNFeatures   = 2;     // 2-D points: easy to reason about
static constexpr int   kNClasses    = 3;     // 3 Gaussian blobs / 3 logits out
static constexpr int   kHidden0     = 64;    // first hidden layer width
static constexpr int   kHidden1     = 32;    // second hidden layer width
static constexpr int   kBatchSize   = 64;    // rows per forward/backward pass
static constexpr int   kEpochs      = 60;    // full passes over the dataset
static constexpr float kLearningRate= 0.10f; // SGD step size: param -= lr * grad
static constexpr float kClusterStd  = 0.60f; // blob spread; smaller => separable
static constexpr int   kReportEvery = 10;    // print metrics every N epochs

// Microbenchmark problem size: square M=N=K so the comparison is symmetric and
// the arithmetic intensity is high enough that tiling can win. 512^3 multiply.
static constexpr int   kBenchM = 512;
static constexpr int   kBenchN = 512;
static constexpr int   kBenchK = 512;

// =============================================================================
// static helper: print_device_props
// -----------------------------------------------------------------------------
// WHAT:  Query and print properties of the active CUDA device (GPU). Purely
//        didactic — it gives the reader the hardware context (core count proxy,
//        clock, memory, shared-memory-per-block) that explains the timings and
//        the choice of kBlockSize / kTileDim.
// WHY:   cudaGetDeviceProperties fills a cudaDeviceProp struct describing the
//        currently selected device. We do not change devices, so device 0 is the
//        default. None of these queries touch the data; they are metadata reads.
// PARAMS: none (operates on device 0).
// RETURNS: void; prints to stdout.
// =============================================================================
static void print_device_props() {
    int device = 0;
    // cudaGetDevice writes the id of the device that subsequent runtime calls and
    // kernel launches will target. On a single-GPU box this is 0.
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop; // plain host struct; cudaGetDeviceProperties fills it.
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("==================== GPU DEVICE ====================\n");
    printf("Device %d: %s\n", device, prop.name);
    // Compute capability gates which features/instructions are available and
    // which -arch=sm_XX you should compile for (see Makefile/CMakeLists).
    printf("  Compute capability      : %d.%d\n", prop.major, prop.minor);
    // Total global memory: where every Matrix.data buffer lives (device DRAM).
    printf("  Global memory           : %.2f GB\n",
           static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    // Shared memory per block: the fast on-chip scratch that gemm_tiled uses for
    // its kTileDim x kTileDim tiles. Two float tiles of 16x16 = 2*1024 bytes.
    printf("  Shared mem per block    : %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("  Multiprocessors (SMs)   : %d\n", prop.multiProcessorCount);
    printf("  Max threads / block     : %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size               : %d\n", prop.warpSize);
    printf("  Clock rate              : %.0f MHz\n", prop.clockRate / 1000.0);
    printf("====================================================\n\n");
}

// =============================================================================
// static helper: run_gemm_microbenchmark
// -----------------------------------------------------------------------------
// WHAT:  Time the NAIVE one-thread-per-element GEMM against the shared-memory
//        TILED GEMM on a fixed square problem (kBenchM x kBenchN x kBenchK), then
//        print both times and the speedup. Both compute C = A * B with no
//        transposes, so the *results* are identical and only the *performance*
//        differs — that is the lesson: tiling reuses operands out of fast shared
//        memory instead of re-reading them from slow global memory.
//
// WHY cudaEvents:  A kernel launch is ASYNCHRONOUS — control returns to the CPU
//        before the GPU finishes. So you cannot time it with a CPU clock around
//        the launch. cudaEvent_t timestamps are recorded INTO the GPU stream;
//        cudaEventElapsedTime measures GPU time between two recorded events, and
//        cudaEventSynchronize blocks the CPU until the "stop" event has actually
//        occurred on the device. The launch_* wrappers already call
//        CUDA_CHECK_LAST() (cudaDeviceSynchronize), so by the time a launcher
//        returns the kernel is done; we still use events for precise GPU timing.
//
// MEMORY: We allocate three device matrices A[M,K], B[K,N], C[M,N] via
//        matrix_alloc (cudaMalloc) and fill A,B with deterministic host data
//        copied H2D. C is written by the kernels. All freed before returning.
//
// PARAMS: none (uses the kBench* constants).
// RETURNS: void; prints timings.
// =============================================================================
static void run_gemm_microbenchmark() {
    const int M = kBenchM, N = kBenchN, K = kBenchK;

    printf("============== GEMM MICROBENCHMARK =================\n");
    printf("Problem: C[%d,%d] = A[%d,%d] * B[%d,%d]  (no transpose)\n",
           M, N, M, K, K, N);

    // ----- allocate device operands (row-major flat float arrays) -----
    // A is [M,K], B is [K,N], C is [M,N]; see GEMM semantics in kernels.cuh.
    Matrix A = matrix_alloc(M, K);
    Matrix B = matrix_alloc(K, N);
    Matrix C = matrix_alloc(M, N);

    // ----- fill A and B with cheap deterministic host data, then copy H2D -----
    // The exact values do not matter for timing; we just need real numbers so the
    // multiply does genuine work and is not optimized away. std::vector is host
    // (CPU) memory; matrix_copy_to_device does the cudaMemcpy(HostToDevice).
    std::vector<float> hA(static_cast<size_t>(M) * K);
    std::vector<float> hB(static_cast<size_t>(K) * N);
    for (size_t i = 0; i < hA.size(); ++i)
        hA[i] = static_cast<float>((i % 13) - 6) * 0.1f; // small bounded values
    for (size_t i = 0; i < hB.size(); ++i)
        hB[i] = static_cast<float>((i % 7) - 3) * 0.1f;
    matrix_copy_to_device(A, hA.data()); // H2D: hA -> A.data (GPU)
    matrix_copy_to_device(B, hB.data()); // H2D: hB -> B.data (GPU)

    // ----- create the two CUDA events used as GPU stopwatches -----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_naive = 0.0f, ms_tiled = 0.0f;

    // ----- WARM-UP -----
    // The very first kernel launch pays one-time costs (JIT/context, caches cold).
    // We run each kernel once untimed so the measured run reflects steady state.
    launch_gemm(A.data, B.data, C.data, M, N, K, false, false);
    launch_gemm_tiled(A.data, B.data, C.data, M, N, K);

    // ----- time the NAIVE GEMM -----
    // cudaEventRecord(start) enqueues a timestamp into the (default) stream.
    CUDA_CHECK(cudaEventRecord(start));
    launch_gemm(A.data, B.data, C.data, M, N, K, false, false);
    CUDA_CHECK(cudaEventRecord(stop));
    // Block the CPU until 'stop' has been reached on the GPU, then read elapsed ms.
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    // ----- time the TILED GEMM (same operands, same result) -----
    CUDA_CHECK(cudaEventRecord(start));
    launch_gemm_tiled(A.data, B.data, C.data, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start, stop));

    // FLOPs for a dense GEMM: each of the M*N outputs is a length-K dot product,
    // i.e. K multiplies + K adds => 2*M*N*K floating-point operations. Dividing
    // by time gives GFLOP/s, a hardware-independent way to feel the speedup.
    const double flops = 2.0 * M * N * K;
    const double gflops_naive = flops / (ms_naive * 1.0e6); // ms -> GFLOP/s
    const double gflops_tiled = flops / (ms_tiled * 1.0e6);

    printf("  naive GEMM : %8.3f ms  (%7.1f GFLOP/s)\n", ms_naive, gflops_naive);
    printf("  tiled GEMM : %8.3f ms  (%7.1f GFLOP/s)\n", ms_tiled, gflops_tiled);
    if (ms_tiled > 0.0f)
        printf("  speedup    : %6.2fx (tiling reuses operands via shared memory)\n",
               ms_naive / ms_tiled);
    printf("====================================================\n\n");

    // ----- cleanup: destroy events, free device matrices -----
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

// =============================================================================
// main — the full pipeline.
// =============================================================================
int main() {
    // -------------------------------------------------------------------------
    // STEP 5 (done first so all following numbers have hardware context):
    // print which GPU we are on and its key limits.
    // -------------------------------------------------------------------------
    print_device_props();

    // -------------------------------------------------------------------------
    // STEP 6: GEMM microbenchmark (naive vs tiled). Independent of the MLP; it
    // exercises the same launch_gemm used in forward/backward, plus the tiled
    // variant, and teaches why shared-memory tiling matters.
    // -------------------------------------------------------------------------
    run_gemm_microbenchmark();

    // -------------------------------------------------------------------------
    // STEP 2: build the synthetic dataset on the HOST, then standardize it.
    //   make_blobs draws kNPerClass points per class from Gaussians with distinct
    //   centers (mostly separable). dataset_standardize rescales each feature
    //   column to zero mean / unit variance so the first GEMM's inputs are well
    //   scaled — this keeps ReLU activations and gradients from blowing up or
    //   vanishing and is standard practice before feeding a network.
    // -------------------------------------------------------------------------
    Dataset data = make_blobs(kNPerClass, kNFeatures, kNClasses,
                              kClusterStd, kSeed);
    dataset_standardize(data);
    const int n_samples = data.n_samples;            // = kNPerClass * kNClasses
    const int n_batches = n_samples / kBatchSize;    // full batches; remainder dropped
    printf("Dataset: %d samples, %d features, %d classes -> %d full batches "
           "of %d (last partial batch dropped)\n\n",
           n_samples, data.n_features, data.n_classes, n_batches, kBatchSize);

    // -------------------------------------------------------------------------
    // STEP 3: define the layer widths and create the network.
    //   layer_sizes = {in, hidden0, hidden1, out}. mlp_create makes
    //   (num_sizes - 1) = 3 layers; the last is the softmax output layer. Weights
    //   are He-initialized (N(0, sqrt(2/in))) for ReLU; biases start at 0. The
    //   per-layer Matrix caches (Z, A, dW, db, dZ, dA) are sized for kBatchSize.
    // -------------------------------------------------------------------------
    const int layer_sizes[] = { kNFeatures, kHidden0, kHidden1, kNClasses };
    const int num_sizes = static_cast<int>(sizeof(layer_sizes) / sizeof(int));
    MLP net = mlp_create(layer_sizes, num_sizes, kBatchSize, kSeed);

    // -------------------------------------------------------------------------
    // STEP 4: allocate ONE reusable device batch and label buffers.
    //   We do NOT reallocate per batch — instead we copy each new batch's data
    //   into these fixed buffers (cudaMalloc is relatively expensive, and reuse
    //   mirrors how real training loops pin a fixed-shape input tensor).
    //     d_batch  : device Matrix [kBatchSize, kNFeatures] — the network input.
    //     d_labels : device int[kBatchSize] — true class per row, for the
    //                gradient (cross_entropy_grad) and the loss kernel.
    //     h_labels : host  int[kBatchSize] — same labels on the CPU, used by
    //                mlp_accuracy (which argmaxes host-side) and grad-check.
    // -------------------------------------------------------------------------
    Matrix d_batch = matrix_alloc(kBatchSize, kNFeatures);
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, sizeof(int) * kBatchSize));
    std::vector<int> h_labels(kBatchSize); // host scratch, length = batch

    // Host scratch for one batch of inputs we will copy H2D each step. Row-major
    // [kBatchSize, kNFeatures], matching d_batch's layout exactly.
    std::vector<float> h_batch(static_cast<size_t>(kBatchSize) * kNFeatures);

    // -------------------------------------------------------------------------
    // Helper lambda: load batch index `b` of the (already-shuffled) dataset into
    // the host scratch buffers and copy them to the device buffers.
    //   - Copies kBatchSize rows starting at row b*kBatchSize.
    //   - X rows are contiguous (n_features each) in data.X, so we can copy the
    //     whole block; labels are copied element-wise into h_labels then H2D.
    // This is the only data motion between CPU and GPU inside the loop.
    // -------------------------------------------------------------------------
    auto load_batch = [&](int b) {
        const int row0 = b * kBatchSize;                 // first sample of batch
        const size_t feat_off = static_cast<size_t>(row0) * data.n_features;
        const size_t feat_cnt = static_cast<size_t>(kBatchSize) * data.n_features;
        // Copy the contiguous [kBatchSize, n_features] block of inputs.
        for (size_t i = 0; i < feat_cnt; ++i)
            h_batch[i] = data.X[feat_off + i];
        // Copy the matching labels.
        for (int r = 0; r < kBatchSize; ++r)
            h_labels[r] = data.y[row0 + r];
        // Push both to the GPU. matrix_copy_to_device wraps cudaMemcpy H2D.
        matrix_copy_to_device(d_batch, h_batch.data());
        CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(),
                              sizeof(int) * kBatchSize, cudaMemcpyHostToDevice));
    };

    // -------------------------------------------------------------------------
    // STEP 7: gradient check BEFORE training, on the first batch.
    //   Load batch 0, run a forward pass (grad-check needs cached probabilities
    //   and will re-run forward internally for the +/- eps evaluations), then
    //   compare analytic dW from mlp_backward to a central finite difference.
    //   Small relative errors (~1e-4 or better) prove the backward math/kernels
    //   are correct without any reference framework.
    // -------------------------------------------------------------------------
    printf("============== GRADIENT CHECK =====================\n");
    load_batch(0);
    mlp_forward(net, d_batch);                       // populate output probs
    mlp_grad_check(net, d_batch, d_labels, h_labels.data());
    printf("====================================================\n\n");

    // -------------------------------------------------------------------------
    // STEP 8 + 10: the epoch training loop, timed end-to-end with cudaEvents.
    //   Per epoch:  shuffle the dataset (so batches differ each epoch), then for
    //   each full batch:
    //       load_batch  -> H2D copy of inputs+labels
    //       mlp_forward -> GEMM + bias + ReLU/softmax through all layers; the
    //                      output layer's A now holds class probabilities
    //       loss/acc    -> read those probs for reporting (does not affect grads)
    //       mlp_backward-> cross_entropy_grad seeds dZ, then GEMMs/ReLU' walk the
    //                      chain backward filling every dW/db (already /batch)
    //       mlp_sgd_step-> param -= lr*grad in place for every W and b
    //   We accumulate loss/acc across batches and print the epoch averages.
    // -------------------------------------------------------------------------
    cudaEvent_t train_start, train_stop;
    CUDA_CHECK(cudaEventCreate(&train_start));
    CUDA_CHECK(cudaEventCreate(&train_stop));
    CUDA_CHECK(cudaEventRecord(train_start)); // mark start of all training

    printf("==================== TRAINING =====================\n");
    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        // Shuffle X rows and y together (Fisher-Yates) so each epoch sees data in
        // a new order. We vary the seed by epoch to get a different but still
        // fully deterministic permutation every epoch.
        dataset_shuffle(data, kSeed + static_cast<unsigned long long>(epoch));

        double epoch_loss = 0.0; // sum of per-batch mean losses
        double epoch_acc  = 0.0; // sum of per-batch accuracies
        for (int b = 0; b < n_batches; ++b) {
            load_batch(b);                       // inputs+labels -> device

            mlp_forward(net, d_batch);           // forward pass through the net

            // Metrics use the freshly-cached output probabilities. mlp_compute_loss
            // runs the per-row CE kernel and averages on the host; mlp_accuracy
            // copies probs to host, argmaxes each row, compares to h_labels.
            epoch_loss += mlp_compute_loss(net, d_labels);
            epoch_acc  += mlp_accuracy(net, h_labels.data());

            mlp_backward(net, d_batch, d_labels); // fill dW/db (already /batch)
            mlp_sgd_step(net, kLearningRate);     // apply the SGD update
        }

        // Report on the first epoch, every kReportEvery epochs, and the last one.
        const bool report = (epoch == 0) ||
                            ((epoch + 1) % kReportEvery == 0) ||
                            (epoch == kEpochs - 1);
        if (report) {
            printf("  epoch %3d/%d  loss %.4f  train_acc %.3f\n",
                   epoch + 1, kEpochs,
                   epoch_loss / n_batches,
                   epoch_acc  / n_batches);
        }
    }

    // Stop the training timer. cudaEventSynchronize guarantees the GPU has
    // finished the last SGD kernel before we read elapsed time.
    CUDA_CHECK(cudaEventRecord(train_stop));
    CUDA_CHECK(cudaEventSynchronize(train_stop));
    float train_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&train_ms, train_start, train_stop));

    // Throughput: total samples processed / total seconds. Each epoch processes
    // n_batches*kBatchSize samples (the dropped remainder is excluded).
    const double total_samples =
        static_cast<double>(kEpochs) * n_batches * kBatchSize;
    const double samples_per_sec = total_samples / (train_ms / 1000.0);
    printf("  total training time: %.2f ms over %d epochs (%.0f samples/sec)\n",
           train_ms, kEpochs, samples_per_sec);
    printf("====================================================\n\n");

    // -------------------------------------------------------------------------
    // STEP 9: clean up EVERYTHING (no leaks), then reset the device.
    //   Order does not matter much here, but we mirror allocation: events, the
    //   reusable device buffers, the network's many device matrices (mlp_free),
    //   and finally the host dataset. cudaDeviceReset() destroys the CUDA context
    //   so any remaining device allocations are reclaimed and tools like
    //   compute-sanitizer / nvprof flush cleanly — good hygiene at program exit.
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(train_start));
    CUDA_CHECK(cudaEventDestroy(train_stop));
    matrix_free(d_batch);            // free reusable input batch (device)
    CUDA_CHECK(cudaFree(d_labels));  // free device label buffer
    mlp_free(net);                   // frees every Layer's W/b/Z/A/dW/db/dZ/dA
    dataset_free(data);              // frees host X and y

    CUDA_CHECK(cudaDeviceReset());   // tear down the CUDA context cleanly
    printf("Done. Device reset; all memory freed.\n");
    return EXIT_SUCCESS;
}
