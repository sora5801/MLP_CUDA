// ============================================================================
// demo/demo.cu  —  GUIDED FEATURE SHOWCASE (the Visual Studio entry point)
// ----------------------------------------------------------------------------
// ROLE
//   This is an ALTERNATE `main()` that exercises *everything* the repo has built
//   up across pushes 0001–0003, as a guided tour you can run and read top to
//   bottom. It is the program the Visual Studio solution (MLP_CUDA.sln) builds.
//
//   The plain `src/main.cu` is the concise demo used by the Makefile / CMake
//   command-line builds. To avoid two `main()` functions in one program, the
//   Visual Studio project compiles this file PLUS the library .cu files
//   (matrix/kernels/mlp/dataset/optim) but NOT src/main.cu. The command-line
//   builds do the opposite. So the two entry points never collide.
//
// WHAT IT SHOWCASES
//   1. GPU device info.
//   2. GEMM microbenchmark: naive vs shared-memory tiled (push 0001).
//   3. On-device counter-based RNG self-test (push 0003) + the parallel
//      reduction (push 0002) used to average it.
//   4. Finite-difference GRADIENT CHECK — the correctness proof (push 0001),
//      run with dropout active (push 0003).
//   5. OPTIMIZER shootout: SGD vs Momentum vs Adam on identical inits (push 0002).
//   6. ACTIVATION comparison: ReLU vs LeakyReLU vs Tanh (push 0002).
//   7. DROPOUT comparison: p = 0.0 vs 0.5, train (clean) vs held-out val
//      (push 0003), illustrating train/inference mode.
//
//   Every section prints a labeled banner so the console output reads like a
//   report. The heavy per-kernel commentary lives in the library files and in
//   docs/; here we comment the ORCHESTRATION and what each result means.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.cuh"   // CUDA_CHECK / CUDA_CHECK_LAST, ceil_div, constants
#include "matrix.cuh"   // Matrix + device-memory helpers
#include "kernels.cuh"  // launch_gemm / launch_gemm_tiled / fill_uniform / reduce_sum
#include "mlp.cuh"      // MLP API (create/forward/backward/evaluate/grad-check)
#include "dataset.cuh"  // make_blobs / standardize / shuffle / split
#include "optim.cuh"    // SGD / Momentum / Adam

// ---- Shared experiment configuration ---------------------------------------
static constexpr unsigned long long kSeed   = 1234567ULL; // master RNG seed
static constexpr int   kNPerClass = 256;   // 3 * 256 = 768 samples total
static constexpr int   kNFeatures = 2;     // 2-D points (easy to reason about)
static constexpr int   kNClasses  = 3;     // 3 Gaussian blobs
static constexpr int   kHidden0   = 64;    // hidden layer 1 width
static constexpr int   kHidden1   = 32;    // hidden layer 2 width
static constexpr int   kBatch     = 64;    // rows per step
static constexpr int   kEpochs    = 50;    // epochs per training run
static constexpr float kClusterStd= 0.60f; // blob spread
static constexpr int   kValSamples= 192;   // held-out validation rows (3 batches)

// =============================================================================
// Section 1 — GPU device info.
// =============================================================================
static void show_device() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int clock_khz = 0;
    // (CUDA 13 removed cudaDeviceProp::clockRate; query the attribute instead.)
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
    printf("================= GPU DEVICE ======================\n");
    printf("  %s  (compute %d.%d, %d SMs, %.0f MHz, %.1f GB)\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           clock_khz / 1000.0,
           (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("==================================================\n\n");
}

// =============================================================================
// Section 2 — GEMM microbenchmark: naive vs tiled (shared-memory) matmul.
// =============================================================================
static void gemm_benchmark() {
    const int M = 512, N = 512, K = 512;     // square so the comparison is symmetric
    Matrix A = matrix_alloc(M, K);
    Matrix B = matrix_alloc(K, N);
    Matrix C = matrix_alloc(M, N);

    // Fill A, B with cheap deterministic host data so the multiply does real work.
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = (float)((i % 13) - 6) * 0.1f;
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (float)((i % 7) - 3) * 0.1f;
    matrix_copy_to_device(A, hA.data());
    matrix_copy_to_device(B, hB.data());

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    float ms_naive = 0.0f, ms_tiled = 0.0f;

    // Warm up once (first launch pays one-time costs), then time each kernel.
    launch_gemm(A.data, B.data, C.data, M, N, K, false, false);
    launch_gemm_tiled(A.data, B.data, C.data, M, N, K);

    CUDA_CHECK(cudaEventRecord(s));
    launch_gemm(A.data, B.data, C.data, M, N, K, false, false);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, s, e));

    CUDA_CHECK(cudaEventRecord(s));
    launch_gemm_tiled(A.data, B.data, C.data, M, N, K);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, s, e));

    printf("============= GEMM 512x512x512 ===================\n");
    printf("  naive : %7.3f ms\n", ms_naive);
    printf("  tiled : %7.3f ms   (%.2fx faster via shared memory)\n",
           ms_tiled, ms_naive / ms_tiled);
    printf("==================================================\n\n");

    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

// =============================================================================
// Section 3 — On-device RNG self-test (RNG ⊗ reduction).
// =============================================================================
static void rng_selftest() {
    const int N = 1 << 20;             // ~1.05M uniforms
    Matrix u = matrix_alloc(1, N);
    launch_fill_uniform(u.data, N, kSeed);                 // counter-based RNG
    float mean = launch_reduce_sum(u.data, N) / (float)N;  // GPU tree reduction
    printf("============= RNG SELF-TEST ======================\n");
    printf("  %d uniforms in [0,1), mean = %.5f  (expected ~0.5)\n", N, mean);
    printf("==================================================\n\n");
    matrix_free(u);
}

// =============================================================================
// Training helper used by every comparison below.
// ----------------------------------------------------------------------------
//   Trains one freshly-built network on `train` and returns clean (inference-
//   mode) train & validation metrics plus the wall-clock time. All configs pass
//   the SAME `kSeed` to mlp_create, so their weight initializations are identical
//   — the only thing that differs is the knob under study (optimizer, activation,
//   or dropout), which makes the comparison fair.
// =============================================================================
struct TrainSummary {
    float train_loss, train_acc;   // measured in inference mode (dropout off)
    float val_loss,   val_acc;     // held-out
    float train_ms;                // total training wall-clock time
};

static TrainSummary train_run(Activation act, float dropout, OptConfig optcfg,
                              Dataset& train, const Dataset& val) {
    const int sizes[] = { kNFeatures, kHidden0, kHidden1, kNClasses };
    const int num_sizes = (int)(sizeof(sizes) / sizeof(int));

    MLP net = mlp_create(sizes, num_sizes, kBatch, kSeed, act, dropout);
    Optimizer opt = optim_create(net, optcfg);

    const int in_f      = net.input_features;
    const int n_batches = train.n_samples / kBatch;

    // Reusable device + host scratch for a single batch.
    Matrix d_batch = matrix_alloc(kBatch, in_f);
    int* d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, sizeof(int) * kBatch));
    std::vector<int>   h_labels(kBatch);
    std::vector<float> h_batch((size_t)kBatch * in_f);

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));

    for (int ep = 0; ep < kEpochs; ++ep) {
        // Reshuffle each epoch (deterministic per epoch) so batches decorrelate.
        dataset_shuffle(train, kSeed + 1u + (unsigned)ep);
        for (int b = 0; b < n_batches; ++b) {
            const int row0 = b * kBatch;
            for (size_t i = 0; i < (size_t)kBatch * in_f; ++i)
                h_batch[i] = train.X[(size_t)row0 * in_f + i];
            for (int r = 0; r < kBatch; ++r)
                h_labels[r] = train.y[row0 + r];
            matrix_copy_to_device(d_batch, h_batch.data());
            CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(),
                                  sizeof(int) * kBatch, cudaMemcpyHostToDevice));

            net.rng_state++;                 // fresh dropout mask each step (push 0003)
            mlp_forward(net, d_batch);       // GEMM + bias + activation (+dropout)
            mlp_backward(net, d_batch, d_labels);
            optim_step(opt, net);            // SGD / Momentum / Adam update
        }
    }

    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    TrainSummary sum;
    CUDA_CHECK(cudaEventElapsedTime(&sum.train_ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));

    // Final metrics in INFERENCE mode (mlp_evaluate flips dropout off), so the
    // "train" numbers are the clean (dropout-free) accuracy on the training data.
    mlp_evaluate(net, train.X, train.y, train.n_samples, sum.train_loss, sum.train_acc);
    mlp_evaluate(net, val.X,   val.y,   val.n_samples,   sum.val_loss,   sum.val_acc);

    matrix_free(d_batch);
    CUDA_CHECK(cudaFree(d_labels));
    optim_free(opt);
    mlp_free(net);
    return sum;
}

// Small helper: print one comparison-table row.
static void print_row(const char* label, const TrainSummary& s) {
    printf("  %-22s  train_acc %.3f  val_acc %.3f  val_loss %.4f  %6.1f ms\n",
           label, s.train_acc, s.val_acc, s.val_loss, s.train_ms);
}

// =============================================================================
// main — run the whole guided tour.
// =============================================================================
int main() {
    printf("\n##############################################################\n");
    printf("#  MLP_CUDA — feature showcase (pushes 0001-0003)            #\n");
    printf("##############################################################\n\n");

    // ---- 1-3: hardware, GEMM, RNG ----
    show_device();
    gemm_benchmark();
    rng_selftest();

    // ---- Build the dataset once: blobs -> standardize -> shuffle -> split ----
    Dataset data = make_blobs(kNPerClass, kNFeatures, kNClasses, kClusterStd, kSeed);
    dataset_standardize(data);
    dataset_shuffle(data, kSeed);
    Dataset train, val;
    dataset_split(data, data.n_samples - kValSamples, train, val);
    printf("Dataset: %d train / %d val samples, %d features, %d classes\n\n",
           train.n_samples, val.n_samples, data.n_features, data.n_classes);

    // ---- 4: gradient check (correctness proof), with dropout active ----
    // We build a small net WITH dropout and run the finite-difference check; it
    // passing (rel.err ~1e-4) proves the whole forward/backward chain — including
    // the dropout backward — matches calculus. (mlp_grad_check freezes the RNG
    // seed so the stochastic mask is identical across its perturbed passes.)
    printf("================ GRADIENT CHECK ==================\n");
    {
        const int sizes[] = { kNFeatures, kHidden0, kHidden1, kNClasses };
        MLP gnet = mlp_create(sizes, 4, kBatch, kSeed, Activation::ReLU, 0.2f);

        // Load the first training batch onto the device for the check.
        Matrix d_batch = matrix_alloc(kBatch, kNFeatures);
        int* d_labels = nullptr;
        CUDA_CHECK(cudaMalloc(&d_labels, sizeof(int) * kBatch));
        std::vector<int> h_labels(kBatch);
        std::vector<float> h_batch((size_t)kBatch * kNFeatures);
        for (size_t i = 0; i < (size_t)kBatch * kNFeatures; ++i) h_batch[i] = train.X[i];
        for (int r = 0; r < kBatch; ++r) h_labels[r] = train.y[r];
        matrix_copy_to_device(d_batch, h_batch.data());
        CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), sizeof(int) * kBatch,
                              cudaMemcpyHostToDevice));

        mlp_forward(gnet, d_batch);
        mlp_grad_check(gnet, d_batch, d_labels, h_labels.data());

        matrix_free(d_batch);
        CUDA_CHECK(cudaFree(d_labels));
        mlp_free(gnet);
    }
    printf("==================================================\n\n");

    // ---- 5: optimizer shootout (ReLU, no dropout, identical inits) ----
    printf("============ OPTIMIZER COMPARISON ================\n");
    printf("  (ReLU hidden, no dropout, %d epochs, same init seed)\n", kEpochs);
    print_row("SGD (lr 0.10)",      train_run(Activation::ReLU, 0.0f, opt_sgd(0.10f),            train, val));
    print_row("Momentum (0.10,0.9)",train_run(Activation::ReLU, 0.0f, opt_momentum(0.10f, 0.9f), train, val));
    print_row("Adam (lr 0.01)",     train_run(Activation::ReLU, 0.0f, opt_adam(0.01f),           train, val));
    printf("==================================================\n\n");

    // ---- 6: activation comparison (Adam, no dropout) ----
    printf("=========== ACTIVATION COMPARISON ===============\n");
    printf("  (Adam lr 0.01, no dropout, %d epochs)\n", kEpochs);
    print_row("ReLU",      train_run(Activation::ReLU,      0.0f, opt_adam(0.01f), train, val));
    print_row("LeakyReLU", train_run(Activation::LeakyReLU, 0.0f, opt_adam(0.01f), train, val));
    print_row("Tanh",      train_run(Activation::Tanh,      0.0f, opt_adam(0.01f), train, val));
    printf("==================================================\n\n");

    // ---- 7: dropout comparison (ReLU, Adam) ----
    // On these easily-separable blobs dropout is not NEEDED (val already ~1.0),
    // but this shows the mechanism runs end-to-end and that inference (val) uses
    // the clean, dropout-free network. With p=0.5, the clean train accuracy is
    // still high while training was done through a heavily-thinned network.
    printf("============= DROPOUT COMPARISON =================\n");
    printf("  (ReLU, Adam lr 0.01, %d epochs)\n", kEpochs);
    print_row("dropout p=0.0", train_run(Activation::ReLU, 0.0f, opt_adam(0.01f), train, val));
    print_row("dropout p=0.5", train_run(Activation::ReLU, 0.5f, opt_adam(0.01f), train, val));
    printf("==================================================\n\n");

    // ---- cleanup ----
    dataset_free(train);
    dataset_free(val);
    dataset_free(data);
    CUDA_CHECK(cudaDeviceReset());
    printf("Showcase complete. All device + host memory freed.\n");
    return EXIT_SUCCESS;
}
