// ============================================================================
//  src/dataset.cu
// ----------------------------------------------------------------------------
//  ROLE IN THE PROJECT
//  This file generates and preprocesses the *synthetic* training data that the
//  MLP learns from. There is NO CUDA here on purpose: data generation and
//  preprocessing are cheap, one-time, host-side (CPU) operations. The arrays
//  produced here live in ordinary host (CPU) memory; main.cu later copies each
//  mini-batch up to the GPU (host->device) inside the training loop.
//
//  WHAT IT CONTAINS
//    * make_blobs        - draw Gaussian "blob" clusters, one per class.
//    * dataset_free      - release the host arrays.
//    * dataset_standardize - per-feature zero-mean / unit-variance scaling.
//    * dataset_shuffle   - Fisher-Yates shuffle that keeps X rows and y in sync.
//
//  HOW IT FITS THE WHOLE
//  The Dataset struct holds X ([n_samples, n_features], row-major) and y
//  ([n_samples]). dataset_standardize is applied once after generation so the
//  network sees well-scaled inputs; dataset_shuffle is applied once per epoch so
//  mini-batches are not correlated with class order. Everything is deterministic
//  (std::mt19937_64 seeded explicitly) so an entire training run reproduces
//  bit-for-bit, which is essential when studying/debugging.
// ============================================================================

#include "dataset.cuh"   // Dataset struct + the four function declarations.

#include <cmath>         // std::sqrt for the unit-variance rescale.
#include <cstdlib>       // std::malloc / std::free for host buffers.
#include <random>        // std::mt19937_64, normal_distribution, uniform_int.

// ----------------------------------------------------------------------------
//  Dataset make_blobs(int n_per_class, int n_features, int n_classes,
//                     float cluster_std, unsigned long long seed)
// ----------------------------------------------------------------------------
//  WHAT IT COMPUTES
//  A classic "isotropic Gaussian blobs" classification dataset. For each class
//  c in [0, n_classes) we pick a fixed center vector and then draw n_per_class
//  points from a Gaussian N(center_c, cluster_std^2 * I) around it. Points are
//  labeled with their class c. Because the centers are spread apart and the
//  per-class spread (cluster_std) is modest, the classes are *mostly* linearly
//  separable -- easy enough that a small MLP reaches high accuracy, but not so
//  trivial that training is uninformative.
//
//  PARAMETERS (shapes / units)
//    n_per_class : number of samples drawn per class           (count, > 0)
//    n_features  : dimensionality of each sample (input dim)   (count, >= 1)
//    n_classes   : number of distinct classes / blobs          (count, >= 1)
//    cluster_std : standard deviation of each blob's Gaussian   (same units as X)
//                  -- larger => fuzzier, more overlapping clusters.
//    seed        : seed for std::mt19937_64; identical seed => identical data.
//
//  RETURNS
//    A Dataset whose:
//      X         : HOST array, length n_samples*n_features, row-major.
//                  Row r is sample r; X[r*n_features + f] is feature f.
//      y         : HOST array, length n_samples; y[r] in [0, n_classes).
//      n_samples : n_per_class * n_classes (samples are grouped by class on
//                  creation; call dataset_shuffle to break that ordering).
//
//  MEMORY LAYOUT
//    X is a single flat malloc of n_samples*n_features floats in ROW-MAJOR
//    order (feature index varies fastest). This matches the [batch, features]
//    convention used everywhere downstream so a contiguous slice of rows is a
//    valid mini-batch with no repacking.
//
//  CENTER PLACEMENT (why a circle)
//    We place class centers evenly on a circle of radius `radius` in the first
//    two feature dimensions: center_c = (radius*cos(theta_c), radius*sin(theta_c)).
//    A circle guarantees every pair of centers is well separated regardless of
//    n_classes. Feature dimensions beyond the first two get center 0 (the blobs
//    differ only in the first two dims there); for the project's n_features=2
//    case all dimensions are used.
Dataset make_blobs(int n_per_class, int n_features, int n_classes,
                   float cluster_std, unsigned long long seed) {
    Dataset d;
    d.n_features = n_features;
    d.n_classes  = n_classes;
    d.n_samples  = n_per_class * n_classes;   // total points across all blobs.

    // Allocate the flat host buffers. We use malloc (not new[]) to mirror the
    // C-style buffer management used elsewhere and to pair cleanly with the
    // std::free in dataset_free. Sizes are in BYTES for malloc.
    d.X = (float*)std::malloc((size_t)d.n_samples * d.n_features * sizeof(float));
    d.y = (int*)  std::malloc((size_t)d.n_samples * sizeof(int));

    // The single source of randomness for this dataset. Seeding explicitly is
    // what makes generation reproducible (GLOBAL RULE 7: determinism).
    std::mt19937_64 rng(seed);

    // A standard normal generator N(0,1); we scale its draws by cluster_std to
    // get N(0, cluster_std^2) noise that we add to each class center.
    std::normal_distribution<float> noise(0.0f, cluster_std);

    // Radius of the circle on which class centers sit. A few * cluster_std keeps
    // neighboring blobs separated relative to their spread (good separability).
    const float radius = 5.0f;

    // 2*pi; used to space the class centers evenly around the circle.
    const float kTwoPi = 6.28318530717958647692f;

    int row = 0;   // running output row index into X / y.
    for (int c = 0; c < n_classes; ++c) {
        // Angle for this class center: classes are spaced uniformly so the
        // angular gap between adjacent centers is 2*pi / n_classes.
        const float theta = kTwoPi * (float)c / (float)n_classes;
        const float cx = radius * std::cos(theta);   // center, feature 0.
        const float cy = radius * std::sin(theta);   // center, feature 1.

        for (int i = 0; i < n_per_class; ++i) {
            // Pointer to the start of this sample's feature row in X.
            float* sample = &d.X[(size_t)row * n_features];

            // Feature 0: center x + Gaussian noise. Guarded because n_features
            // could (in principle) be 0; for this project it is always >= 2.
            if (n_features > 0) sample[0] = cx + noise(rng);
            // Feature 1: center y + Gaussian noise.
            if (n_features > 1) sample[1] = cy + noise(rng);
            // Any extra dimensions are pure noise around 0 (center component 0).
            for (int f = 2; f < n_features; ++f) sample[f] = noise(rng);

            d.y[row] = c;   // label this sample with its generating class.
            ++row;
        }
    }
    return d;
}

// ----------------------------------------------------------------------------
//  void dataset_free(Dataset& d)
// ----------------------------------------------------------------------------
//  Releases the two host arrays and zeroes the struct fields so a stale, freed
//  pointer can never be dereferenced or double-freed by mistake. Safe to call
//  on a Dataset whose pointers are already null (std::free(nullptr) is a no-op).
void dataset_free(Dataset& d) {
    std::free(d.X);
    std::free(d.y);
    d.X = nullptr;
    d.y = nullptr;
    d.n_samples  = 0;
    d.n_features = 0;
    d.n_classes  = 0;
}

// ----------------------------------------------------------------------------
//  void dataset_standardize(Dataset& d)
// ----------------------------------------------------------------------------
//  WHAT IT DOES
//  Rescales each FEATURE COLUMN of X, in place, to have zero mean and unit
//  variance across all samples:
//      X[:,f] <- (X[:,f] - mean_f) / std_f
//  computed independently per feature f.
//
//  WHY (this matters for training)
//  Neural nets train far better on standardized inputs. With He-initialized
//  weights, the variance of a layer's pre-activations scales with the variance
//  of its inputs; features on wildly different scales make some weights'
//  gradients dominate and slow/destabilize gradient descent. Centering to mean
//  0 and scaling to std 1 keeps the first layer's activations well-conditioned,
//  so a single global learning rate works for every input dimension.
//
//  PARAMETER (shape)
//    d : Dataset whose X is [n_samples, n_features] row-major; modified in place.
//
//  MEMORY-ACCESS NOTE
//  Because X is row-major, a feature COLUMN is strided (stride = n_features), so
//  each pass over a column touches memory with gaps. That is fine on the host
//  for a one-time preprocessing step; we favor clarity over cache-optimality
//  here. (optimization: a single fused pass accumulating per-feature sums and
//  sums-of-squares is left as an exercise.)
void dataset_standardize(Dataset& d) {
    const int n = d.n_samples;
    const int F = d.n_features;
    if (n <= 0 || F <= 0) return;   // nothing to do for an empty dataset.

    for (int f = 0; f < F; ++f) {
        // --- Pass 1: mean of feature f over all samples. ---
        double sum = 0.0;   // double accumulator to limit float rounding drift.
        for (int r = 0; r < n; ++r) {
            sum += (double)d.X[(size_t)r * F + f];
        }
        const double mean = sum / (double)n;

        // --- Pass 2: variance of feature f (population variance, /n). ---
        double sq = 0.0;
        for (int r = 0; r < n; ++r) {
            const double diff = (double)d.X[(size_t)r * F + f] - mean;
            sq += diff * diff;
        }
        double var = sq / (double)n;

        // Standard deviation. Guard against a zero/near-zero std (a constant
        // feature): dividing by it would produce inf/nan, so floor it to 1 so
        // the (already mean-zero) column is left effectively unscaled.
        double stddev = std::sqrt(var);
        if (stddev < 1e-12) stddev = 1.0;

        // --- Pass 3: apply (x - mean) / std in place to the whole column. ---
        for (int r = 0; r < n; ++r) {
            float* x = &d.X[(size_t)r * F + f];
            *x = (float)(((double)(*x) - mean) / stddev);
        }
    }
}

// ----------------------------------------------------------------------------
//  void dataset_shuffle(Dataset& d, unsigned long long seed)
// ----------------------------------------------------------------------------
//  WHAT IT DOES
//  Randomly permutes the SAMPLE ORDER of the dataset in place using the
//  Fisher-Yates (a.k.a. Knuth) shuffle, keeping each X row glued to its label
//  y so sample identity is preserved -- only the ordering changes.
//
//  WHY
//  make_blobs emits samples grouped by class (all of class 0, then all of
//  class 1, ...). If we trained on contiguous mini-batches of that order, an
//  early batch would contain only one class -> hugely biased gradients. Calling
//  this once per epoch decorrelates batch contents from class order, which is
//  standard practice for stochastic gradient descent. Seeding per call keeps
//  the whole run reproducible (e.g. pass `base_seed + epoch`).
//
//  FISHER-YATES (why it is a *uniform* permutation)
//  Iterate i from the last index down to 1; pick j uniformly in [0, i]; swap
//  elements i and j. Each of the n! orderings is equally likely. We swap BOTH
//  the feature row (n_features floats) and the scalar label together, so the
//  (X_row, y) pairing is never broken.
//
//  PARAMETERS
//    d    : Dataset to permute in place; X is [n_samples, n_features] row-major.
//    seed : seed for std::mt19937_64 driving the index draws (determinism).
//
//  MEMORY LAYOUT NOTE
//  Swapping two rows means swapping their n_features contiguous floats; we do it
//  element-by-element with a scalar temp (no extra heap allocation). The label
//  swap is a single int exchange.
void dataset_shuffle(Dataset& d, unsigned long long seed) {
    const int n = d.n_samples;
    const int F = d.n_features;
    if (n <= 1) return;   // 0 or 1 sample: already "shuffled".

    std::mt19937_64 rng(seed);

    // Walk from the last index down to 1 (index 0 needs no further swap).
    for (int i = n - 1; i > 0; --i) {
        // Draw j uniformly in [0, i] INCLUSIVE. The inclusive upper bound (j may
        // equal i, a no-op swap) is what makes the permutation unbiased.
        std::uniform_int_distribution<int> pick(0, i);
        const int j = pick(rng);
        if (j == i) continue;   // swapping a row with itself is a no-op.

        // --- Swap the two feature rows, float by float. ---
        float* ri = &d.X[(size_t)i * F];   // start of row i.
        float* rj = &d.X[(size_t)j * F];   // start of row j.
        for (int f = 0; f < F; ++f) {
            const float tmp = ri[f];
            ri[f] = rj[f];
            rj[f] = tmp;
        }

        // --- Swap the matching labels so each row keeps its class. ---
        const int ty = d.y[i];
        d.y[i] = d.y[j];
        d.y[j] = ty;
    }
}
