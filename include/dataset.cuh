// =============================================================================
// include/dataset.cuh
// -----------------------------------------------------------------------------
// ROLE IN THE PROJECT
//   This header declares the *synthetic dataset* layer of the MLP study repo.
//   It defines a tiny, self-contained data source ("Gaussian blobs") plus the
//   host-side preprocessing utilities (standardization and shuffling) used to
//   feed the network. There is deliberately NO file/disk I/O and NO external
//   data dependency: the whole dataset is generated in RAM from a seed, so the
//   demo is reproducible and the reader can focus on the CUDA/MLP mechanics
//   rather than on data plumbing.
//
//   IMPORTANT — WHERE THE DATA LIVES:
//   Everything declared here is *HOST* (CPU) memory. The `Dataset` struct holds
//   plain `std::malloc`-allocated arrays on the host. Generation, standardization and
//   shuffling are all ordinary CPU code (no kernels). The training loop in
//   main.cu later copies one mini-batch at a time from this host storage into a
//   reusable device `Matrix` (via cudaMemcpy H2D). Keeping the dataset on the
//   host and streaming batches to the GPU is a common, easy-to-reason-about
//   pattern and keeps this file free of any CUDA-runtime concerns.
//
//   The implementations live in src/dataset.cu.
// =============================================================================

#pragma once

// We only need the C++ standard library for the *declarations* here (in fact we
// need nothing at all for the prototypes). The implementation file pulls in
// <random>, <cmath>, etc. We include <cstddef> so that any future size_t use is
// well-defined and to keep this header self-sufficient.
#include <cstddef>

// -----------------------------------------------------------------------------
// struct Dataset
// -----------------------------------------------------------------------------
// A flat, host-resident collection of labeled feature vectors.
//
// Memory layout:
//   X : pointer to a contiguous HOST array of `n_samples * n_features` floats,
//       stored ROW-MAJOR. Row r (sample r) occupies X[r*n_features ..
//       r*n_features + n_features - 1]; feature c of sample r is
//       X[r*n_features + c]. This matches the repo-wide convention that a batch
//       of activations has shape [rows, features] with element [r,c] at
//       index r*features + c — so a contiguous block of rows can be copied
//       straight into a device Matrix[batch, n_features] with no repacking.
//   y : pointer to a contiguous HOST array of `n_samples` ints. y[r] is the
//       integer class label of sample r, in the range [0, n_classes). These are
//       *indices*, not one-hot vectors; the cross-entropy kernels consume the
//       index form directly.
//
// Both X and y are owned by the Dataset and must be released with
// dataset_free(). A default/zeroed Dataset (X == nullptr, y == nullptr) is the
// "empty" state and is safe to pass to dataset_free().
//
// Scalar fields:
//   n_samples  : total number of points = n_per_class * n_classes (see below).
//   n_features : dimensionality of each point (e.g. 2 for easy 2-D plotting).
//   n_classes  : number of distinct labels / Gaussian clusters.
struct Dataset {
    float* X;        // HOST, [n_samples, n_features] row-major, owned.
    int*   y;        // HOST, [n_samples] class labels in [0, n_classes), owned.
    int n_samples;   // total points across all classes
    int n_features;  // features per point (columns of X)
    int n_classes;   // number of clusters / labels
};

// -----------------------------------------------------------------------------
// make_blobs
// -----------------------------------------------------------------------------
// Generate a synthetic, mostly-separable classification dataset of Gaussian
// "blobs". For each class c in [0, n_classes), we pick a fixed center in
// feature space and draw `n_per_class` points from an isotropic normal
// distribution N(center_c, cluster_std^2 * I). Class centers are spread out
// (e.g. around a circle / on a grid in the implementation) so the clusters are
// well separated for small cluster_std and start to overlap as it grows — a
// convenient knob for making the learning problem easier or harder.
//
// WHY synthetic blobs: they give a clean, low-dimensional, linearly-or-nearly
// separable problem where a small MLP should reach high accuracy quickly. That
// makes it obvious from the training output whether forward/backward are
// correct, without needing to download or parse a real dataset.
//
// Determinism: all random draws come from a single std::mt19937_64 seeded with
// `seed`, so the same arguments always produce byte-identical data. This is
// what lets the grad-check and training curves be reproducible run to run.
//
// Parameters:
//   n_per_class : number of points generated PER class (so total points =
//                 n_per_class * n_classes). Choose it so the total is divisible
//                 by the batch size if you want no dropped samples.
//   n_features  : dimensionality of each point (length of each row of X).
//   n_classes   : number of clusters/labels to generate.
//   cluster_std : standard deviation (in feature units) of each isotropic
//                 Gaussian blob; larger => more spread/overlap => harder.
//   seed        : 64-bit seed for the mt19937_64 RNG (reproducibility).
//
// Returns: a fully-populated Dataset with freshly host-allocated X and y. The
//          caller owns the result and must call dataset_free() on it.
Dataset make_blobs(int n_per_class, int n_features, int n_classes,
                   float cluster_std, unsigned long long seed);

// -----------------------------------------------------------------------------
// dataset_free
// -----------------------------------------------------------------------------
// Release the host arrays owned by `d` (std::free X and y) and reset all fields
// to their empty state (pointers to nullptr, counts to 0). Passing an
// already-empty Dataset is safe (std::free on nullptr is a no-op). Takes the
// Dataset by reference so the caller's copy is left in a clean, reusable state.
void dataset_free(Dataset& d);

// -----------------------------------------------------------------------------
// dataset_standardize
// -----------------------------------------------------------------------------
// Standardize each FEATURE COLUMN of d.X to zero mean and unit variance, IN
// PLACE, on the host. For each column c, compute the mean and (population)
// standard deviation over all n_samples rows, then replace every entry with
//   X[r,c] <- (X[r,c] - mean_c) / std_c.
// (A tiny epsilon guards against division by zero for a constant column.)
//
// WHY standardize: neural-net training behaves far better when inputs are
// well-scaled. He-initialized weights assume roughly unit-variance, zero-mean
// inputs; if one feature had a much larger scale it would dominate the dot
// products, push ReLU units into saturation/death, and make a single learning
// rate work poorly across features. Centering and scaling each feature keeps
// the first-layer pre-activations in a sane range and lets SGD converge with a
// single, stable learning rate. This mirrors what real pipelines do to inputs.
//
// Operates only on d.X; labels d.y are untouched. n_samples/n_features/n_classes
// are unchanged.
void dataset_standardize(Dataset& d);

// -----------------------------------------------------------------------------
// dataset_shuffle
// -----------------------------------------------------------------------------
// Randomly permute the sample ORDER of the dataset in place using a
// Fisher-Yates (a.k.a. Knuth) shuffle on the host. Each iteration swaps the
// current sample with a uniformly chosen earlier-or-equal sample; X rows and
// the corresponding y labels are swapped TOGETHER so that feature/label pairing
// is preserved (a whole row of n_features floats moves with its single label).
//
// WHY shuffle every epoch: mini-batch SGD assumes each batch is a roughly i.i.d.
// sample of the data. make_blobs lays samples out class-by-class (all of class
// 0, then all of class 1, ...). Without shuffling, early batches would contain
// only one class, producing biased, oscillating gradients. Re-shuffling at the
// start of each epoch decorrelates batch contents and improves convergence.
//
// Determinism: uses a std::mt19937_64 seeded with `seed`, so a given seed yields
// a fixed permutation. Callers typically vary the seed per epoch (e.g.
// base_seed + epoch) to get a different-but-reproducible order each epoch.
//
// Parameters:
//   d    : dataset to permute in place (both d.X rows and d.y entries).
//   seed : 64-bit seed for the shuffle RNG.
void dataset_shuffle(Dataset& d, unsigned long long seed);
