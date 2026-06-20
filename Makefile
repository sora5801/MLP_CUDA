# =============================================================================
# Makefile  —  nvcc build for the MLP_CUDA study repo (Linux / WSL)
# -----------------------------------------------------------------------------
# Role in the repo:
#   This is the simplest of the two build systems shipped with the project
#   (the other being CMakeLists.txt for cross-platform / Windows + MSVC use).
#   It drives the NVIDIA CUDA compiler `nvcc` to turn every translation unit in
#   src/*.cu into an object file under build/, then links them into a single
#   executable build/mlp.
#
# Mental model for a CUDA build (didactic):
#   A .cu file contains BOTH host (CPU) code and device (GPU) code. `nvcc` is a
#   compiler *driver*: it splits each .cu into a host part (handed to the system
#   C++ compiler, e.g. g++) and a device part (compiled to GPU assembly, PTX,
#   and/or SASS), then stitches the results into one object file. From the
#   Makefile's point of view this looks just like a normal C/C++ build:
#   compile each source to a .o, then link the .o files together. The CUDA
#   runtime library that the launchers/macros depend on (cudaMalloc, cudaMemcpy,
#   kernel<<<>>> launches, etc.) is linked in automatically by `nvcc`, which is
#   why we use `nvcc` as the linker too rather than calling g++ directly.
#
# Usage:
#   make            # or `make all` — build build/mlp
#   make run        # build (if needed) then execute build/mlp
#   make clean      # remove the build/ directory and its artifacts
#   make ARCH=sm_75 # build targeting a specific GPU architecture (see ARCH note)
# =============================================================================


# -----------------------------------------------------------------------------
# Toolchain.
#   `?=` is a conditional assignment: it only sets NVCC if it is not already
#   defined in the environment or on the command line. This lets a user point
#   at a non-default compiler, e.g. `make NVCC=/usr/local/cuda-12/bin/nvcc`,
#   without editing this file.
# -----------------------------------------------------------------------------
NVCC ?= nvcc


# -----------------------------------------------------------------------------
# ARCH note — choosing the GPU compute capability.
#   By default we do NOT pass `-arch`, so `nvcc` uses its built-in default
#   target. That keeps the build portable across CUDA toolkit versions, but the
#   default may be older than your GPU and can emit deprecation warnings.
#
#   To target a specific GPU, pass its compute capability as `ARCH`, e.g.:
#       make ARCH=sm_75   # Turing  (RTX 20xx, GTX 16xx)
#       make ARCH=sm_86   # Ampere  (RTX 30xx)
#       make ARCH=sm_89   # Ada     (RTX 40xx)
#   `sm_XX` selects the real GPU architecture the code is compiled (SASS) for.
#
#   ARCH defaults to empty. The conditional below appends `-arch=<ARCH>` to the
#   compiler flags only when ARCH is non-empty, so the unset case stays clean.
# -----------------------------------------------------------------------------
ARCH ?=


# -----------------------------------------------------------------------------
# Compiler / linker flags.
#   -O2          : optimization level 2 for the host code (good default).
#   -std=c++14   : the C++ dialect mandated by the build spec (§0.6). nvcc
#                  forwards this to the host compiler and uses it for the host
#                  side of each .cu translation unit.
#   -Iinclude    : add ./include to the header search path so sources can do
#                  `#include "common.cuh"`, `#include "mlp.cuh"`, etc. without
#                  relative paths.
# -----------------------------------------------------------------------------
NVCCFLAGS = -O2 -std=c++14 -Iinclude

# If the user supplied ARCH (e.g. `make ARCH=sm_86`), append the architecture
# flag. `ifneq (,$(ARCH))` reads as "if ARCH is not equal to empty".
ifneq (,$(ARCH))
NVCCFLAGS += -arch=$(ARCH)
endif


# -----------------------------------------------------------------------------
# Directory and file layout.
#   SRC_DIR   : where the .cu translation units live.
#   BUILD_DIR : where object files and the final binary are written. Kept out
#               of source control (see .gitignore) and removable via `clean`.
#   TARGET    : the final linked executable.
#
#   SOURCES   : every .cu file in src/. `$(wildcard ...)` expands at parse time
#               to the matching paths, so adding a new src/*.cu file is picked
#               up automatically with no Makefile edit.
#   OBJECTS   : the corresponding object files under build/. The substitution
#               `$(SOURCES:src/%.cu=build/%.o)` rewrites each `src/NAME.cu` into
#               `build/NAME.o`, preserving the base name.
# -----------------------------------------------------------------------------
SRC_DIR   = src
BUILD_DIR = build
TARGET    = $(BUILD_DIR)/mlp

SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)


# -----------------------------------------------------------------------------
# .PHONY declares targets that are NOT real files. Without this, a stray file
# named "all"/"run"/"clean" in the directory would make `make` think the target
# is already up to date and skip the recipe. Marking them phony forces the
# recipe to run every time it is requested.
# -----------------------------------------------------------------------------
.PHONY: all run clean


# -----------------------------------------------------------------------------
# Default goal: `make` with no arguments builds the executable.
#   Listed first so it is the goal `make` picks when invoked bare.
# -----------------------------------------------------------------------------
all: $(TARGET)


# -----------------------------------------------------------------------------
# Link rule: combine all object files into the final executable.
#   Prerequisites: every $(OBJECTS). If any object is newer than $(TARGET)
#   (or the target is missing), this recipe re-runs.
#   Recipe variables:
#     $@  = the target being built          (build/mlp)
#     $^  = ALL prerequisites, space-joined  (all the build/*.o)
#   We invoke `nvcc` (not g++) as the linker so it pulls in the CUDA runtime
#   and device-code stubs automatically.
# -----------------------------------------------------------------------------
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@


# -----------------------------------------------------------------------------
# Compile rule (pattern rule): turn src/NAME.cu into build/NAME.o.
#   `build/%.o: src/%.cu` matches any object/source pair sharing the stem `%`.
#   Prerequisite `| $(BUILD_DIR)` is an ORDER-ONLY prerequisite (note the `|`):
#   it guarantees the build/ directory exists before compiling, but changes to
#   the directory's timestamp do NOT force a recompile (directory mtimes bump
#   whenever any file inside changes, which would otherwise rebuild everything).
#   Recipe variables:
#     $<  = the FIRST prerequisite (the .cu source)
#     $@  = the target object file
#   `-c` tells nvcc to compile-only (emit a .o), not link.
# -----------------------------------------------------------------------------
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


# -----------------------------------------------------------------------------
# Create the build directory on demand.
#   This is the order-only prerequisite target referenced above. `mkdir -p`
#   is idempotent: it succeeds whether or not the directory already exists.
# -----------------------------------------------------------------------------
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


# -----------------------------------------------------------------------------
# `make run`: build the executable if needed, then run it.
#   Depends on $(TARGET) so a stale binary is rebuilt first. We run it via its
#   relative path; the program is self-contained (it generates its own
#   synthetic dataset), so no arguments are required.
# -----------------------------------------------------------------------------
run: $(TARGET)
	./$(TARGET)


# -----------------------------------------------------------------------------
# `make clean`: remove all build artifacts.
#   `-rf` so it never errors when build/ is already absent. This deletes the
#   whole build/ tree (objects + the linked binary) — source files are untouched.
# -----------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
