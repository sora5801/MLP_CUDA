# =============================================================================
# CMakeLists.txt  —  cross-platform build for the MLP_CUDA study repo
# -----------------------------------------------------------------------------
# ROLE IN THE PROJECT
#   This file teaches CMake to treat the repo as a *CUDA-language* project and
#   produce a single executable (the `mlp` demo in src/main.cu) from all the
#   .cu translation units under src/. It is the portable build path that works
#   on BOTH Windows (MSVC host compiler + the CUDA toolkit) and Linux/WSL
#   (GCC/Clang host compiler + the CUDA toolkit). The hand-written Makefile is
#   the Linux/WSL-only alternative; this CMake build is the recommended one on
#   Windows where there is no `make`.
#
#   Prerequisite for either path: an installed NVIDIA CUDA Toolkit that ships
#   `nvcc`. CMake locates it automatically once we enable the CUDA "language"
#   below; you do not invoke nvcc by hand.
#
# WHY "CUDA as a first-class language" matters
#   Modern CMake (>= 3.8, and we require 3.18 for the architecture handling
#   below) understands CUDA the same way it understands C or C++: list `CUDA`
#   in project(LANGUAGES ...) and every `.cu` file is compiled with nvcc, which
#   in turn drives the host C++ compiler for the host-side code. We therefore
#   do NOT need find_package(CUDA) (the old, deprecated module) or any custom
#   compile rules — CMake wires up include paths, separable compilation, and
#   linking for us.
# =============================================================================

# -----------------------------------------------------------------------------
# Minimum CMake version.
#   3.18 is required because that is the release that introduced the
#   CUDA_ARCHITECTURES target property and the special "native" value (see the
#   CMAKE_CUDA_ARCHITECTURES block far below). Asking for it up front makes the
#   failure mode "your CMake is too old" instead of a confusing later error.
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)

# -----------------------------------------------------------------------------
# Declare the project and, crucially, the languages it uses.
#   LANGUAGES CXX CUDA  ->  enable BOTH the C++ toolchain (for any plain host
#   code and because nvcc needs a host compiler) AND the CUDA toolchain. This
#   single line is what makes `.cu` files compile with nvcc automatically.
#   If the CUDA toolkit is missing, CMake fails here with a clear message.
# -----------------------------------------------------------------------------
project(MLP_CUDA LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# Language standards.
#   The spec pins C++14 (§0.6). We set it for BOTH languages:
#     * CMAKE_CUDA_STANDARD 14  -> the C++ dialect nvcc uses for device/host
#       code in .cu files (nvcc compiles C++; "CUDA standard" here means the
#       C++ standard applied to CUDA sources).
#     * CMAKE_CXX_STANDARD  14  -> the dialect for any pure .cpp host code.
#   *_STANDARD_REQUIRED ON makes an unsupported standard a hard error rather
#   than silently downgrading.
# -----------------------------------------------------------------------------
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Target GPU architectures (compute capability / SASS generation).
#   nvcc must be told which GPU architecture(s) to emit code for. CMake exposes
#   this via CMAKE_CUDA_ARCHITECTURES. We default to "native", which (CMake
#   3.24+) detects the compute capability of the GPU actually present in the
#   build machine and compiles only for it — the fast, no-config choice for a
#   study repo built and run on one machine.
#
#   How to override:
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native   # this default
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75        # e.g. Turing
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="70;75;86" # fat binary
#
#   We only set the default if the user did not pass one on the command line,
#   so an explicit -D... always wins. ("native" needs CMake >= 3.24; on older
#   3.18–3.23 pass an explicit number like 75 instead.)
# -----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# -----------------------------------------------------------------------------
# Header search path.
#   All public declarations live in include/ (common.cuh, matrix.cuh,
#   kernels.cuh, mlp.cuh, dataset.cuh). Adding it here lets every .cu source
#   use   #include "kernels.cuh"   without relative "../include/" prefixes.
#   This mirrors the Makefile's  -Iinclude  flag.
# -----------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# -----------------------------------------------------------------------------
# Gather the CUDA translation units.
#   GLOB collects every src/*.cu (matrix.cu, kernels.cu, mlp.cu, dataset.cu,
#   main.cu). CONFIGURE_DEPENDS asks CMake to re-run globbing when the set of
#   matching files changes, so adding a new .cu does not require manually
#   editing this file. (Note: globbing is convenient for a small study repo;
#   large projects often prefer listing sources explicitly so CMake need not
#   re-glob — that trade-off is left as an exercise.)
# -----------------------------------------------------------------------------
file(GLOB MLP_CUDA_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

# -----------------------------------------------------------------------------
# The one executable for the demo.
#   Because CUDA is an enabled language, CMake compiles each .cu with nvcc and
#   links them into the `mlp` binary (mlp.exe on Windows, mlp on Linux). The
#   entry point is main() in src/main.cu.
# -----------------------------------------------------------------------------
add_executable(mlp ${MLP_CUDA_SOURCES})

# -----------------------------------------------------------------------------
# Optional convenience target:  cmake --build build --target run
#   Builds (via the dependency on `mlp`) and then executes the demo. Purely a
#   developer nicety mirroring the Makefile's `run` target; harmless if unused.
# -----------------------------------------------------------------------------
add_custom_target(run
    COMMAND mlp
    DEPENDS mlp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running the MLP_CUDA demo executable")
