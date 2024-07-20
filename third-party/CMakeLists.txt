# --- ceres-solver
set(BUILD_TESTING OFF CACHE BOOL "disable tests build for ceres" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "do not build ceres examples" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "do not build ceres benchmarks" FORCE)
set(USE_CUDA OFF CACHE BOOL "disable CUDA support for ceres" FORCE)
add_subdirectory(ceres-solver)

# --- sophus
set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "Do not build Sophus tests" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Do not build Sophus python bindings" FORCE)
add_subdirectory(Sophus)
