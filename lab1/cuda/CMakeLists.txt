cmake_minimum_required(VERSION 3.24)

# Override with -DCMAKE_CUDA_ARCHITECTURES=80 for A100, 86 for RTX 30xx, etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "GPU architectures for the lab")
endif()
project(mlsys_gemm_lab LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.0 REQUIRED)
add_executable(gemm_lab main.cu baselines.cu student.cu)
target_compile_features(gemm_lab PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(gemm_lab PROPERTIES CUDA_EXTENSIONS OFF CXX_EXTENSIONS OFF)
target_link_libraries(gemm_lab PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(gemm_lab PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

enable_testing()
# An untouched student.cu MUST make this test fail. No skip-as-pass behavior.
add_test(NAME student_gemm_correctness COMMAND gemm_lab --test)
