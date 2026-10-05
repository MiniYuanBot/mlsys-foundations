cmake_minimum_required(VERSION 3.24)
# Must come before project(), which otherwise picks an old default arch.
# RTX 40xx = sm_89. Override with -DCMAKE_CUDA_ARCHITECTURES=80 (A100), 90 (H100), ...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(lab1_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(l1kernels STATIC
  src/core.cu src/basics_student.cu src/sgemm_student.cu src/hgemm_student.cu)
target_include_directories(l1kernels PUBLIC include)
target_link_libraries(l1kernels PUBLIC CUDA::cudart CUDA::cublas)
# -lineinfo maps SASS back to source lines in Nsight Compute at no runtime cost.
# --ptxas-options=-v prints registers / shared memory / spills per kernel at build time.
target_compile_options(l1kernels PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
option(L1_PTXAS_VERBOSE "Print per-kernel register and shared-memory usage" OFF)
if(L1_PTXAS_VERBOSE)
  target_compile_options(l1kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()

enable_testing()
foreach(part a_basics b_sgemm c_hgemm)
  add_executable(test_${part} tests/test_${part}.cu)
  target_link_libraries(test_${part} PRIVATE l1kernels)
endforeach()
add_test(NAME partA COMMAND test_a_basics)
add_test(NAME partB COMMAND test_b_sgemm)
add_test(NAME partC COMMAND test_c_hgemm)
set_tests_properties(partA PROPERTIES LABELS partA)
set_tests_properties(partB PROPERTIES LABELS partB)
set_tests_properties(partC PROPERTIES LABELS partC)

add_executable(bench_basics bench/bench_basics.cu)
target_link_libraries(bench_basics PRIVATE l1kernels)
add_executable(bench_gemm bench/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE l1kernels)
