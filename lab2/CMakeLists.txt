cmake_minimum_required(VERSION 3.24)
# Must come before project(). RTX 40xx = sm_89; override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(lab2_attention LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Part C1: a shared library loaded from Python with ctypes (attn/cuda.py).
add_library(flash_mma SHARED csrc/flash_mma_student.cu)
target_compile_options(flash_mma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
option(L2_PTXAS_VERBOSE "Print per-kernel register and shared-memory usage" OFF)
if(L2_PTXAS_VERBOSE)
  target_compile_options(flash_mma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()
