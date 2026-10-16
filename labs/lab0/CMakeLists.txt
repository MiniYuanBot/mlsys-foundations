cmake_minimum_required(VERSION 3.18)
project(lab0_minitensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -march=native lets the compiler use AVX2/FMA (x86) or NEON (Apple M-series)
# for your micro-kernel. Turn it off to see how much vectorisation matters.
option(MT_NATIVE "Optimise for the build machine's CPU" ON)
option(MT_SANITIZE "Build with AddressSanitizer + UBSan (GCC/Clang, not MinGW)" OFF)

find_package(Threads REQUIRED)

add_library(
  minitensor
  src/tensor_core.cpp
  src/tensor_student.cpp
  src/gemm_core.cpp
  src/gemm_student.cpp
  src/lazy_core.cpp
  src/lazy_student.cpp
)
target_include_directories(minitensor PUBLIC include)
target_link_libraries(minitensor PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(minitensor PUBLIC /W4 /permissive-)
  if(MT_NATIVE)
    target_compile_options(minitensor PUBLIC /arch:AVX2)
  endif()
  target_compile_options(minitensor PUBLIC /openmp:experimental) # enables `#pragma omp simd` only
else()
  target_compile_options(minitensor PUBLIC -Wall -Wextra)
  # Honour `#pragma omp simd` without pulling in the OpenMP runtime.
  target_compile_options(minitensor PUBLIC -fopenmp-simd)
  if(MT_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
      target_compile_options(minitensor PUBLIC -march=native)
    else()
      check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE) # Apple clang on arm64
      if(HAS_MCPU_NATIVE)
        target_compile_options(minitensor PUBLIC -mcpu=native)
      endif()
    endif()
  endif()
  if(MT_SANITIZE)
    target_compile_options(
      minitensor
      PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g
    )
    target_link_options(minitensor PUBLIC -fsanitize=address,undefined)
  endif()
endif()

enable_testing()
foreach(part a_tensor b_gemm c_lazy)
  add_executable(test_${part} tests/test_${part}.cpp)
  target_link_libraries(test_${part} PRIVATE minitensor)
endforeach()
add_test(NAME partA COMMAND test_a_tensor)
add_test(NAME partB COMMAND test_b_gemm)
add_test(NAME partC COMMAND test_c_lazy)
set_tests_properties(partA PROPERTIES LABELS partA)
set_tests_properties(partB PROPERTIES LABELS partB)
set_tests_properties(partC PROPERTIES LABELS partC)

add_executable(bench_gemm bench/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE minitensor)
add_executable(bench_fusion bench/bench_fusion.cpp)
target_link_libraries(bench_fusion PRIVATE minitensor)
