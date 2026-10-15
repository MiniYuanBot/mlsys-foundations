cmake_minimum_required(VERSION 3.24)
project(lab3_tinycomp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The compiler itself is plain C++; CUDA is only used through NVRTC (runtime
# compilation) and the driver API, so no nvcc is needed to build this lab.
find_package(CUDAToolkit REQUIRED)

add_library(
  tinycomp
  STATIC
  src/ir.cpp
  src/plan_core.cpp
  src/codegen_core.cpp
  src/runtime.cpp
  src/interp_student.cpp
  src/codegen_student.cpp
  src/fusion_student.cpp
  src/search_student.cpp
)
target_include_directories(tinycomp PUBLIC include)
target_link_libraries(tinycomp PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
target_compile_options(tinycomp PRIVATE -Wall -Wextra)

enable_testing()
foreach(part a b c)
  add_executable(test_${part} tests/test_${part}.cpp)
  target_link_libraries(test_${part} PRIVATE tinycomp)
  add_test(NAME part${part} COMMAND test_${part})
endforeach()
set_tests_properties(parta PROPERTIES LABELS partA)
set_tests_properties(partb PROPERTIES LABELS partB)
set_tests_properties(partc PROPERTIES LABELS partC)

add_executable(tc_dump tools/tc_dump.cpp)
target_link_libraries(tc_dump PRIVATE tinycomp)
foreach(b fusion search)
  add_executable(bench_${b} bench/bench_${b}.cpp)
  target_link_libraries(bench_${b} PRIVATE tinycomp)
endforeach()
