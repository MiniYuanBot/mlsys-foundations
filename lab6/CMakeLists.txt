cmake_minimum_required(VERSION 3.20)
project(lab6_collectives LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

add_library(coll STATIC src/coll_core.cpp src/coll_student.cpp src/train_core.cpp src/train_student.cpp)
target_include_directories(coll PUBLIC include)
target_link_libraries(coll PUBLIC MPI::MPI_CXX Threads::Threads)
target_compile_options(coll PUBLIC -Wall -Wextra -O3 -march=native -fopenmp-simd)

# Open MPI refuses to start more ranks than cores unless told to.
set(LAB6_MPIEXEC_FLAGS "" CACHE STRING "extra mpiexec flags")
if(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Open MPI" AND LAB6_MPIEXEC_FLAGS STREQUAL "")
  set(LAB6_MPIEXEC_FLAGS "--oversubscribe")
endif()
separate_arguments(LAB6_MPIEXEC_FLAGS_LIST UNIX_COMMAND "${LAB6_MPIEXEC_FLAGS}")

enable_testing()
foreach(t coll train)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE coll)
endforeach()
foreach(np 1 2 3 4 8)
  add_test(NAME coll_np${np} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${LAB6_MPIEXEC_FLAGS_LIST}
           $<TARGET_FILE:test_coll>)
  set_tests_properties(coll_np${np} PROPERTIES LABELS coll TIMEOUT 120)
endforeach()
foreach(np 1 2 4)
  add_test(NAME train_np${np} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${LAB6_MPIEXEC_FLAGS_LIST}
           $<TARGET_FILE:test_train>)
  set_tests_properties(train_np${np} PROPERTIES LABELS train TIMEOUT 120)
endforeach()

add_executable(bench_coll bench/bench_coll.cpp)
target_link_libraries(bench_coll PRIVATE coll)
add_executable(train bench/train.cpp)
target_link_libraries(train PRIVATE coll)
