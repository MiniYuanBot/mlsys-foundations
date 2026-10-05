cmake_minimum_required(VERSION 3.18)
project(mlsys_host_ring LANGUAGES CXX)
find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(ring_allreduce ring_allreduce.cpp)
target_compile_features(ring_allreduce PRIVATE cxx_std_17)
target_link_libraries(ring_allreduce PRIVATE MPI::MPI_CXX)
enable_testing()
add_test(NAME reference_two_ranks COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ring_allreduce> ${MPIEXEC_POSTFLAGS} --reference)
add_test(NAME student_one_rank COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ring_allreduce> ${MPIEXEC_POSTFLAGS})
add_test(NAME student_two_ranks COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ring_allreduce> ${MPIEXEC_POSTFLAGS})
add_test(NAME student_three_ranks COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ring_allreduce> ${MPIEXEC_POSTFLAGS})
set_tests_properties(reference_two_ranks student_one_rank student_two_ranks student_three_ranks PROPERTIES TIMEOUT 20)
