cmake_minimum_required(VERSION 3.18)
project(mlsys_tensor_layout LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(tensor_lab main.cpp student.cpp)
enable_testing()
add_test(NAME student_contract COMMAND tensor_lab --test)
add_test(NAME supplied_reference COMMAND tensor_lab --reference-test)

