cmake_minimum_required(VERSION 3.21)
project(QuidditchCompiler LANGUAGES CXX C)

set(IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../iree" CACHE STRING "IREE source code path")
set(QUIDDITCH_TOOLCHAIN_ROOT "" CACHE STRING "Path to the Quidditch toolchain")

add_subdirectory(iree-configuration)

enable_testing()
include(CTest)

add_subdirectory(tests)
