cmake_minimum_required(VERSION 3.21)
project(QuidditchCompiler LANGUAGES CXX C)

set(IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../iree" CACHE STRING "IREE source code path")

add_subdirectory(iree-configuration)
