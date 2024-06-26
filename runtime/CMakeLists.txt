cmake_minimum_required(VERSION 3.21)

project(QuidditchRuntime LANGUAGES C ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEFINFO ON)

set(IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../iree" CACHE PATH "IREE source code path")
set(SNITCH_CLUSTER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../snitch_cluster" CACHE PATH "snitch_cluster source code path")
set(QUIDDITCH_CODEGEN_BUILD_DIR "" CACHE PATH "CMake generation directory with a compiled 'iree-compile'")

list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake/
)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)

# Configuration for our target that must be present in all code using IREE.
add_definitions(-DIREE_PLATFORM_GENERIC)
add_definitions(-D_ISOC11_SOURCE)
add_definitions(-DIREE_USER_CONFIG_H="${CMAKE_CURRENT_LIST_DIR}/iree-configuration/config.h")
add_subdirectory(iree-configuration)

add_subdirectory(snitch_cluster SYSTEM EXCLUDE_FROM_ALL)

add_subdirectory(samples)

enable_testing()
add_subdirectory(tests)
