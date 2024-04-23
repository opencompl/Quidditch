cmake_minimum_required(VERSION 3.28)
project(QuidditchRuntime LANGUAGES C ASM)

set(CMAKE_CXX_STANDARD 17)

set(IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../iree" CACHE STRING "IREE source code path")

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)

# Configuration for our target that must be present in all code using IREE.
add_definitions(-DIREE_PLATFORM_GENERIC)
add_definitions(-DIREE_USER_CONFIG_H="${CMAKE_CURRENT_LIST_DIR}/iree-configuration/config.h")
add_subdirectory(iree-configuration)

add_subdirectory(samples)
