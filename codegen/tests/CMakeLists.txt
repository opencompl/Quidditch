# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Python3 COMPONENTS Interpreter REQUIRED)

cmake_path(GET Python3_EXECUTABLE PARENT_PATH python_bin_dir)
cmake_path(GET python_bin_dir PARENT_PATH python_bin_dir)
find_program(XDSL_OPT_PATH xdsl-opt
        PATHS ${python_bin_dir}
        PATH_SUFFIXES "bin"
        NO_DEFAULT_PATH
        DOC "Path of the xdsl-opt file"
        REQUIRED
)

set(QUIDDITCH_TEST_DEPENDS
        FileCheck count not split-file
        iree-compile quidditch-opt
)

set(QUIDDITCH_LIT_SITE_CFG_OUT ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${QUIDDITCH_LIT_SITE_CFG_OUT}
        @ONLY
)

set(QUIDDITCH_BIN_DIR ${CMAKE_BINARY_DIR}/bin)
set(QUIDDITCH_LIT_CFG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
cmake_path(RELATIVE_PATH QUIDDITCH_LIT_SITE_CFG_OUT BASE_DIRECTORY ${QUIDDITCH_BIN_DIR})
cmake_path(RELATIVE_PATH QUIDDITCH_LIT_CFG BASE_DIRECTORY ${QUIDDITCH_BIN_DIR})
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/quidditch-lit.py.in
        ${QUIDDITCH_BIN_DIR}/quidditch-lit.py
        @ONLY
)

set(_LIT_PATH_ARGS)
foreach (_TOOL IN LISTS QUIDDITCH_TEST_DEPENDS)
    list(APPEND _LIT_PATH_ARGS "--path" "$<TARGET_FILE_DIR:${_TOOL}>")
endforeach ()

add_custom_target(quidditch-test-depends DEPENDS ${QUIDDITCH_TEST_DEPENDS})

add_custom_target(check-quidditch
        COMMAND
        ${Python3_EXECUTABLE}
        ${QUIDDITCH_BIN_DIR}/quidditch-lit.py
        "${CMAKE_CURRENT_BINARY_DIR}"
        -v ${_LIT_PATH_ARGS}
    DEPENDS quidditch-test-depends)
