cmake_minimum_required(VERSION 3.21)
project(Quidditch C CXX)

set(QUIDDITCH_TOOLCHAIN_FILE "" CACHE STRING "Path to the quidditch toolchain's toolchain file")
set(PULP_CLANG_PATH "" CACHE STRING "Path to clang within the pulp toolchain")
set(OVERRIDE_VENV OFF CACHE BOOL "")

if (NOT DEFINED ENV{VIRTUAL_ENV} AND NOT ${OVERRIDE_VENV})
  message(FATAL_ERROR [[
  Please use a virtual environment as this cmake script will call pip install.
  Reconfigure with -DOVERRIDE_VENV=ON if this is a false positive or you don't
  care.
]])
  return()
endif ()

set(Python3_FIND_VIRTUALENV "FIRST")
find_package(Python3 REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip install -r ${CMAKE_CURRENT_LIST_DIR}/requirements.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND_ERROR_IS_FATAL ANY)

find_program(CARGO_PATH cargo REQUIRED DOC "Path to cargo required to build bender")

include(ExternalProject)

macro(append_if_defined list variable)
  if (DEFINED ${variable})
    list(APPEND ${list} "-D${variable}=${${variable}}")
  endif ()
endmacro()

set(host_cmake_args)
append_if_defined(host_cmake_args CMAKE_BUILD_TYPE)
append_if_defined(host_cmake_args CMAKE_C_COMPILER)
append_if_defined(host_cmake_args CMAKE_CXX_COMPILER)
append_if_defined(host_cmake_args CMAKE_C_COMPILER_LAUNCHER)
append_if_defined(host_cmake_args CMAKE_CXX_COMPILER_LAUNCHER)
ExternalProject_Add(codegen
    CMAKE_ARGS
    -DIREE_ENABLE_LLD=ON
    -DPULP_CLANG_PATH=${PULP_CLANG_PATH}
    ${host_cmake_args}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/codegen
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/codegen
    BUILD_COMMAND ${CMAKE_COMMAND}
    --build ${CMAKE_CURRENT_BINARY_DIR}/codegen
    --target quidditch-test-depends
    TEST_COMMAND ""
    INSTALL_COMMAND ""
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
    USES_TERMINAL_TEST ON
)

ExternalProject_Add(bender
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/bender
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
    ${CARGO_PATH} build
    --release
    --target-dir=${CMAKE_CURRENT_BINARY_DIR}/bender
    --manifest-path=${CMAKE_CURRENT_LIST_DIR}/bender/Cargo.toml
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bender
    INSTALL_COMMAND ""
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
    USES_TERMINAL_TEST ON
)

set(runtime_cmake_args)
append_if_defined(runtime_cmake_args CMAKE_BUILD_TYPE)
append_if_defined(runtime_cmake_args CMAKE_C_COMPILER_LAUNCHER)
append_if_defined(runtime_cmake_args CMAKE_CXX_COMPILER_LAUNCHER)
ExternalProject_Add(runtime
    CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${QUIDDITCH_TOOLCHAIN_FILE}
    -DPULP_CLANG_PATH=${PULP_CLANG_PATH}
    -DQUIDDITCH_CODEGEN_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/codegen
    -DBENDER_PATH=${CMAKE_CURRENT_BINARY_DIR}/bender/release/bender
    ${runtime_cmake_args}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/runtime
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/runtime
    INSTALL_COMMAND ""
    TEST_COMMAND ""
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
    USES_TERMINAL_TEST ON
)
add_dependencies(runtime codegen bender)

enable_testing()
include(ProcessorCount)
ProcessorCount(N)

include(CTest)
set(CMAKE_CTEST_ARGUMENTS "--verbose")
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/CTestCustom.cmake CONTENT [[
set(CTEST_CUSTOM_PRE_TEST "@CMAKE_COMMAND@ --build @CMAKE_BINARY_DIR@")
]] @ONLY)
# lit uses maximum parallelism by default.
add_test(NAME
    codegen-tests
    COMMAND
    ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/codegen --target check-quidditch
)
add_test(NAME
    runtime-tests
    COMMAND
    ${CMAKE_CTEST_COMMAND} --extra-verbose -j${N}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/runtime
)
