cmake_minimum_required(VERSION 3.21)
project(Quidditch C CXX)

set(QUIDDITCH_TOOLCHAIN_FILE "" CACHE FILEPATH "Path to the quidditch toolchain's toolchain file")
set(OVERRIDE_VENV OFF CACHE BOOL "")
cmake_path(REMOVE_FILENAME QUIDDITCH_TOOLCHAIN_FILE OUTPUT_VARIABLE QUIDDITCH_TOOLCHAIN_ROOT)

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
    -DQUIDDITCH_TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT}
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
    BUILD_ALWAYS ON
)

set(runtime_cmake_args)
append_if_defined(runtime_cmake_args CMAKE_BUILD_TYPE)
append_if_defined(runtime_cmake_args CMAKE_C_COMPILER_LAUNCHER)
append_if_defined(runtime_cmake_args CMAKE_CXX_COMPILER_LAUNCHER)
ExternalProject_Add(runtime
    CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${QUIDDITCH_TOOLCHAIN_FILE}
    -DQUIDDITCH_CODEGEN_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/codegen
    ${runtime_cmake_args}
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/runtime
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/runtime
    INSTALL_COMMAND ""
    TEST_COMMAND ""
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
    USES_TERMINAL_TEST ON
    BUILD_ALWAYS ON
)
add_dependencies(runtime codegen)

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
