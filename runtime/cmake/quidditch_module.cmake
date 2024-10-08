file(GLOB cmake_build_dirs LIST_DIRECTORIES true "${PROJECT_SOURCE_DIR}/../codegen/cmake-build-*")
find_program(IREE_COMPILE_PATH iree-compile
    PATHS ${QUIDDITCH_CODEGEN_BUILD_DIR}
    "${PROJECT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/../codegen"
    # Conventional build directories created by IDEs or humans.
    "${PROJECT_SOURCE_DIR}/../codegen/build"
    ${cmake_build_dirs}
    PATH_SUFFIXES
    "iree-configuration/iree/tools"
    REQUIRED
    NO_DEFAULT_PATH
    DOC "Path to the build directory of <root>/codegen"
)
message(STATUS "Using iree-compile at ${IREE_COMPILE_PATH}")
if (IREE_COMPILE_PATH MATCHES "iree-configuration/iree/tools/iree-compile$")
  string(LENGTH "/iree-configuration/iree/tools/iree-compile" length_suffix)
  string(LENGTH ${IREE_COMPILE_PATH} length_path)
  math(EXPR length "${length_path} - ${length_suffix}")
  string(SUBSTRING ${IREE_COMPILE_PATH} 0 ${length} maybe_build_dir)
  if (EXISTS "${maybe_build_dir}/CMakeCache.txt")
    message(STATUS "Detected iree-compile within another cmake build")
    # This makes the assumption that the other cmake build uses the same cmake
    # executable.
    add_custom_target(iree-keep-up-to-date
        BYPRODUCTS ${IREE_COMPILE_PATH}
        COMMAND ${CMAKE_COMMAND} --build ${maybe_build_dir} --target iree-compile
        COMMENT "Updating iree-compile"
        USES_TERMINAL
    )
  endif ()
endif ()

find_package(Python3 REQUIRED)
cmake_path(GET Python3_EXECUTABLE PARENT_PATH python_bin_dir)
cmake_path(GET python_bin_dir PARENT_PATH python_bin_dir)
find_program(XDSL_OPT_PATH xdsl-opt
    PATHS ${python_bin_dir}
    PATH_SUFFIXES "bin"
    NO_DEFAULT_PATH
    DOC "Path of the xdsl-opt file"
    REQUIRED
)

# Compiles an IREE MLIR file given by the 'SRC' argument to a static library
# suitable for use with quidditch's and IREE's static library loader.
# Uses the Quidditch backend by default unless the 'LLVM' flag is given.
# Options listed under 'FLAGS' are passed to 'iree-compile' directly.
# Targets or files listed after 'DEPENDS' are added as input dependencies to the
# the compile command.
#
# The resulting library is the source file's name with the extension removed and
# '_module' appended.
function(quidditch_module)
  cmake_parse_arguments(_RULE "LLVM;ASSERT_XDSL" "SRC;DST" "FLAGS;DEPENDS" ${ARGN})

  set(_MLIR_SRC "${_RULE_SRC}")
  if (NOT _RULE_DST)
    cmake_path(GET _MLIR_SRC STEM _RULE_DST)
    set(_RULE_DST "${_RULE_DST}")
  endif ()

  get_filename_component(_MLIR_SRC "${_MLIR_SRC}" REALPATH)
  set(_O_QUIDDITCH_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_DST}/${_RULE_DST}.o")
  set(_O_LLVM_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_DST}/${_RULE_DST}_llvm.o")
  set(_H_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_DST}/${_RULE_DST}_module.h")
  set(_MODULE_NAME "${_RULE_DST}_module")

  set(_COMPILER_ARGS ${_RULE_FLAGS})
  list(APPEND _COMPILER_ARGS "--iree-vm-bytecode-module-strip-source-map=true")
  list(APPEND _COMPILER_ARGS "--iree-vm-emit-polyglot-zip=false")
  list(APPEND _COMPILER_ARGS "--iree-input-type=auto")
  # TODO: xDSL cannot deal with anything but f64 right now.
  list(APPEND _COMPILER_ARGS "--iree-input-demote-f64-to-f32=0")

  set(_OUTPUT_FILES "${_H_FILE_NAME}")
  set(_OBJECT_FILES)

  set(_EXTRA_DEPENDS ${_RULE_DEPENDS})
  if (NOT _RULE_LLVM)
    list(APPEND _COMPILER_ARGS "--iree-hal-target-backends=quidditch")
    list(APPEND _COMPILER_ARGS "--iree-quidditch-static-library-output-path=${_O_QUIDDITCH_FILE_NAME}")
    list(APPEND _COMPILER_ARGS "--iree-quidditch-xdsl-opt-path=${XDSL_OPT_PATH}")
    list(APPEND _COMPILER_ARGS "--iree-quidditch-toolchain-root=${QUIDDITCH_TOOLCHAIN_ROOT}")
    if (_RULE_ASSERT_XDSL)
      list(APPEND _COMPILER_ARGS "--iree-quidditch-assert-compiled=true")
    endif ()

    list(APPEND _EXTRA_DEPENDS "${XDSL_OPT_PATH}")
    list(APPEND _EXTRA_DEPENDS "${QUIDDITCH_TOOLCHAIN_ROOT}/bin/pulp-as")

    list(APPEND _OUTPUT_FILES "${_O_QUIDDITCH_FILE_NAME}")
    list(APPEND _OBJECT_FILES "${_O_QUIDDITCH_FILE_NAME}")
    if (NOT _RULE_ASSERT_XDSL)
      list(APPEND _OBJECT_FILES "${_O_LLVM_FILE_NAME}")
    endif ()

    string(REPLACE ".o" ".h" _STATIC_HDR_PATH "${_O_QUIDDITCH_FILE_NAME}")
    list(APPEND _OUTPUT_FILES "${_STATIC_HDR_PATH}")
  else ()
    set(_O_LLVM_FILE_NAME ${_O_QUIDDITCH_FILE_NAME})
    list(APPEND _OBJECT_FILES "${_O_LLVM_FILE_NAME}")
  endif ()

  string(REPLACE ".o" ".h" _STATIC_HDR_PATH "${_O_LLVM_FILE_NAME}")
  list(APPEND _OUTPUT_FILES "${_STATIC_HDR_PATH}")

  if (NOT _RULE_ASSERT_XDSL)
    list(APPEND _COMPILER_ARGS "--iree-hal-target-backends=llvm-cpu")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-debug-symbols=true")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-triple=riscv32-unknown-elf")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-cpu=generic-rv32")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-cpu-features=+m,+f,+d,+zfh")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-abi=ilp32d")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-float-abi=hard")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-link-embedded=false")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-link-static")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-number-of-threads=8")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-static-library-output-path=${_O_LLVM_FILE_NAME}")

    list(APPEND _OUTPUT_FILES "${_O_LLVM_FILE_NAME}")
  endif ()

  list(APPEND _COMPILER_ARGS "--output-format=vm-c")
  list(APPEND _COMPILER_ARGS "--iree-vm-target-index-bits=32")
  list(APPEND _COMPILER_ARGS "${_MLIR_SRC}")
  list(APPEND _COMPILER_ARGS "-o")
  list(APPEND _COMPILER_ARGS "${_H_FILE_NAME}")

  add_custom_command(
      OUTPUT ${_OUTPUT_FILES}
      COMMAND ${IREE_COMPILE_PATH} ${_COMPILER_ARGS}
      DEPENDS ${IREE_COMPILE_PATH} ${_MLIR_SRC} ${_EXTRA_DEPENDS}
  )

  cmake_path(REPLACE_EXTENSION _H_FILE_NAME LAST_ONLY "c" OUTPUT_VARIABLE _C_FILE_NAME)
  file(CONFIGURE OUTPUT ${_C_FILE_NAME} CONTENT [[
#define EMITC_IMPLEMENTATION
#include "@_MODULE_NAME@.h"
]] @ONLY)
  add_library(${_RULE_DST}
      STATIC ${_C_FILE_NAME} ${_OBJECT_FILES}
      ${_H_FILE_NAME}
  )
  target_link_libraries(${_RULE_DST}
      PUBLIC
      iree::vm
  )
  target_include_directories(${_RULE_DST} INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_DST})
endfunction()

# Use iree-turbine to convert a PyTorch model to MLIR.
# 'SRC' should be the path to the python file, 'DTYPE' one of "f32" or "F64" and
# 'DST' the path to the output file.
# The python script should take one positional argument, which is the 'DST'
# argument and output the MLIR file there. Additionally the 'DTYPE' flag is
# communicated via a `--dtype=` flag.
macro(iree_turbine)
  cmake_parse_arguments(_RULE "" "SRC;DTYPE;DST" "" ${ARGN})

  cmake_path(GET _RULE_SRC STEM filename)
  cmake_path(ABSOLUTE_PATH _RULE_SRC BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE source_path)

  add_custom_command(
      OUTPUT ${_RULE_DST}
      COMMAND ${Python3_EXECUTABLE} ${source_path} ${_RULE_DST} --dtype=${_RULE_DTYPE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${_RULE_SRC}
      COMMENT "Translating ${filename} using iree-turbine"
  )
endmacro()
