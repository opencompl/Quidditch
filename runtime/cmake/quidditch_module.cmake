set(IREE_COMPILE_PATH "${QUIDDITCH_CODEGEN_BUILD_DIR}/iree-configuration/iree/tools/iree-compile")

function(quidditch_module)
    cmake_parse_arguments(_RULE "" "SRC" "FLAGS;DEPENDS" ${ARGN})

    set(_MLIR_SRC "${_RULE_SRC}")

    cmake_path(GET _MLIR_SRC STEM filename)

    set(_COMPILER_ARGS ${_RULE_FLAGS})
    list(APPEND _COMPILER_ARGS "--iree-hal-target-backends=llvm-cpu")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-debug-symbols=false")
    list(APPEND _COMPILER_ARGS "--iree-vm-bytecode-module-strip-source-map=true")
    list(APPEND _COMPILER_ARGS "--iree-vm-emit-polyglot-zip=false")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-triple=riscv32-unknown-elf")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-cpu=generic-rv32")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-cpu-features=+m,+f,+d,+zfh")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-target-abi=ilp32d")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-link-embedded=false")
    list(APPEND _COMPILER_ARGS "--iree-input-type=auto")

    set(_O_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${filename}/${filename}.o")
    set(_H_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${filename}/${filename}_module.h")
    set(_MODULE_NAME "${filename}_module")

    get_filename_component(_MLIR_SRC "${_MLIR_SRC}" REALPATH)
    list(APPEND _COMPILER_ARGS "--output-format=vm-c")
    list(APPEND _COMPILER_ARGS "--iree-vm-target-index-bits=32")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-link-static")
    list(APPEND _COMPILER_ARGS "--iree-llvmcpu-static-library-output-path=${_O_FILE_NAME}")
    list(APPEND _COMPILER_ARGS "${_MLIR_SRC}")
    list(APPEND _COMPILER_ARGS "-o")
    list(APPEND _COMPILER_ARGS "${_H_FILE_NAME}")

    set(_OUTPUT_FILES "${_H_FILE_NAME}")
    string(REPLACE ".o" ".h" _STATIC_HDR_PATH "${_O_FILE_NAME}")
    list(APPEND _OUTPUT_FILES "${_O_FILE_NAME}" "${_STATIC_HDR_PATH}")

    add_custom_command(
            OUTPUT ${_OUTPUT_FILES}
            COMMAND ${IREE_COMPILE_PATH} ${_COMPILER_ARGS}
            DEPENDS ${IREE_COMPILE_PATH} ${_MLIR_SRC}
    )

    add_library(${_MODULE_NAME}
            STATIC
            ${_H_FILE_NAME} ${_O_FILE_NAME}
    )
    target_include_directories(${_MODULE_NAME} INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/${filename})
    target_compile_definitions(${_MODULE_NAME} PUBLIC EMITC_IMPLEMENTATION=\"${_H_FILE_NAME}\")
    set_target_properties(
            ${_MODULE_NAME}
            PROPERTIES
            LINKER_LANGUAGE C
    )
endfunction()
