# Many IDEs set this for configuring the toolchain. If present, use it as the
# default value inferred for the pulp toolchain.
set(normal_var "${CMAKE_C_COMPILER}")
cmake_path(GET normal_var PARENT_PATH pulp_toolchain_root_default)
cmake_path(GET pulp_toolchain_root_default PARENT_PATH pulp_toolchain_root_default)

set(PULP_TOOLCHAIN_ROOT ${pulp_toolchain_root_default} CACHE PATH "")

set(CMAKE_SYSTEM_NAME               Generic CACHE INTERNAL "")
set(CMAKE_SYSTEM_PROCESSOR          riscv32 CACHE INTERNAL "")

# Without that flag CMake is not able to pass test compilation check
set(CMAKE_TRY_COMPILE_TARGET_TYPE   STATIC_LIBRARY CACHE INTERNAL "")

set(CMAKE_AR                        ${PULP_TOOLCHAIN_ROOT}/bin/llvm-ar${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_ASM_COMPILER              ${PULP_TOOLCHAIN_ROOT}/bin/clang${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_C_COMPILER                ${PULP_TOOLCHAIN_ROOT}/bin/clang${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_CXX_COMPILER              ${PULP_TOOLCHAIN_ROOT}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_OBJCOPY                   ${PULP_TOOLCHAIN_ROOT}/bin/llvm-objcopy${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_RANLIB                    ${PULP_TOOLCHAIN_ROOT}/bin/llvm-ranlib${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_SIZE                      ${PULP_TOOLCHAIN_ROOT}/bin/llvm-size${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_STRIP                     ${PULP_TOOLCHAIN_ROOT}/bin/llvm-strip${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")

set(codegen_opts "-mcpu=snitch -menable-experimental-extensions")
set(CMAKE_ASM_FLAGS                   "${codegen_opts}" CACHE INTERNAL "")
set(CMAKE_C_FLAGS                   "${codegen_opts} -ffunction-sections -fdata-sections" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS                 "${CMAKE_C_FLAGS} -fno-exceptions -fno-rtti" CACHE INTERNAL "")
set(linker_flags "-Wl,--gc-sections -Wl,-z,norelro -fuse-ld=lld -nostartfiles")
set(CMAKE_EXE_LINKER_FLAGS ${linker_flags} CACHE INTERNAL "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
