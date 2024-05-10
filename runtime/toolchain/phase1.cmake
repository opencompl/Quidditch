set(CMAKE_C_COMPILER clang CACHE STRING "")
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "")
set(LLVM_DEFAULT_TARGET_TRIPLE riscv32-unknown-unknown-elf CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS ON CACHE BOOL "")
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(LLVM_ENABLE_PROJECTS clang;lld CACHE STRING "")
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_USE_CRT_RELEASE "MT" CACHE STRING "")
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
set(LLVM_USE_LLD ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS llvm-dis;llvm-ar;llvm-ranlib;llvm-nm;llvm-objcopy;llvm-objdump;llvm-rc;llvm-profdata;llvm-symbolizer;llvm-strip;llvm-cov;llvm-cxxfilt;llvm-size;llvm-undname;llvm-addr2line;llvm-as;llvm-cat;llvm-cxxdump;llvm-dis;llvm-dwp;llvm-mc;llvm-mca;llvm-mt;obj2yaml;yaml2obj;llvm-link;llvm-lto;llvm-lto2;llvm-readobj;llvm-shlib;llvm-split;llvm-strings;sancov;llvm-dwarfdump;llvm-cat;llvm-diff CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "RISCV" CACHE STRING "")

set(LIBCLANG_BUILD_STATIC ON CACHE BOOL "")
set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
set(CLANG_DEFAULT_OBJCOPY llvm-objcopy CACHE STRING "")
set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")
set(CLANG_DEFAULT_CXX_STDLIB libc++ CACHE STRING "")
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")

file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)
file(WRITE ${CMAKE_INSTALL_PREFIX}/bin/riscv32-unknown-unknown-elf.cfg [=[
-march=rv32imafdzfh
-mabi=ilp32d
-mcmodel=medany
-static
-ftls-model=local-exec
-fno-common
-mcpu=generic-rv32
-nostartfiles
# For libc++ and friends.
--sysroot=<CFGDIR>/..
-Wno-unused-command-line-argument
-ffunction-sections
-fdata-sections
-Wl,--gc-sections
-Wl,-z,norelro
-fvisibility=hidden
]=])
file(WRITE ${CMAKE_INSTALL_PREFIX}/bin/clang++.cfg [=[
-nostdlib++
-lc++
-lc++abi
]=])
file(WRITE ${CMAKE_INSTALL_PREFIX}/bin/clang++.cfg [=[
-nostdlib++
-lc++
-lc++abi
]=])
file(WRITE ${CMAKE_INSTALL_PREFIX}/ToolchainFile.cmake [=[
set(QUIDDITCH_TOOLCHAIN_ROOT ${CMAKE_CURRENT_LIST_DIR})

# Without that flag CMake is not able to pass test compilation check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY CACHE INTERNAL "")

set(CMAKE_SYSTEM_NAME               Generic CACHE INTERNAL "")
set(CMAKE_SYSTEM_PROCESSOR          riscv32 CACHE INTERNAL "")

set(CMAKE_AR ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/llvm-ar${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_ASM_COMPILER ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/clang${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_C_COMPILER ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/clang${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_CXX_COMPILER ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_OBJCOPY ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/llvm-objcopy${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_RANLIB ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/llvm-ranlib${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_SIZE ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/llvm-size${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_STRIP ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/llvm-strip${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_CROSSCOMPILING_EMULATOR ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/snitch_cluster.vlt CACHE INTERNAL "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
]=])
