
# Disable for now to be more accepting of compilers and not needing to patch or contribute to upstream immediately.
set(IREE_ENABLE_WERROR_FLAG OFF)
set(IREE_ENABLE_TESTS OFF)
set(IREE_CMAKE_PLUGIN_PATHS "${CMAKE_CURRENT_LIST_DIR}/../compiler")
set(LLVM_TARGETS_TO_BUILD "RISCV" CACHE STRING "")
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_TARGET_BACKEND_DEFAULTS OFF)
set(IREE_TARGET_BACKEND_LLVM_CPU ON)
add_subdirectory(${IREE_SOURCE_DIR} iree)
