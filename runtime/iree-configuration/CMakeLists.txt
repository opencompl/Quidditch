
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ENABLE_THREADING OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_WERROR_FLAG OFF)

# Include the IREE runtime as part of this build.
set(IREE_CMAKE_PLUGIN_PATHS "${CMAKE_CURRENT_LIST_DIR}/../runtime")
add_subdirectory(${IREE_SOURCE_DIR} iree EXCLUDE_FROM_ALL)

