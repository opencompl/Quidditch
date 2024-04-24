
# Disable for now to be more accepting of compilers and not needing to patch or contribute to upstream immediately.
set(IREE_ENABLE_WERROR_FLAG OFF)
set(IREE_CMAKE_PLUGIN_PATHS "${CMAKE_CURRENT_LIST_DIR}/../compiler")
add_subdirectory(${IREE_SOURCE_DIR} iree)
