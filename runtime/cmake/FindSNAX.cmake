# - Find SNAX library
# This module creates an imported target for the SNAX library
#
# Variables that can be set to guide the search:
#   SNAX_ROOT_DIR      - Root directory where SNAX is installed
#   SNAX_INCLUDE_DIR   - Directory containing SNAX headers
#   SNAX_LIBRARY       - Full path to SNAX static library
#
# Output variables:
#   SNAX_FOUND         - True if SNAX was found
#   SNAX_INCLUDE_DIRS  - SNAX include directories
#   SNAX_LIBRARIES     - SNAX library to link against
#
# Creates target:
#   SNAX::SNAX - Imported target for SNAX library

include(FindPackageHandleStandardArgs)

# Search for header files
find_path(SNAX_INCLUDE_DIR
    NAMES snrt.h
    HINTS ${SNAX_ROOT_DIR}/target/snitch_cluster/sw/runtime/rtl/src/
          ${SNAX_INCLUDE_DIR}
    DOC "Path to SNAX include directory"
)

find_path(SNAX_COMMON_INCLUDE_DIR
    NAMES snitch_cluster_defs.h
    HINTS ${SNAX_ROOT_DIR}/target/snitch_cluster/sw/runtime/common/
          ${SNAX_COMMON_INCLUDE_DIR}
    DOC "Path to SNAX common include directory"
)

find_path(SNAX_SW_INCLUDE_DIR
    NAMES alloc.h
    HINTS ${SNAX_ROOT_DIR}/sw/snRuntime/src/
          ${SNAX_SW_INCLUDE_DIR}
    DOC "Path to SNAX sw include directory"
)

find_path(SNAX_OMP_INCLUDE_DIR
    NAMES eu.h
    HINTS ${SNAX_ROOT_DIR}/sw/snRuntime/src/omp/
    ${SNAX_OMP_INCLUDE_DIR}
    DOC "Path to SNAX OMP include directory"
)

find_path(SNAX_API_OMP_INCLUDE_DIR
    NAMES eu_decls.h
    HINTS ${SNAX_ROOT_DIR}/sw/snRuntime/api/omp/
    ${SNAX_API_OMP_INCLUDE_DIR}
    DOC "Path to SNAX API OMP include directory"
)

# Search for library
find_library(SNAX_LIBRARY
    NAMES libsnRuntime.a
    HINTS ${SNAX_ROOT_DIR}/rtl-generic/build/
          ${SNAX_LIBRARY}
    DOC "Path to SNAX static library"
)

# Handle standard arguments
find_package_handle_standard_args(SNAX
  REQUIRED_VARS SNAX_LIBRARY SNAX_INCLUDE_DIR SNAX_COMMON_INCLUDE_DIR SNAX_SW_INCLUDE_DIR SNAX_OMP_INCLUDE_DIR SNAX_API_OMP_INCLUDE_DIR
)

if(SNAX_FOUND)
  set(SNAX_INCLUDE_DIRS ${SNAX_INCLUDE_DIR} ${SNAX_COMMON_INCLUDE_DIR} ${SNAX_SW_INCLUDE_DIR} ${SNAX_OMP_INCLUDE_DIR} ${SNAX_API_OMP_INCLUDE_DIR})
    set(SNAX_LIBRARIES ${SNAX_LIBRARY})

    set(SNAX_LINKER_SCRIPT "${SNAX_ROOT_DIR}/sw/snRuntime/base.ld")

    #if(SNAX_LINKER_SCRIPT AND NOT EXISTS "${SNAX_LINKER_SCRIPT}")
      #message(FATAL_ERROR "Linker script not found: ${SNAX_LINKER_SCRIPT}")
    #endif()

    if(NOT TARGET SNAX::SNAX)
        add_library(SNAX::SNAX STATIC IMPORTED)
        set_target_properties(SNAX::SNAX PROPERTIES
            IMPORTED_LOCATION "${SNAX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SNAX_API_OMP_INCLUDE_DIR};${SNAX_OMP_INCLUDE_DIR};${SNAX_SW_INCLUDE_DIR};${SNAX_COMMON_INCLUDE_DIR};${SNAX_INCLUDE_DIR}"
            INTERFACE_LINK_DIRECTORIES "${SNAX_ROOT_DIR}/target/snitch_cluster/sw/runtime/rtl-generic/"
            INTERFACE_LINK_OPTIONS "-T${SNAX_LINKER_SCRIPT}"
        )
    endif()
endif()

mark_as_advanced(SNAX_INCLUDE_DIR SNAX_LIBRARY)
