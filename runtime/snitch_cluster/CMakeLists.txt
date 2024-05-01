project(snRuntime C ASM)

find_package(Python3 REQUIRED)
find_program(bender_path NAMES bender REQUIRED)

set(SNITCH_RUNTIME_TARGET "rtl" CACHE STRING "Snitch runtime target to use (one of banshee or rtl)")

set(snRuntimeSrc ${SNITCH_CLUSTER_SOURCE_DIR})
set(header_dir ${CMAKE_CURRENT_BINARY_DIR}/cluster_gen)
set(config_file ${snRuntimeSrc}/target/snitch_cluster/cfg/default.hjson)
set(runtime_dir ${snRuntimeSrc}/target/snitch_cluster/sw/runtime/${SNITCH_RUNTIME_TARGET})

execute_process(COMMAND ${bender_path} path register_interface
    OUTPUT_VARIABLE reggen_base_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${snRuntimeSrc})
set(reggen_path ${reggen_base_path}/vendor/lowrisc_opentitan/util/regtool.py)

add_custom_command(OUTPUT "${header_dir}/snitch_cluster_peripheral.h"
    COMMAND ${Python3_EXECUTABLE} ${reggen_path} -D
    -o "${header_dir}/snitch_cluster_peripheral.h"
    ${snRuntimeSrc}/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.hjson
    WORKING_DIRECTORY ${snRuntimeSrc}
    DEPENDS ${snRuntimeSrc}/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.hjson)
add_custom_target(snitch_cluster_peripheral.h DEPENDS "${header_dir}/snitch_cluster_peripheral.h")

add_custom_target(cluster_gen)
macro(run_cluster_gen input_file)
  set(filename ${input_file})
  cmake_path(GET filename FILENAME filename)
  cmake_path(GET filename STEM LAST_ONLY filename)
  add_custom_command(OUTPUT "${header_dir}/${filename}"
      COMMAND ${Python3_EXECUTABLE} ${snRuntimeSrc}/util/clustergen.py
      -c ${config_file} --outdir ${header_dir} --template ${input_file}
      DEPENDS ${config_file} ${input_file}
  )
  add_custom_target(${filename}-target DEPENDS "${header_dir}/${filename}")
  add_dependencies(cluster_gen ${filename}-target)
endmacro()

run_cluster_gen(${snRuntimeSrc}/target/snitch_cluster/sw/runtime/common/snitch_cluster_addrmap.h.tpl)
run_cluster_gen(${snRuntimeSrc}/target/snitch_cluster/sw/runtime/common/snitch_cluster_cfg.h.tpl)


add_library(snRuntime
    ${runtime_dir}/src/snitch_cluster_start.S
    ${runtime_dir}/src/snrt.c
)
target_include_directories(snRuntime PRIVATE
    ${snRuntimeSrc}/target/snitch_cluster/sw/runtime/common/
    ${snRuntimeSrc}/sw/snRuntime/src
    ${snRuntimeSrc}/sw/snRuntime/src/omp
    ${header_dir}
    PUBLIC
    ${snRuntimeSrc}/sw/snRuntime/api
    ${snRuntimeSrc}/sw/snRuntime/api/omp
)
add_dependencies(snRuntime cluster_gen snitch_cluster_peripheral.h)
target_link_options(snRuntime INTERFACE -Tbase.ld)
target_link_directories(snRuntime
    INTERFACE
    ${snRuntimeSrc}/sw/snRuntime
    ${runtime_dir}
)
