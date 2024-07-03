
iree_turbine(SRC NsNet2.py DST ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc DTYPE "f64")
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc DST nsnet2)
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc LLVM DST nsnet2_llvm)

add_library(nsnet2_util nsnet2_util.c)
target_link_libraries(nsnet2_util
    PRIVATE
    samples_util
    snRuntimeInterface
    Quidditch::dispatch::dispatch
)
target_include_directories(nsnet2_util INTERFACE ${CMAKE_CURRENT_LIST_DIR})

macro(create_experiment_variant target_name iree_module query_func)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.c "\
#include <${iree_module}.h>

#include \"nsnet2_util.h\"

int main() {
  return run_nsnet2_experiment(${query_func});
}
")
  add_executable(${target_name} ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.c)
  target_link_libraries(
      ${target_name}
      PRIVATE
      nsnet2_util
      ${iree_module}
      snRuntime
  )
endmacro()

create_experiment_variant(NsNet2 nsnet2 "quidditch_compiled_ns_net2_linked_quidditch_library_query")
create_experiment_variant(NsNet2LLVM nsnet2_llvm "compiled_ns_net2_linked_llvm_cpu_library_query")
