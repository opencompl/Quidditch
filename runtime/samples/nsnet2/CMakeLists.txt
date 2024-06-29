
iree_turbine(SRC NsNet2.py DST ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc DTYPE "f64")
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc DST nsnet2)
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc LLVM DST nsnet2_llvm)
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc DST nsnet2st)
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc LLVM DST nsnet2st_llvm)

add_library(nsnet2_util nsnet2_util.c)
target_link_libraries(nsnet2_util
    PRIVATE
    samples_util
    snRuntimeInterface
    Quidditch::dispatch::dispatch
)
target_include_directories(nsnet2_util INTERFACE ${CMAKE_CURRENT_LIST_DIR})

macro(create_experiment_variant)
  cmake_parse_arguments(_RULE "PRECOMMIT;NIGHTLY" "TARGET;IREE_MODULE;QUERY_FUNC" "" ${ARGN})

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c "\
#include <${_RULE_IREE_MODULE}.h>

#include \"nsnet2_util.h\"

int main() {
  return run_nsnet2_experiment(${_RULE_QUERY_FUNC});
}
")
  add_executable(${_RULE_TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c)
  target_link_libraries(
      ${_RULE_TARGET}
      PRIVATE
      nsnet2_util
      ${_RULE_IREE_MODULE}
      snRuntime
  )
endmacro()

create_experiment_variant(
    TARGET NsNet2
    IREE_MODULE nsnet2
    QUERY_FUNC "quidditch_compiled_ns_net2_linked_quidditch_library_query"
)
create_experiment_variant(
    TARGET NsNet2LLVM
    IREE_MODULE nsnet2_llvm
    QUERY_FUNC "compiled_ns_net2_linked_llvm_cpu_library_query"
)
create_experiment_variant(
    TARGET NsNet2ST
    IREE_MODULE nsnet2st
    QUERY_FUNC "quidditch_compiled_ns_net2_linked_quidditch_library_query"
)
create_experiment_variant(
    TARGET NsNet2STLLVM
    IREE_MODULE nsnet2st_llvm
    QUERY_FUNC "compiled_ns_net2_linked_llvm_cpu_library_query"
)
