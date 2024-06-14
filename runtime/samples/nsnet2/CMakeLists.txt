
iree_turbine(SRC NsNet2.py DST ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc DTYPE "f64")
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc)

add_executable(NsNet2LLVM main.c)
target_link_libraries(
    NsNet2LLVM
    PRIVATE
    samples_util
    nsnet2_module
    snRuntime
    Quidditch::dispatch::dispatch
)
