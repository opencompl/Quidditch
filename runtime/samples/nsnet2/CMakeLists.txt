
iree_turbine(SRC NsNet2.py DST ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc DTYPE "f32")
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc LLVM)

add_executable(NsNet2LLVM main.c)
target_link_libraries(
    NsNet2LLVM
    PRIVATE
    samples_util
    nsnet2_module
    snRuntime
)
