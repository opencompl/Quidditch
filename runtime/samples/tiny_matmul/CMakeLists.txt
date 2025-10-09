quidditch_module(SRC tiny_matmul.mlir ASSERT_XDSL)

add_executable(my_tiny_matmul main.c)
target_link_libraries(
    my_tiny_matmul
    PRIVATE
    samples_util
    tiny_matmul
    snRuntime
    Quidditch::dispatch::dispatch
)
