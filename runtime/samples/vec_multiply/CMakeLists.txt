quidditch_module(SRC simple_add.mlir ASSERT_XDSL)

add_executable(vec_multiply main.c)
target_link_libraries(
    vec_multiply
    PRIVATE
    samples_util
    simple_add
    snRuntime
    Quidditch::dispatch::dispatch
)

