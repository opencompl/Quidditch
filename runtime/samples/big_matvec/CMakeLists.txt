quidditch_module(SRC big_matvec.mlir ASSERT_XDSL)

add_executable(big_matvec_sample main.c)
target_link_libraries(
    big_matvec_sample
    PRIVATE
    samples_util
    big_matvec
    snRuntime
    Quidditch::dispatch::dispatch
)

