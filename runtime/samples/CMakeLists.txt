include(quidditch_module)

quidditch_module(SRC simple_add.mlir)

add_subdirectory(util)

add_executable(IREE_HelloWorld main.c)
target_link_libraries(
    IREE_HelloWorld
    PRIVATE
    samples_util
    simple_add_module
    snRuntime
)

