include(quidditch_module)

quidditch_module(SRC simple_add.mlir)

add_executable(IREE_HelloWorld main.c)
target_link_libraries(
    IREE_HelloWorld
    PRIVATE
    snRuntime
    iree::base
    iree::vm
    iree::modules::hal
    iree::modules::hal::types
    iree::hal::local::local
    iree::hal::local::loaders::static_library_loader
    Quidditch::device
    simple_add_module
)

