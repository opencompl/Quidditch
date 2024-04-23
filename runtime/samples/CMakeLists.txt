add_executable(IREE_HelloWorld main.c)
target_link_libraries(
        IREE_HelloWorld
        PRIVATE
        iree::base
        iree::vm
        iree::modules::hal::types
        iree::hal::local::loaders::static_library_loader
        Quidditch::device
)

