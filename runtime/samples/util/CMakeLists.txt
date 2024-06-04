
add_library(samples_util run_model.c)
target_link_libraries(
    samples_util
    PUBLIC
    iree::base
    iree::vm
    PRIVATE
    snRuntime
    iree::modules::hal
    iree::modules::hal::types
    iree::hal::local::local
    Quidditch::device::device
    Quidditch::loader::loader
)
