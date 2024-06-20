add_executable(quidditch-opt quidditch-opt.cpp)
target_link_libraries(quidditch-opt
        PRIVATE
        MLIROptLib
        Quidditch::Target::Passes
        iree::compiler::Dialect::LinalgExt::IR
        iree::compiler::Tools::init_passes_and_dialects
)
