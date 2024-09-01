add_executable(quidditch-opt quidditch-opt.cpp)
target_link_libraries(quidditch-opt
        PRIVATE
        MLIROptLib
        Quidditch::Conversion::ConvertSnitchToLLVM
        Quidditch::Conversion::ConvertToRISCV
        Quidditch::Dialect::DMA::Extensions::DMACoreSpecializationOpInterfaceImpl
        Quidditch::Dialect::SnitchDMA::IR::SnitchDMADialect
        Quidditch::Dialect::Snitch::IR::QuidditchSnitchDialect
        Quidditch::Dialect::Snitch::Transforms::Passes
        Quidditch::Target::Passes
        iree::compiler::Dialect::LinalgExt::IR
        iree::compiler::Tools::init_passes_and_dialects
)
