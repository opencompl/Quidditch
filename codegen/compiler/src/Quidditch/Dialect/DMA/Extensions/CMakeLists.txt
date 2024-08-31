
iree_cc_library(
        NAME
        DMACoreSpecializationOpInterfaceImpl
        HDRS
        "DMACoreSpecializationOpInterfaceImpl.h"
        SRCS
        "DMACoreSpecializationOpInterfaceImpl.cpp"
        DEPS
        Quidditch::Dialect::Snitch::IR::QuidditchSnitchDialect
        Quidditch::Dialect::DMA::IR::DMADialect
)
