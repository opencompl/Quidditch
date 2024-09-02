iree_tablegen_library(
        NAME
        PassesIncGen
        TD_FILE
        "Passes.td"
        OUTS
        -name=Transforms --gen-pass-decls Passes.h.inc
)

iree_cc_library(
        NAME
        Passes
        HDRS
        "Passes.h"
        "Passes.h.inc"
        SRCS
        "LegalizeDMAOperations.cpp"
        DEPS
        ::PassesIncGen
        Quidditch::Dialect::DMA::IR::DMADialect
        Quidditch::Dialect::Snitch::IR::QuidditchSnitchDialect
        MLIRIR
        MLIRAffineDialect
        MLIRArithDialect
        MLIRSCFDialect
)
