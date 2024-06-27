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
        "PromoteToL1.cpp"
        "LowerL1Allocations.cpp"
        "SpecializeDMACode.cpp"
        DEPS
        ::PassesIncGen
        Quidditch::Dialect::Snitch::IR::QuidditchSnitchDialect
        MLIRIR
        MLIRLinalgDialect
        MLIRBufferizationDialect
)
