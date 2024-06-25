iree_add_all_subdirs()

iree_tablegen_library(
        NAME
        PassesIncGen
        TD_FILE
        "Passes.td"
        OUTS
        --gen-pass-decls Passes.h.inc
)

iree_cc_library(
        NAME
        ConvertToRISCV
        HDRS
        "Passes.h"
        "Passes.h.inc"
        SRCS
        "ConvertToRISCV.cpp"
        DEPS
        ::PassesIncGen
        Quidditch::Dialect::Snitch::IR::QuidditchSnitchDialect
        MLIRFuncDialect
        MLIRIR
)
