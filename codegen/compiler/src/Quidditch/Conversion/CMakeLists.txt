iree_add_all_subdirs()

iree_tablegen_library(
        NAME
        PassesIncGen
        TD_FILE
        "Passes.td"
        OUTS
        -name=Conversion --gen-pass-decls Passes.h.inc
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

iree_cc_library(
        NAME
        ConvertSnitchToLLVM
        SRCS
        "ConvertSnitchToLLVM.cpp"
        DEPS
        Quidditch::Dialect::Snitch::IR::QuidditchSnitchDialect
        MLIRAnalysis
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRSCFDialect
        MLIRTransforms
)
