iree_add_all_subdirs()

iree_cc_library(
        NAME
        SnitchDMADialect
        HDRS
        "SnitchDMADialect.h"
        "SnitchDMAOps.h"
        TEXTUAL_HDRS
        "SnitchDMAAttrs.cpp.inc"
        "SnitchDMAAttrs.h.inc"
        "SnitchDMADialect.cpp.inc"
        "SnitchDMADialect.h.inc"
        "SnitchDMAOps.cpp.inc"
        "SnitchDMAOps.h.inc"
        "SnitchDMATypes.cpp.inc"
        "SnitchDMATypes.h.inc"
        SRCS
        "SnitchDMAAttrs.cpp"
        "SnitchDMADialect.cpp"
        "SnitchDMAOps.cpp"
        "SnitchDMATypes.cpp"
        DEPS
        ::SnitchDMAAttrsGen
        ::SnitchDMADialectGen
        ::SnitchDMAOpsGen
        ::SnitchDMATypesGen
        LLVMSupport
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
        PUBLIC
)

iree_tablegen_library(
        NAME
        SnitchDMAOpsGen
        TD_FILE
        "SnitchDMAOps.td"
        OUTS
        --gen-op-decls SnitchDMAOps.h.inc
        --gen-op-defs SnitchDMAOps.cpp.inc
)

iree_tablegen_library(
        NAME
        SnitchDMADialectGen
        TD_FILE
        "SnitchDMADialect.td"
        OUTS
        --gen-dialect-decls SnitchDMADialect.h.inc
        --gen-dialect-defs SnitchDMADialect.cpp.inc
)

iree_tablegen_library(
        NAME
        SnitchDMAAttrsGen
        TD_FILE
        "SnitchDMAAttrs.td"
        OUTS
        --gen-attrdef-decls SnitchDMAAttrs.h.inc
        --gen-attrdef-defs SnitchDMAAttrs.cpp.inc
)

iree_tablegen_library(
        NAME
        SnitchDMATypesGen
        TD_FILE
        "SnitchDMATypes.td"
        OUTS
        --gen-typedef-decls SnitchDMATypes.h.inc
        --gen-typedef-defs SnitchDMATypes.cpp.inc
)
