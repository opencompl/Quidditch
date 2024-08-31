iree_add_all_subdirs()

iree_cc_library(
        NAME
        DMADialect
        HDRS
        "DMADialect.h"
        "DMAOps.h"
        TEXTUAL_HDRS
        "DMAAttrs.cpp.inc"
        "DMAAttrs.h.inc"
        "DMADialect.cpp.inc"
        "DMADialect.h.inc"
        "DMAOps.cpp.inc"
        "DMAOps.h.inc"
        "DMATypes.cpp.inc"
        "DMATypes.h.inc"
        SRCS
        "DMAAttrs.cpp"
        "DMADialect.cpp"
        "DMAOps.cpp"
        "DMATypes.cpp"
        DEPS
        ::DMAAttrsGen
        ::DMADialectGen
        ::DMAOpsGen
        ::DMATypesGen
        LLVMSupport
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
        PUBLIC
)

iree_tablegen_library(
        NAME
        DMAOpsGen
        TD_FILE
        "DMAOps.td"
        OUTS
        --gen-op-decls DMAOps.h.inc
        --gen-op-defs DMAOps.cpp.inc
)

iree_tablegen_library(
        NAME
        DMADialectGen
        TD_FILE
        "DMADialect.td"
        OUTS
        --gen-dialect-decls DMADialect.h.inc
        --gen-dialect-defs DMADialect.cpp.inc
)

iree_tablegen_library(
        NAME
        DMAAttrsGen
        TD_FILE
        "DMAAttrs.td"
        OUTS
        --gen-attrdef-decls DMAAttrs.h.inc
        --gen-attrdef-defs DMAAttrs.cpp.inc
)

iree_tablegen_library(
        NAME
        DMATypesGen
        TD_FILE
        "DMATypes.td"
        OUTS
        --gen-typedef-decls DMATypes.h.inc
        --gen-typedef-defs DMATypes.cpp.inc
)
