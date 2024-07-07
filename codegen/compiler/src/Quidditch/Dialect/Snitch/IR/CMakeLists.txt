iree_add_all_subdirs()

iree_cc_library(
        NAME
        QuidditchSnitchDialect
        HDRS
        "QuidditchSnitchDialect.h"
        "QuidditchSnitchOps.h"
        TEXTUAL_HDRS
        "QuidditchSnitchAttrs.cpp.inc"
        "QuidditchSnitchAttrs.h.inc"
        "QuidditchSnitchDialect.cpp.inc"
        "QuidditchSnitchDialect.h.inc"
        "QuidditchSnitchOps.cpp.inc"
        "QuidditchSnitchOps.h.inc"
        "QuidditchSnitchTypes.cpp.inc"
        "QuidditchSnitchTypes.h.inc"
        SRCS
        "QuidditchSnitchAttrs.cpp"
        "QuidditchSnitchDialect.cpp"
        "QuidditchSnitchOps.cpp"
        "QuidditchSnitchTypes.cpp"
        DEPS
        ::QuidditchSnitchAttrsGen
        ::QuidditchSnitchDialectGen
        ::QuidditchSnitchOpsGen
        ::QuidditchSnitchTypesGen
        LLVMSupport
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
        iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
        PUBLIC
)

iree_tablegen_library(
        NAME
        QuidditchSnitchOpsGen
        TD_FILE
        "QuidditchSnitchOps.td"
        OUTS
        --gen-op-decls QuidditchSnitchOps.h.inc
        --gen-op-defs QuidditchSnitchOps.cpp.inc
)

iree_tablegen_library(
        NAME
        QuidditchSnitchDialectGen
        TD_FILE
        "QuidditchSnitchDialect.td"
        OUTS
        --gen-dialect-decls QuidditchSnitchDialect.h.inc
        --gen-dialect-defs QuidditchSnitchDialect.cpp.inc
)

iree_tablegen_library(
        NAME
        QuidditchSnitchAttrsGen
        TD_FILE
        "QuidditchSnitchAttrs.td"
        OUTS
        --gen-attrdef-decls QuidditchSnitchAttrs.h.inc
        --gen-attrdef-defs QuidditchSnitchAttrs.cpp.inc
)

iree_tablegen_library(
        NAME
        QuidditchSnitchTypesGen
        TD_FILE
        "QuidditchSnitchTypes.td"
        OUTS
        --gen-typedef-decls QuidditchSnitchTypes.h.inc
        --gen-typedef-defs QuidditchSnitchTypes.cpp.inc
)
