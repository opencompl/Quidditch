iree_add_all_subdirs()

iree_cc_library(
        NAME
        QuidditchSnitchDialect
        HDRS
        "QuidditchSnitchDialect.h"
        "QuidditchSnitchOps.h"
        TEXTUAL_HDRS
        "QuidditchSnitchDialect.cpp.inc"
        "QuidditchSnitchDialect.h.inc"
        "QuidditchSnitchOps.cpp.inc"
        "QuidditchSnitchOps.h.inc"
        SRCS
        "QuidditchSnitchDialect.cpp"
        "QuidditchSnitchOps.cpp"
        DEPS
        ::QuidditchSnitchDialectGen
        ::QuidditchSnitchOpsGen
        LLVMSupport
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
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
