
iree_cc_library(
    NAME
    loader
    SRCS
    loader.c
    DEPS
    snRuntime
    iree::base
    Quidditch::executable::executable
    PUBLIC
)
