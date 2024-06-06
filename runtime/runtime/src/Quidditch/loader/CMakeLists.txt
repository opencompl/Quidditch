
iree_cc_library(
    NAME
    loader
    SRCS
    loader.c
    DEPS
    iree::base
    Quidditch::executable::executable
    PUBLIC
)
