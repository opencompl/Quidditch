
iree_cc_library(
    NAME
    dispatch
    SRCS
    dispatch.c
    DEPS
    snRuntimeInterface
    iree::base
    PUBLIC
)
