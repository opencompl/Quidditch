
iree_cc_library(
    NAME
    dispatch
    SRCS
    dispatch.c
    DEPS
    snRuntime
    iree::base
    PUBLIC
)
