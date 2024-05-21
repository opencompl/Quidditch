
iree_cc_library(
    NAME
    executable
    HDRS
    "executable.h"
    SRCS
    "executable.c"
    DEPS
    snRuntime
    Quidditch::dispatch::dispatch
    iree::base
    iree::hal::local::executable_library
    PUBLIC
)
