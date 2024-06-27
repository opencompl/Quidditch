
iree_cc_library(
    NAME
    executable
    HDRS
    "executable.h"
    "executable_library.h"
    SRCS
    "executable.c"
    "executable_library.c"
    DEPS
    snRuntimeInterface
    Quidditch::dispatch::dispatch
    iree::base
    iree::hal::local::executable_library
    PUBLIC
)
