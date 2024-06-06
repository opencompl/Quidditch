
iree_cc_library(
    NAME
    command_buffer
    HDRS
    "command_buffer.h"
    SRCS
    "command_buffer.c"
    DEPS
    iree::base
    iree::base::internal
    iree::base::internal::cpu
    iree::base::internal::fpu_state
    iree::hal
    Quidditch::executable::executable
    PUBLIC
)
