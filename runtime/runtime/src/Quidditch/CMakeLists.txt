add_subdirectory(registration)

iree_cc_library(
        NAME
        device
        SRCS
        device.c
        DEPS
        iree::base
        PUBLIC
)
