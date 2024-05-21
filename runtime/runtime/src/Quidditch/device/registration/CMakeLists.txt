iree_register_external_hal_driver(
        NAME
        Quidditch
        DRIVER_TARGET
        Quidditch::registration::registration
        REGISTER_FN
        iree_hal_quidditch_driver_module_register
)

iree_cc_library(
        NAME
        registration
        SRCS
        registration.c
        DEPS
        iree::base
        PUBLIC
)
