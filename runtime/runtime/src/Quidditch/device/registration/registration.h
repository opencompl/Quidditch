
#pragma once

#include <iree/base/status.h>
#include <iree/hal/driver_registry.h>

iree_status_t iree_hal_quidditch_driver_module_register(
    iree_hal_driver_registry_t *registry);
