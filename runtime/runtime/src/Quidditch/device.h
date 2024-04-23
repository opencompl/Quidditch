
#pragma once

#include <iree/base/allocator.h>
#include <iree/base/config.h>
#include <iree/hal/local/executable_loader.h>

iree_status_t quidditch_device_create(
    iree_host_size_t loader_count, iree_hal_executable_loader_t **loaders,
    iree_hal_allocator_t *device_allocator, iree_allocator_t host_allocator,
    iree_hal_device_t **out_device);
