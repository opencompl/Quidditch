
#pragma once

#include <iree/base/allocator.h>
#include <iree/base/config.h>
#include <iree/hal/local/executable_loader.h>

// Parameters configuring an iree_hal_sync_device_t.
// Must be initialized with iree_hal_sync_device_params_initialize prior to use.
typedef struct quidditch_device_params_t {
  // Total size of each block in the device shared block pool.
  // Larger sizes will lower overhead and ensure the heap isn't hit for
  // transient allocations while also increasing memory consumption.
  iree_host_size_t arena_block_size;
} quidditch_device_params_t;

// Initializes |out_params| to default values.
void quidditch_device_params_initialize(quidditch_device_params_t *out_params);

iree_status_t quidditch_device_create(iree_string_view_t identifier,
                                      const quidditch_device_params_t *params,
                                      iree_host_size_t loader_count,
                                      iree_hal_executable_loader_t **loaders,
                                      iree_hal_allocator_t *device_allocator,
                                      iree_allocator_t host_allocator,
                                      iree_hal_device_t **out_device);
