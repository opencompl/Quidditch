#include "device.h"

#include <iree/hal/local/local_executable_cache.h>

typedef struct quidditch_device_t {
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;

  iree_host_size_t loader_count;
  iree_hal_executable_loader_t *loaders[];
} quidditch_device_t;

static const iree_hal_device_vtable_t quidditch_device_vtable;

static quidditch_device_t *cast_device(iree_hal_device_t *device) {
  IREE_HAL_ASSERT_TYPE(device, &quidditch_device_vtable);
  return (quidditch_device_t *)device;
}

static void destroy(iree_hal_device_t *base_device) {
  quidditch_device_t *device = cast_device(base_device);
  iree_hal_allocator_release(device->device_allocator);
  for (iree_host_size_t i = 0; i < device->loader_count; i++) {
    iree_hal_executable_loader_release(device->loaders[i]);
  }
}

static iree_allocator_t host_allocator(iree_hal_device_t *base_device) {
  return cast_device(base_device)->host_allocator;
}

static iree_status_t create_executable_cache(
    iree_hal_device_t *base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t **out_executable_cache) {
  quidditch_device_t *device = cast_device(base_device);
  return iree_hal_local_executable_cache_create(
      identifier, /*worker_capacity=*/1, device->loader_count, device->loaders,
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

static const iree_hal_device_vtable_t quidditch_device_vtable = {
    .destroy = destroy,
    .host_allocator = host_allocator,
    .create_executable_cache = create_executable_cache,
};

iree_status_t quidditch_device_create(iree_host_size_t loader_count,
                                      iree_hal_executable_loader_t **loaders,
                                      iree_hal_allocator_t *device_allocator,
                                      iree_allocator_t host_allocator,
                                      iree_hal_device_t **out_device) {
  IREE_ASSERT_ARGUMENT(loaders || loader_count == 0);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_device);

  // Set the out device to null in case any steps fail.
  *out_device = NULL;

  quidditch_device_t *device = NULL;
  iree_host_size_t allocation_size =
      sizeof(quidditch_device_t) +
      loader_count * sizeof(*device->loaders);  // NOLINT(*-sizeof-expression)
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void **)&device));
  memset(device, 0, allocation_size);
  iree_hal_resource_initialize(&quidditch_device_vtable, &device->resource);
  device->host_allocator = host_allocator;
  device->device_allocator = device_allocator;

  // Make sure to increase the ref counts of any entities we reference.
  iree_hal_allocator_retain(device_allocator);
  device->loader_count = loader_count;
  for (iree_host_size_t i = 0; i < device->loader_count; ++i) {
    device->loaders[i] = loaders[i];
    iree_hal_executable_loader_retain(device->loaders[i]);
  }

  *out_device = (iree_hal_device_t *)device;
  return iree_ok_status();
}
