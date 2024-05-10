#include "device.h"

#include <iree/hal/local/local_executable_cache.h>
#include <iree/hal/local/local_pipeline_layout.h>

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

static iree_hal_allocator_t *device_allocator(iree_hal_device_t *base_device) {
  return cast_device(base_device)->device_allocator;
}

static iree_status_t create_executable_cache(
    iree_hal_device_t *base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t **out_executable_cache) {
  quidditch_device_t *device = cast_device(base_device);
  return iree_hal_local_executable_cache_create(
      identifier, /*worker_capacity=*/1, device->loader_count, device->loaders,
      iree_hal_device_host_allocator(base_device), out_executable_cache);
}

static iree_status_t query_i64(iree_hal_device_t *base_device,
                               iree_string_view_t category,
                               iree_string_view_t key, int64_t *out_value) {
  quidditch_device_t *device = cast_device(base_device);

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value =
        iree_hal_query_any_executable_loader_support(
            device->loader_count, device->loaders, /*caching_mode=*/0, key)
            ? 1
            : 0;

    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t create_descriptor_set_layout(
    iree_hal_device_t *base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t *bindings,
    iree_hal_descriptor_set_layout_t **out_descriptor_set_layout) {
  return iree_hal_local_descriptor_set_layout_create(
      flags, binding_count, bindings,
      iree_hal_device_host_allocator(base_device), out_descriptor_set_layout);
}

static iree_status_t create_pipeline_layout(
    iree_hal_device_t *base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t *const *set_layouts,
    iree_hal_pipeline_layout_t **out_pipeline_layout) {
  return iree_hal_local_pipeline_layout_create(
      push_constants, set_layout_count, set_layouts,
      iree_hal_device_host_allocator(base_device), out_pipeline_layout);
}

static iree_status_t create_semaphore(iree_hal_device_t *base_device,
                                      uint64_t initial_value,
                                      iree_hal_semaphore_t **out_semaphore) {
  IREE_ATTRIBUTE_UNUSED quidditch_device_t *device = cast_device(base_device);

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static const iree_hal_device_vtable_t quidditch_device_vtable = {
    .destroy = destroy,
    .host_allocator = host_allocator,
    .device_allocator = device_allocator,
    .create_executable_cache = create_executable_cache,
    .query_i64 = query_i64,
    .create_descriptor_set_layout = create_descriptor_set_layout,
    .create_pipeline_layout = create_pipeline_layout,
    .create_semaphore = create_semaphore,
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
