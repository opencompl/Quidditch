#include <Quidditch/device.h>

#include <iree/base/allocator.h>
#include <iree/hal/allocator.h>
#include <iree/hal/local/loaders/static_library_loader.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/instance.h>

#include <team_decls.h>

static iree_status_t setup() {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_vm_instance_t *instance;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator, &instance));

  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  const iree_hal_executable_library_query_fn_t libraries[] = {};

  iree_hal_executable_loader_t *loader;
  IREE_RETURN_IF_ERROR(iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries,
      iree_hal_executable_import_provider_null(), host_allocator, &loader));

  // TODO: Replace with more sophisticated allocator representing cluster
  // memory.
  iree_hal_allocator_t *device_allocator;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
      iree_make_cstring_view("quidditch"), host_allocator, host_allocator,
      &device_allocator));

  iree_hal_device_t *device;
  IREE_RETURN_IF_ERROR(quidditch_device_create(
      /*loader_count=*/1, &loader, device_allocator, host_allocator, &device))

  return iree_ok_status();
}

int main() {
  // TODO: Remove/redirect compute cores once implemented.
  if (!snrt_is_dm_core()) return 0;

  iree_status_t result = setup();
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return -1;
  }

  return 0;
}
