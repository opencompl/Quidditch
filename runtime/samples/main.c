#include <Quidditch/device.h>

#include <iree/base/allocator.h>
#include <iree/hal/allocator.h>
#include <iree/hal/local/loaders/static_library_loader.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/instance.h>

#include <team_decls.h>

static iree_status_t setup_instance_and_device(
    iree_allocator_t host_allocator, iree_vm_instance_t** out_instance,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_instance);
  IREE_ASSERT_ARGUMENT(out_device);

  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator, out_instance));

  iree_status_t result = iree_hal_module_register_all_types(*out_instance);
  if (!iree_status_is_ok(result)) goto error_release_vm;

  const iree_hal_executable_library_query_fn_t libraries[] = {};

  iree_hal_executable_loader_t* loader;
  result = iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries,
      iree_hal_executable_import_provider_null(), host_allocator, &loader);
  if (!iree_status_is_ok(result)) goto error_release_vm;

  // TODO: Replace with more sophisticated allocator representing cluster
  // memory.
  iree_hal_allocator_t* device_allocator;
  result = iree_hal_allocator_create_heap(iree_make_cstring_view("quidditch"),
                                          host_allocator, host_allocator,
                                          &device_allocator);
  if (!iree_status_is_ok(result)) goto error_release_library_loader;

  result = quidditch_device_create(
      /*loader_count=*/1, &loader, device_allocator, host_allocator,
      out_device);
  iree_hal_executable_loader_release(loader);
  iree_hal_allocator_release(device_allocator);
  return result;

error_release_library_loader:
  iree_hal_executable_loader_release(loader);
error_release_vm:
  iree_vm_instance_release(*out_instance);
  return result;
}

int main() {
  // TODO: Remove/redirect compute cores once implemented.
  if (!snrt_is_dm_core()) return 0;

  iree_allocator_t host_allocator = iree_allocator_system();

  iree_vm_instance_t* vmInstance;
  iree_hal_device_t* device;
  iree_status_t result =
      setup_instance_and_device(host_allocator, &vmInstance, &device);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return -1;
  }

  // TODO: Create EmitC module here.

  iree_vm_module_t* hal_module = NULL;
  result =
      iree_hal_module_create(vmInstance, /*device_count=*/1,
                             /*devices=*/&device, IREE_HAL_MODULE_FLAG_NONE,
                             host_allocator, &hal_module);
  if (!iree_status_is_ok(result)) goto error_release_instance_and_device;

  iree_vm_module_t* modules[] = {hal_module};

  iree_vm_context_t* context;
  result = iree_vm_context_create_with_modules(
      vmInstance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      host_allocator, &context);
  if (!iree_status_is_ok(result)) goto error_release_hal_module;

  // TODO: Run modules.

  iree_vm_context_release(context);
error_release_hal_module:
  iree_vm_module_release(hal_module);

error_release_instance_and_device:
  iree_hal_device_release(device);
  iree_vm_instance_release(vmInstance);

exit:
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return -1;
  }

  return 0;
}
