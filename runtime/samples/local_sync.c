#include <iree/base/allocator.h>
#include <iree/hal/allocator.h>
#include <iree/hal/drivers/local_sync/sync_device.h>
#include <iree/hal/local/loaders/static_library_loader.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/instance.h>

#include <simple_add.h>
#include <simple_add_module.h>
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

  const iree_hal_executable_library_query_fn_t libraries[] = {
      add_dispatch_0_library_query};

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

  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  result =
      iree_hal_sync_device_create(IREE_SV("quidditch-base-line"), &params,
                                  /*loader_count=*/1, &loader, device_allocator,
                                  host_allocator, out_device);
  iree_hal_executable_loader_release(loader);
  iree_hal_allocator_release(device_allocator);
  return result;

error_release_library_loader:
  iree_hal_executable_loader_release(loader);
error_release_vm:
  iree_vm_instance_release(*out_instance);
  return result;
}

static float data[128];

int main() {
  // TODO: Remove/redirect compute cores once implemented.
  if (!snrt_is_dm_core()) return 0;

  for (int i = 0; i < 128; i++) {
    data[i] = i;
  }

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

  iree_vm_module_t* hal_module = NULL;
  result =
      iree_hal_module_create(vmInstance, /*device_count=*/1,
                             /*devices=*/&device, IREE_HAL_MODULE_FLAG_NONE,
                             host_allocator, &hal_module);
  if (!iree_status_is_ok(result)) goto error_release_instance_and_device;

  iree_vm_module_t* mlir_module = NULL;
  result = test_simple_add_create(vmInstance, host_allocator, &mlir_module);
  if (!iree_status_is_ok(result)) goto error_release_hal_module;

  iree_vm_module_t* modules[] = {hal_module, mlir_module};

  iree_vm_context_t* context;
  result = iree_vm_context_create_with_modules(
      vmInstance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      host_allocator, &context);
  if (!iree_status_is_ok(result)) goto error_release_mlir_module;

  iree_const_byte_span_t span = iree_make_const_byte_span(data, sizeof(data));

  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_NONE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_params_canonicalize(&params);

  iree_hal_buffer_view_t* buffer = NULL;
  result = iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), 1, (iree_hal_dim_t[]){128},
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params, span, &buffer);
  if (!iree_status_is_ok(result)) goto error_release_context;

  iree_vm_list_t* inputs = NULL;
  result = iree_vm_list_create(
      /*element_type=*/iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/2, iree_allocator_system(), &inputs);
  if (!iree_status_is_ok(result)) goto error_release_context;

  iree_vm_ref_t arg_buffer_view_ref;
  arg_buffer_view_ref = iree_hal_buffer_view_move_ref(buffer);
  result = iree_vm_list_push_ref_retain(inputs, &arg_buffer_view_ref);
  if (!iree_status_is_ok(result)) goto error_release_input;

  result = iree_vm_list_push_ref_move(inputs, &arg_buffer_view_ref);
  if (!iree_status_is_ok(result)) goto error_release_input;

  iree_vm_function_t main_function;
  result = iree_vm_context_resolve_function(
      context, iree_make_cstring_view("test_simple_add.add"), &main_function);

  iree_vm_list_t* outputs = NULL;
  result = iree_vm_list_create(
      /*element_type=*/iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/1, iree_allocator_system(), &outputs);
  if (!iree_status_is_ok(result)) goto error_release_output;

  IREE_CHECK_OK(iree_vm_invoke(
      context, main_function, IREE_VM_CONTEXT_FLAG_NONE,
      /*policy=*/NULL, inputs, outputs, iree_allocator_system()));
  if (!iree_status_is_ok(result)) goto error_release_output;

  iree_hal_buffer_view_t* ret_buffer_view =
      iree_vm_list_get_ref_deref(outputs, /*i=*/0, iree_hal_buffer_view_type());
  if (ret_buffer_view == NULL) goto error_release_output;

  iree_hal_buffer_mapping_t mapping;
  result = iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(ret_buffer_view),
      IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      IREE_WHOLE_BUFFER, &mapping);
  if (!iree_status_is_ok(result)) goto error_release_output;

  for (int i = 0; i < 128; i++) {
    if (((float*)mapping.contents.data)[i] == i * 2) continue;

    result = iree_make_status(IREE_STATUS_UNKNOWN, "output incorrect");
    break;
  }

  iree_hal_buffer_unmap_range(&mapping);

error_release_output:
  iree_vm_list_release(outputs);
error_release_input:
  iree_vm_list_release(inputs);
error_release_context:
  iree_vm_context_release(context);
error_release_mlir_module:
  iree_vm_module_release(mlir_module);
error_release_hal_module:
  iree_vm_module_release(hal_module);

error_release_instance_and_device:
  iree_hal_device_release(device);
  iree_vm_instance_release(vmInstance);

  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return -1;
  }

  return 0;
}
