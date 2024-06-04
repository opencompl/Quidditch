#include "run_model.h"

#include <Quidditch/device/device.h>
#include <Quidditch/dispatch/dispatch.h>
#include <Quidditch/loader/loader.h>

#include <iree/base/allocator.h>
#include <iree/hal/allocator.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/instance.h>

#include <snitch_cluster_defs.h>
#include <team_decls.h>

iree_allocator_t l1_allocator() {
  uint32_t snrt_l1_start_addr();
  uint32_t snrt_l1_end_addr();

  static iree_allocator_inline_storage_t l1_arena;

  l1_arena.buffer = (uint8_t*)snrt_l1_start_addr();
  l1_arena.length = 0;
  unsigned stack_size_per_core = 1 << SNRT_LOG2_STACK_SIZE;
  l1_arena.capacity =
      (snrt_l1_end_addr() - snrt_cluster_core_num() * stack_size_per_core) -
      snrt_l1_start_addr();

  return iree_allocator_inline_arena(&l1_arena);
}

static iree_status_t setup_instance_and_device(
    const model_config_t* config, iree_allocator_t host_allocator,
    iree_vm_instance_t** out_instance, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_instance);
  IREE_ASSERT_ARGUMENT(out_device);

  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator, out_instance));

  iree_status_t result = iree_hal_module_register_all_types(*out_instance);
  if (!iree_status_is_ok(result)) goto error_release_vm;

  iree_hal_executable_loader_t* loader;
  result = quidditch_loader_create(config->num_libraries, config->libraries,
                                   iree_hal_executable_import_provider_null(),
                                   host_allocator, &loader);
  if (!iree_status_is_ok(result)) goto error_release_vm;

  iree_hal_allocator_t* device_allocator;
  result = iree_hal_allocator_create_heap(iree_make_cstring_view("quidditch"),
                                          config->device_allocator,
                                          host_allocator, &device_allocator);
  if (!iree_status_is_ok(result)) goto error_release_library_loader;

  quidditch_device_params_t params;
  quidditch_device_params_initialize(&params);
  result =
      quidditch_device_create(IREE_SV("snitch"), &params,
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

iree_status_t run_model(const model_config_t* config) {
  if (!snrt_is_dm_core()) {
    int ret = quidditch_dispatch_enter_worker_loop();
    if (!ret) return iree_ok_status();
    return iree_make_status(IREE_STATUS_UNKNOWN);
  }

  iree_allocator_t host_allocator = iree_allocator_system();

  iree_vm_instance_t* vmInstance;
  iree_hal_device_t* device;
  iree_status_t result =
      setup_instance_and_device(config, host_allocator, &vmInstance, &device);
  IREE_RETURN_IF_ERROR(result);

  iree_vm_module_t* hal_module = NULL;
  result =
      iree_hal_module_create(vmInstance, /*device_count=*/1,
                             /*devices=*/&device, IREE_HAL_MODULE_FLAG_NONE,
                             host_allocator, &hal_module);
  if (!iree_status_is_ok(result)) goto error_release_instance_and_device;

  iree_vm_module_t* mlir_module = NULL;
  result = config->module_constructor(vmInstance, host_allocator, &mlir_module);
  if (!iree_status_is_ok(result)) goto error_release_hal_module;

  iree_vm_module_t* modules[] = {hal_module, mlir_module};

  iree_vm_context_t* context;
  result = iree_vm_context_create_with_modules(
      vmInstance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      host_allocator, &context);
  if (!iree_status_is_ok(result)) goto error_release_mlir_module;

  iree_vm_list_t* inputs = NULL;
  result = iree_vm_list_create(
      /*element_type=*/iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/config->num_inputs, iree_allocator_system(),
      &inputs);
  if (!iree_status_is_ok(result)) goto error_release_context;

  for (iree_host_size_t i = 0; i < config->num_inputs; i++) {
    iree_const_byte_span_t span = iree_make_const_byte_span(
        config->input_data[i], config->input_sizes[i] * sizeof(double));

    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        .access = IREE_HAL_MEMORY_ACCESS_NONE,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    };
    iree_hal_buffer_params_canonicalize(&params);

    iree_hal_buffer_view_t* buffer = NULL;
    result = iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), config->input_ranks[i],
        config->input_shapes[i], IREE_HAL_ELEMENT_TYPE_FLOAT_64,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params, span, &buffer);
    if (!iree_status_is_ok(result)) goto error_release_context;

    iree_vm_ref_t arg_buffer_view_ref;
    arg_buffer_view_ref = iree_hal_buffer_view_move_ref(buffer);
    result = iree_vm_list_push_ref_retain(inputs, &arg_buffer_view_ref);
    if (!iree_status_is_ok(result)) goto error_release_context;
  }

  iree_vm_function_t main_function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(context, config->main_function,
                                                 &main_function));

  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(
      /*element_type=*/iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/1, iree_allocator_system(), &outputs));
  IREE_CHECK_OK(iree_vm_invoke(
      context, main_function, IREE_VM_CONTEXT_FLAG_NONE,
      /*policy=*/NULL, inputs, outputs, iree_allocator_system()));

  if (!iree_status_is_ok(result)) goto error_release_output;

  for (iree_host_size_t i = 0; i < config->num_outputs; i++) {
    iree_hal_buffer_view_t* ret_buffer_view =
        iree_vm_list_get_ref_deref(outputs, i, iree_hal_buffer_view_type());
    if (ret_buffer_view == NULL) goto error_release_output;

    iree_hal_device_transfer_d2h(
        device, iree_hal_buffer_view_buffer(ret_buffer_view), 0,
        config->output_data[i], config->output_sizes[i] * sizeof(double),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

error_release_output:
  iree_vm_list_release(outputs);
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

  if (!iree_status_is_ok(result)) return result;

  quidditch_dispatch_quit();
  return 0;
}
