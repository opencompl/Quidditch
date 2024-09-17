#include <Quidditch/device/device.h>
#include <Quidditch/dispatch/dispatch.h>
#include <Quidditch/loader/loader.h>

#include <iree/base/allocator.h>
#include <iree/hal/allocator.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/vm/instance.h>

#include <big_matvec.h>
#include <big_matvec_module.h>
#include <team_decls.h>
#include <util/run_model.h>

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
                                          /*data_allocator=*/host_allocator,
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

int main() {
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              quidditch_big_matvec_linked_quidditch_library_query,
          },
      .num_libraries = 1,
      .module_constructor = big_matvec_create,

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 2,
      .input_sizes = (const iree_host_size_t[]){1 * 320, 320 * 400},
      .input_ranks = (const iree_host_size_t[]){2, 2},
      .input_shapes = (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){1, 400},
                                                (iree_hal_dim_t[]){320, 400}},

      .num_outputs = 1,
      .output_sizes = (const iree_host_size_t[]){1, 320},
  };

  // Inlined 'run_model' to avoid constructing and destructing the device
  // and host module multiple times.

  iree_allocator_t host_allocator = iree_allocator_system();

  iree_vm_instance_t* vmInstance;
  iree_hal_device_t* device;
  IREE_CHECK_OK(
      setup_instance_and_device(&config, host_allocator, &vmInstance, &device));

  iree_vm_module_t* hal_module = NULL;
  IREE_CHECK_OK(iree_hal_module_create(vmInstance, /*device_count=*/1,
                                       /*devices=*/&device,
                                       IREE_HAL_MODULE_FLAG_NONE,
                                       host_allocator, &hal_module));

  iree_vm_module_t* mlir_module = NULL;
  IREE_CHECK_OK(
      config.module_constructor(vmInstance, host_allocator, &mlir_module));

  iree_vm_module_t* modules[] = {hal_module, mlir_module};

  iree_vm_context_t* context;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      vmInstance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      host_allocator, &context));

  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(
      /*element_type=*/iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/config.num_inputs, host_allocator, &inputs));

  for (iree_host_size_t i = 0; i < config.num_inputs; i++) {
    iree_const_byte_span_t span = iree_make_const_byte_span(
        config.input_data[i],
        config.input_sizes[i] *
            iree_hal_element_dense_byte_count(config.element_type));

    iree_device_size_t out_size;
    IREE_CHECK_OK(iree_hal_buffer_compute_view_size(
        config.input_ranks[i], config.input_shapes[i], config.element_type,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &out_size));

    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        .access = IREE_HAL_MEMORY_ACCESS_NONE,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    };
    iree_hal_buffer_params_canonicalize(&params);

    iree_hal_buffer_t* buffer = NULL;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, out_size, &buffer));

    iree_hal_buffer_view_t* buffer_view;
    IREE_CHECK_OK(iree_hal_buffer_view_create(
        buffer, config.input_ranks[i], config.input_shapes[i],
        config.element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        host_allocator, &buffer_view));

    iree_vm_ref_t arg_buffer_view_ref;
    arg_buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
    IREE_CHECK_OK(iree_vm_list_push_ref_retain(inputs, &arg_buffer_view_ref));
  }

  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(
      /*element_type=*/iree_vm_make_undefined_type_def(),
      /*initial_capacity=*/1, host_allocator, &outputs));

  iree_string_view_t functions[] = {
      iree_make_cstring_view("big_matvec.test32"),
      iree_make_cstring_view("big_matvec.test40"),
      iree_make_cstring_view("big_matvec.test64"),
      iree_make_cstring_view("big_matvec.test32_100"),
      iree_make_cstring_view("big_matvec.test40_100"),
  };

  for (int i = 0; i < IREE_ARRAYSIZE(functions); i++) {
    iree_vm_function_t main_function;
    IREE_CHECK_OK(iree_vm_context_resolve_function(context, functions[i],
                                                   &main_function));

    IREE_CHECK_OK(
        iree_vm_invoke(context, main_function, IREE_VM_CONTEXT_FLAG_NONE,
                       /*policy=*/NULL, inputs, outputs, host_allocator));
  }

  iree_vm_list_release(outputs);
  iree_vm_list_release(inputs);
  iree_vm_context_release(context);
  iree_vm_module_release(mlir_module);
  iree_vm_module_release(hal_module);
  iree_hal_device_release(device);
  iree_vm_instance_release(vmInstance);

  quidditch_dispatch_quit();

  return 0;
}
