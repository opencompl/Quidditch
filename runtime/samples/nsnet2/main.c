#include <nsnet2.h>
#include <nsnet2_module.h>
#include <team_decls.h>
#include <util/run_model.h>

int main() {
  float data[161];

  for (int i = 0; i < IREE_ARRAYSIZE(data); i++) {
    data[i] = (i + 1);
  }

  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              compiled_ns_net2_linked_llvm_cpu_library_query},
      .num_libraries = 1,
      .module_constructor = compiled_ns_net2_create,
      .main_function = iree_make_cstring_view("compiled_ns_net2.main"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32,

      .num_inputs = 1,
      .input_data = (const void*[]){data, data},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(data)},
      .input_ranks = (const iree_host_size_t[]){3},
      .input_shapes = (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){1, 1, 161}},

      .num_outputs = 1,
      .output_data = (void*[]){data},
      .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(data)},
      .device_allocator = l3_allocator(),
  };

  IREE_CHECK_OK(run_model(&config));

  if (!snrt_is_dm_core()) return 0;

  for (int i = 0; i < IREE_ARRAYSIZE(data); i++) {
    float value = data[i];
    printf("%f\n", value);
  }
  return 0;
}
