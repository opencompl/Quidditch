#include <simple_add.h>
#include <simple_add_module.h>
#include <team_decls.h>

#include "util/run_model.h"

int main() {
  double data[4];

  for (int i = 0; i < IREE_ARRAYSIZE(data); i++) {
    data[i] = (i + 1);
  }

  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              add_dispatch_0_library_query},
      .num_libraries = 1,
      .module_constructor = test_simple_add_create,
      .main_function = iree_make_cstring_view("test_simple_add.add"),

      .num_inputs = 2,
      .input_data = (const double*[]){data, data},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(data),
                                                IREE_ARRAYSIZE(data)},
      .input_ranks = (const iree_host_size_t[]){1, 1},
      .input_shapes =
          (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){IREE_ARRAYSIZE(data)},
                                    (iree_hal_dim_t[]){IREE_ARRAYSIZE(data)}},

      .num_outputs = 1,
      .output_data = (double*[]){data},
      .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(data)},
      .device_allocator = l1_allocator(),
  };

  IREE_CHECK_OK(run_model(&config));

  if (!snrt_is_dm_core()) return 0;

  for (int i = 0; i < IREE_ARRAYSIZE(data); i++) {
    double value = data[i];
    printf("%f\n", value);
    if (value == (i + 1) * 2) continue;
  }
  return 0;
}
