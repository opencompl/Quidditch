#include <Quidditch/dispatch/dispatch.h>

#include <tiny_matmul.h>
#include <tiny_matmul_module.h>
#include <team_decls.h>
#include <util/run_model.h>

int main() {
  iree_alignas(64) double res[9];
  iree_alignas(64) double data[9];
  iree_alignas(64) double correct[9] = {30,36,42,66,81,96,102,126,150};

  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  // [1, 2, 3, 4, 5, 6, 7, 8, 9]
  for (int i = 0; i < IREE_ARRAYSIZE(data); i++) {
    data[i] = (i + 1);
    res[i]=0;
  }

  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              quidditch_tiny_matmul_dispatch_0_library_query,
          },
      .num_libraries = 1,
      .module_constructor = test_tiny_matmul_create,
      .main_function = iree_make_cstring_view("test_tiny_matmul.tiny_matmul"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 3,
      .input_data = (const void*[]){data, data, res},
      .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(data),
                                                IREE_ARRAYSIZE(data),
                                                IREE_ARRAYSIZE(res)},
      .input_ranks = (const iree_host_size_t[]){2, 2, 2},
      .input_shapes =
          (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){3,3},
                                    (iree_hal_dim_t[]){3,3},
                                    (iree_hal_dim_t[]){3,3}},

      .num_outputs = 1,
      .output_data = (void*[]){res},
      .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(res)},
  };

  IREE_CHECK_OK(run_model(&config));

  if (!snrt_is_dm_core()) return 0;

  // check correctness
  for (int i = 0; i < IREE_ARRAYSIZE(data); i++) {
    double value = data[i];
    printf("%f\n", value);
    if (value != correct[i]) return 1;
  }
  return 0;
}
