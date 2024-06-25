
#pragma once

#include <iree/base/status.h>
#include <iree/hal/api.h>
#include <iree/hal/local/executable_library.h>
#include <iree/vm/instance.h>
#include <iree/vm/module.h>

typedef struct {
  /// Array of static library kernels that must be imported.
  iree_hal_executable_library_query_fn_t* libraries;
  /// Number of elements in 'libraries'.
  iree_host_size_t num_libraries;

  /// EmitC host module created by iree-compile that should be executed.
  iree_status_t (*module_constructor)(iree_vm_instance_t*,
                                      iree_allocator_t host_allocator,
                                      iree_vm_module_t** module_out);
  /// Main function that should be called.
  iree_string_view_t main_function;

  iree_hal_element_type_t element_type;

  /// Number of input tensors.
  iree_host_size_t num_inputs;
  /// Input tensor data in dense row major encoding.
  const void** input_data;
  /// Number of elements for each input in 'input_data'.
  const iree_host_size_t* input_sizes;
  /// Ranks of each input tensor.
  const iree_host_size_t* input_ranks;
  /// Shapes for each input tensor.
  const iree_hal_dim_t** input_shapes;

  /// Number of output tensors.
  iree_host_size_t num_outputs;
  /// Memory for the output data.
  void** output_data;
  /// Number of elements for each array in 'output_data'.
  const iree_host_size_t* output_sizes;
} model_config_t;

/// Runs the given IREE module according to the config. Input and output data
/// are copied into and out of the given memory.
iree_status_t run_model(const model_config_t* config);
