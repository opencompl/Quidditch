// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable.h"

#include <team_decls.h>

#include "Quidditch/dispatch/dispatch.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/executable_library_util.h"

static void quidditch_executable_destroy(
    iree_hal_executable_t* base_executable) {
  quidditch_executable_t* executable = (quidditch_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_library_deinitialize_imports(&executable->environment,
                                                   host_allocator);

  for (iree_host_size_t i = 0; i < executable->pipeline_layout_count; ++i) {
    iree_hal_pipeline_layout_release(executable->pipeline_layouts[i]);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_vtable_t quidditch_executable_vtable = {
    .destroy = quidditch_executable_destroy,
};

void quidditch_executable_initialize(
    iree_host_size_t pipeline_layout_count,
    iree_hal_pipeline_layout_t* const* source_pipeline_layouts,
    iree_hal_pipeline_layout_t** target_pipeline_layouts,
    iree_allocator_t host_allocator,
    quidditch_executable_t* out_base_executable) {
  iree_hal_resource_initialize(&quidditch_executable_vtable,
                               &out_base_executable->resource);
  out_base_executable->host_allocator = host_allocator;

  out_base_executable->pipeline_layout_count = pipeline_layout_count;
  out_base_executable->pipeline_layouts = target_pipeline_layouts;
  for (iree_host_size_t i = 0; i < pipeline_layout_count; ++i) {
    target_pipeline_layouts[i] = source_pipeline_layouts[i];
    iree_hal_pipeline_layout_retain(source_pipeline_layouts[i]);
  }

  // Function attributes are optional and populated by the parent type.
  out_base_executable->dispatch_attrs = NULL;

  // Default environment with no imports assigned.
  iree_hal_executable_environment_initialize(host_allocator,
                                             &out_base_executable->environment);
}

static int quidditch_executable_import_thunk_v0(
    iree_hal_executable_import_v0_t fn_ptr, void* params, void* context,
    void* reserved) {
  return fn_ptr(params, context, reserved);
}

iree_status_t quidditch_executable_create(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_library_header_t** library_header,
    const iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(!executable_params->pipeline_layout_count ||
                       executable_params->pipeline_layouts);
  IREE_ASSERT_ARGUMENT(!executable_params->constant_count ||
                       executable_params->constants);
  IREE_ASSERT_ARGUMENT(library_header);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  quidditch_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_params->pipeline_layout_count * sizeof(*executable->layouts) +
      executable_params->constant_count * sizeof(*executable_params->constants);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    quidditch_executable_initialize(executable_params->pipeline_layout_count,
                                    executable_params->pipeline_layouts,
                                    &executable->layouts[0], host_allocator,
                                    executable);
    executable->library.header = library_header;
    executable->identifier = iree_make_cstring_view((*library_header)->name);
    executable->dispatch_attrs = executable->library.v0->exports.attrs;
  }

  // Copy executable constants so we own them.
  if (iree_status_is_ok(status) && executable_params->constant_count > 0) {
    uint32_t* target_constants =
        (uint32_t*)((uint8_t*)executable + sizeof(*executable) +
                    executable_params->pipeline_layout_count *
                        sizeof(*executable->layouts));
    memcpy(target_constants, executable_params->constants,
           executable_params->constant_count *
               sizeof(*executable_params->constants));
    executable->environment.constants = target_constants;
  }

  // Resolve imports, if any.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_initialize_imports(
        &executable->environment, import_provider,
        &executable->library.v0->imports, quidditch_executable_import_thunk_v0,
        host_allocator);
  }

  // Verify that the library matches the executable params.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_verify(executable_params,
                                                executable->library.v0);
  }

  // Publish the executable sources with the tracing infrastructure.
  if (iree_status_is_ok(status)) {
    iree_hal_executable_library_publish_source_files(executable->library.v0);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    *out_executable = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

quidditch_executable_t* quidditch_executable_cast(
    iree_hal_executable_t* base_value) {
  return (quidditch_executable_t*)base_value;
}

iree_status_t quidditch_executable_issue_dispatch_inline(
    quidditch_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    uint32_t processor_id, iree_byte_span_t local_memory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO(benvanik): annotate with executable name to calculate total time.

  const uint32_t workgroup_count_x = dispatch_state->workgroup_count_x;
  const uint32_t workgroup_count_y = dispatch_state->workgroup_count_y;
  const uint32_t workgroup_count_z = dispatch_state->workgroup_count_z;

#if IREE_HAL_VERBOSE_TRACING_ENABLE
  // TODO(benvanik): tracing.h helper that speeds this up; too slow.
  IREE_TRACE({
    char xyz_string[32];
    int xyz_string_length =
        snprintf(xyz_string, IREE_ARRAYSIZE(xyz_string), "%ux%ux%u",
                 workgroup_count_x, workgroup_count_y, workgroup_count_z);
    IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(z0, xyz_string, xyz_string_length);
  });
#endif  // IREE_HAL_VERBOSE_TRACING_ENABLE

  iree_hal_executable_workgroup_state_v0_t workgroup_state;

  workgroup_state.local_memory = local_memory.data;
  workgroup_state.local_memory_size = (size_t)local_memory.data_length;

  const iree_hal_executable_library_v0_t* library = executable->library.v0;

  if (IREE_UNLIKELY(ordinal >= library->exports.count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }
  iree_hal_executable_dispatch_v0_t kernel = library->exports.ptrs[ordinal];

  // Note that this is technically not required as kernel execution has as
  // post-condition that all compute cores are parked.
  quidditch_dispatch_wait_for_workers();

  quidditch_dispatch_set_kernel(kernel, &executable->environment,
                                dispatch_state);

  uint32_t worker_to_pop = 0;
  for (uint32_t z = 0; z < workgroup_count_z; ++z) {
    workgroup_state.workgroup_id_z = z;
    for (uint32_t y = 0; y < workgroup_count_y; ++y) {
      workgroup_state.workgroup_id_y = y;
      for (uint32_t x = 0; x < workgroup_count_x; ++x) {
        workgroup_state.workgroup_id_x = x;
        workgroup_state.processor_id = worker_to_pop;

        quidditch_dispatch_submit_workgroup(&workgroup_state);

        worker_to_pop++;
        if (worker_to_pop == snrt_cluster_compute_core_num()) {
          // Wait for all workers to be done before scheduling more of them.
          // This is easier than waiting for the next compute core to be done.
          quidditch_dispatch_wait_for_workers();
          worker_to_pop = 0;
        }
      }
    }
  }

  quidditch_dispatch_wait_for_workers();

  if (quidditch_dispatch_errors_occurred())
    return iree_make_status(IREE_STATUS_INTERNAL);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
