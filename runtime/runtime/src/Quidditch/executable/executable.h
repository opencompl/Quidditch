// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <iree/hal/local/executable_loader.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct quidditch_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Optional pipeline layout
  // Not all users require the layouts (such as when directly calling executable
  // functions) and in those cases they can be omitted. Users routing through
  // the HAL command buffer APIs will usually require them.
  //
  // TODO(benvanik): make this a flag we set and can query instead - poking into
  // this from dispatch code is a layering violation.
  iree_host_size_t pipeline_layout_count;
  iree_hal_pipeline_layout_t** pipeline_layouts;

  // Defines per-entry point how much workgroup local memory is required.
  // Contains entries with 0 to indicate no local memory is required or >0 in
  // units of IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE for the minimum amount
  // of memory required by the function.
  const iree_hal_executable_dispatch_attrs_v0_t* dispatch_attrs;

  // Execution environment.
  iree_hal_executable_environment_v0_t environment;

  // Name used for the file field in tracy and debuggers.
  iree_string_view_t identifier;

  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;

  iree_hal_pipeline_layout_t* layouts[];
} quidditch_executable_t;

iree_status_t quidditch_executable_create(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_library_header_t** library_header,
    iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Initializes the local executable base type.
//
// Callers must allocate memory for |target_pipeline_layouts| with at least
// `pipeline_layout_count * sizeof(*target_pipeline_layouts)` bytes.
void quidditch_executable_initialize(
    iree_host_size_t pipeline_layout_count,
    iree_hal_pipeline_layout_t* const* source_pipeline_layouts,
    iree_hal_pipeline_layout_t** target_pipeline_layouts,
    iree_allocator_t host_allocator,
    quidditch_executable_t* out_base_executable);

quidditch_executable_t* quidditch_executable_cast(
    iree_hal_executable_t* base_value);

iree_status_t quidditch_executable_issue_call(
    quidditch_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state);

iree_status_t quidditch_executable_issue_dispatch_inline(
    quidditch_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    uint32_t processor_id, iree_byte_span_t local_memory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
