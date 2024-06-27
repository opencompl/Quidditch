// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/hal/local/executable_library.h"

// A table of exported functions arranged as a struct-of-arrays for more
// efficient packing and faster lookup. Each subarray - when not omitted and
// NULL - is indexed by export ordinal and has up to |count| entries.
typedef struct quidditch_executable_export_table_v0_t {
  // Total number of exports in the table.
  uint32_t count;

  // Function pointers for each exported entry point.
  const iree_hal_executable_dispatch_v0_t* compute_core_ptrs;

  // Optional table of attributes 1:1 with ptrs.
  // Omitting the table entirely means that no exports need workgroup local
  // memory (or whatever else we pack into the attributes).
  const iree_hal_executable_dispatch_attrs_v0_t* attrs;

  // Optional table of export function entry point names 1:1 with ptrs.
  // These names are only used for tracing/debugging and can be omitted to save
  // binary size.
  const char* const* names;

  // Optional table of entry point tags 1:1 with ptrs.
  // Used to describe the entry point in a human-readable format useful for
  // verbose logging. The string values, when present, may be attached to
  // tracing/debugging events related to the entry point.
  const char* const* tags;

  // Optional table of source locations 1:1 with ptrs.
  // These are the canonical source location in the compiler.
  const iree_hal_executable_source_location_v0_t* source_locations;

  // Optional table of source locations by compilation stage 1:1 with ptrs.
  // These may provide additional internal compilation results at various
  // stages of compilation.
  const iree_hal_executable_stage_location_table_v0_t* stage_locations;

  const iree_hal_executable_dispatch_v0_t* dma_core_ptrs;
} quidditch_executable_export_table_v0_t;

// Structure used for v0 library interfaces.
// The entire structure is designed to be read-only and able to live embedded in
// the binary .rdata section.
//
// The information held within the structure is not cached by the runtime.
// Implementations may choose to heap allocate this structure and modify its
// members at runtime so long as they observe the thread-safety guarantees.
// For example, a JIT may default all exports to JIT thunk functions and then
// atomically swap them out for the translated function pointers as they are
// available.
typedef struct quidditch_executable_library_v0_t {
  // Version/metadata header.
  // Will have a version of quidditch_EXECUTABLE_LIBRARY_VERSION_*.
  const iree_hal_executable_library_header_t* header;

  // Table of imported functions available to functions in the executable.
  iree_hal_executable_import_table_v0_t imports;

  // Table of exported functions from the executable.
  quidditch_executable_export_table_v0_t exports;

  // Table of executable-level constants.
  iree_hal_executable_constant_table_v0_t constants;

  // Table of optional sources used for debugging.
  // Exports may reference locations within the sources by path.
  iree_hal_executable_source_file_table_v0_t sources;
} quidditch_executable_library_v0_t;
