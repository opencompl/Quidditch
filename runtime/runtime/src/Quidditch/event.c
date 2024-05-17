// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "event.h"

#include <stddef.h>

typedef struct quidditch_event_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} quidditch_event_t;

static const iree_hal_event_vtable_t quidditch_event_vtable;

static quidditch_event_t* quidditch_event_cast(iree_hal_event_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &quidditch_event_vtable);
  return (quidditch_event_t*)base_value;
}

iree_status_t quidditch_event_create(iree_allocator_t host_allocator,
                                     iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  quidditch_event_t* event = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&quidditch_event_vtable, &event->resource);
    event->host_allocator = host_allocator;
    *out_event = (iree_hal_event_t*)event;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void quidditch_event_destroy(iree_hal_event_t* base_event) {
  quidditch_event_t* event = quidditch_event_cast(base_event);
  iree_allocator_t host_allocator = event->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_event_vtable_t quidditch_event_vtable = {
    .destroy = quidditch_event_destroy,
};
