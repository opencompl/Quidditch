
#pragma once

#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/executable_loader.h"

iree_status_t quidditch_loader_create(
    iree_host_size_t library_count,
    const iree_hal_executable_library_query_fn_t* library_query_fns,
    iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);
