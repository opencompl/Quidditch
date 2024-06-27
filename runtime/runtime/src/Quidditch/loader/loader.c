#include "loader.h"

#include "Quidditch/executable/executable.h"
#include "iree/hal/local/executable_environment.h"

//===----------------------------------------------------------------------===//
// quidditch_loader_t, fork of iree_hal_static_library_loader_t.
//===----------------------------------------------------------------------===//

typedef struct quidditch_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
  iree_host_size_t library_count;
  const iree_hal_executable_library_header_t** const libraries[];
} quidditch_loader_t;

static const iree_hal_executable_loader_vtable_t quidditch_loader_vtable;

iree_status_t quidditch_loader_create(
    iree_host_size_t library_count,
    const iree_hal_executable_library_query_fn_t* library_query_fns,
    iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(!library_count || library_query_fns);
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  quidditch_loader_t* executable_loader = NULL;
  iree_host_size_t total_size =
      sizeof(*executable_loader) +
      sizeof(executable_loader->libraries[0]) * library_count;
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size,
                                               (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &quidditch_loader_vtable, import_provider, &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    executable_loader->library_count = library_count;

    // Default environment to enable initialization.
    iree_hal_executable_environment_v0_t environment;
    iree_hal_executable_environment_initialize(host_allocator, &environment);

    // Query and verify the libraries provided all match our expected version.
    // It's rare they won't, however static libraries generated with a newer
    // version of the IREE compiler that are then linked with an older version
    // of the runtime are difficult to spot otherwise.
    for (iree_host_size_t i = 0; i < library_count; ++i) {
      const iree_hal_executable_library_header_t* const* header_ptr =
          library_query_fns[i](IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
                               &environment);
      if (!header_ptr) {
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "failed to query library header for runtime version %d",
            IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
        break;
      }
      const iree_hal_executable_library_header_t* header = *header_ptr;
      IREE_TRACE_ZONE_APPEND_TEXT(z0, header->name);
      if (header->version > IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "executable does not support this version of the "
            "runtime (executable: %d, runtime: %d)",
            header->version, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
        break;
      }
      memcpy((void*)&executable_loader->libraries[i], &header_ptr,
             sizeof(header_ptr));
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  } else {
    iree_allocator_free(host_allocator, executable_loader);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void quidditch_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  quidditch_loader_t* executable_loader =
      (quidditch_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool quidditch_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("static")) ||
         iree_string_view_equal(executable_format,
                                iree_make_cstring_view("snitch"));
}

static iree_status_t quidditch_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  quidditch_loader_t* executable_loader =
      (quidditch_loader_t*)base_executable_loader;

  // The executable data is just the name of the library.
  iree_string_view_t library_name = iree_make_string_view(
      (const char*)executable_params->executable_data.data,
      executable_params->executable_data.data_length);

  // Linear scan of the registered libraries; there's usually only one per
  // module (aka source model) and as such it's a small list and probably not
  // worth optimizing. We could sort the libraries list by name on loader
  // creation to perform a binary-search fairly easily, though, at the cost of
  // the additional code size.
  for (iree_host_size_t i = 0; i < executable_loader->library_count; ++i) {
    const iree_hal_executable_library_header_t* header =
        *executable_loader->libraries[i];
    if (iree_string_view_equal(library_name,
                               iree_make_cstring_view(header->name))) {
      return quidditch_executable_create(
          executable_params, executable_loader->libraries[i],
          base_executable_loader->import_provider,
          executable_loader->host_allocator, out_executable);
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no static library with the name '%.*s' registered",
                          (int)library_name.size, library_name.data);
}

static const iree_hal_executable_loader_vtable_t quidditch_loader_vtable = {
    .destroy = quidditch_loader_destroy,
    .query_support = quidditch_loader_query_support,
    .try_load = quidditch_loader_try_load,
};
