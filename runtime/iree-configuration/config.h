
#pragma once

// TODO: This only necessary to work with the pulp-toolchain which does not
//  support 'aligned_alloc', a function called by in status.c.
#define IREE_STATUS_FEATURES 0

#define IREE_SYNCHRONIZATION_DISABLE_UNSAFE 1
#define IREE_FILE_IO_ENABLE 0
#define IREE_TIME_NOW_FN return 0;
#define IREE_WAIT_UNTIL_FN(...) true
