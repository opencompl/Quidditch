
#pragma once

#define IREE_SYNCHRONIZATION_DISABLE_UNSAFE 1
#define IREE_FILE_IO_ENABLE 0
#define IREE_TIME_NOW_FN return 0;
#define IREE_WAIT_UNTIL_FN(...) true
