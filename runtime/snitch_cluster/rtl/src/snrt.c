// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"

#include "cluster_interrupts.c"
#include "dma.c"
#include "printf.c"
#include "riscv.c"
#include "snitch_cluster_memory.c"
#include "snitch_cluster_start.c"
#include "stack_decls.h"
#include "sync.c"
#include "team.c"

uint32_t snrt_get_stack_size_per_core() { return 1 << SNRT_LOG2_STACK_SIZE; }
