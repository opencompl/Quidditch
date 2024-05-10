// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"

#include "alloc.c"
#include "cls.c"
#include "cluster_interrupts.c"
#include "dma.c"
#include "eu.c"
#include "kmp.c"
#include "printf.c"
#include "riscv.c"
#include "snitch_cluster_start.c"
#include "sync.c"
#include "team.c"

// TODO: Remove declarations that are workarounds until
//  https://github.com/pulp-platform/snitch_cluster/pull/136 landed.

extern uint32_t snrt_l1_start_addr();
extern uint32_t snrt_l1_end_addr();

extern volatile uint32_t *snrt_zero_memory_ptr();

extern cls_t* cls();

extern snrt_allocator_t *snrt_l1_allocator();
extern snrt_allocator_t *snrt_l3_allocator();

extern uint32_t snrt_global_all_to_all_reduction(uint32_t value);

extern uint32_t snrt_global_compute_core_num();

extern uint32_t snrt_global_compute_core_idx();
