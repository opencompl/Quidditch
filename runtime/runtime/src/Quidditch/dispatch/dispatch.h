
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "iree/hal/local/executable_library.h"

/// Entry point for compute cores to be parked and called upon for kernel
/// execution. Cores are halted within the function until
/// 'quidditch_dispatch_quit' is called.
int quidditch_dispatch_enter_worker_loop(void);

/// Called by the host core before exiting to release all computes cores from
/// the work loop.
void quidditch_dispatch_quit(void);

/// Returns true if any kernel execution of any compute core ever caused an
/// error.
bool quidditch_dispatch_errors_occurred();

/// Configures the kernel, environment and dispatch state to use for subsequent
/// 'quidditch_dispatch_queue_workgroup' calls. It is impossible for a cluster
/// to execute more than one kernel at a time.
void quidditch_dispatch_set_kernel(
    iree_hal_executable_dispatch_v0_t kernel,
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state);

/// Queues a workgroup on a compute core with the last configured kernel and
/// dispatch state. May block and execute if all cores have a workgroup
/// assigned to them.
void quidditch_dispatch_queue_workgroup(
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state);

/// Executes all queued workgroups and waits for them to finish.
void quidditch_dispatch_execute_workgroups();
