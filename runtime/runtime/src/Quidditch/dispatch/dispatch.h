
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

/// Causes the host core to wait for all workers to enter a parked state again.
void quidditch_dispatch_wait_for_workers(void);

/// Returns true if any kernel execution of any compute core ever caused an
/// error.
bool quidditch_dispatch_errors_occurred();

/// Configures the kernel, environment and dispatch state to use for subsequent
/// 'quidditch_dispatch_submit_workgroup' calls. It is impossible for a cluster
/// to execute more than one kernel at a time.
void quidditch_dispatch_set_kernel(
    iree_hal_executable_dispatch_v0_t kernel,
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state);

/// Dispatches the compute core with the id 'workgroup_state->processorId' to
/// execute the last configured kernel with the given workgroup state.
void quidditch_dispatch_submit_workgroup(
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state);
