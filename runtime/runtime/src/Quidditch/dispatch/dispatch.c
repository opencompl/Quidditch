
#include "dispatch.h"

#include <iree/base/alignment.h>

#include <assert.h>
#include <encoding.h>
#include <riscv_decls.h>
#include <ssr_decls.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <sync_decls.h>
#include <team_decls.h>

// TODO: This only works for a single cluster by using globals. Should be
// cluster local.
static iree_hal_executable_dispatch_v0_t configuredKernel;
static const iree_hal_executable_environment_v0_t* configuredEnvironment;
static const iree_hal_executable_dispatch_state_v0_t* configuredDispatchState;
static iree_alignas(64)
    iree_hal_executable_workgroup_state_v0_t configuredWorkgroupState[8];
static atomic_bool error = false;
static atomic_bool exit = false;
static uint8_t nextCoreToUse = 0;

static void reset_workgroup_state() {
  nextCoreToUse = 0;
  // Sentinel value for processor has no workgroup assigned.
  for (iree_hal_executable_workgroup_state_v0_t* iter =
           configuredWorkgroupState;
       iter !=
       configuredWorkgroupState + IREE_ARRAYSIZE(configuredWorkgroupState);
       iter++)
    iter->processor_id = -1;
}

bool quidditch_dispatch_errors_occurred() { return error; }

void quidditch_dispatch_set_kernel(
    iree_hal_executable_dispatch_v0_t kernel,
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state) {
  configuredKernel = kernel;
  configuredEnvironment = environment;
  configuredDispatchState = dispatch_state;
  reset_workgroup_state();
}

int quidditch_dispatch_enter_worker_loop() {
  // Worker communication contract: Workers go into the hardware barrier
  // immediately, DMA core wakes them using a hardware barrier, works wake DMA
  // core using another hardware barrier.

  while (!exit) {
    snrt_cluster_hw_barrier();
    if (exit) break;

    iree_hal_executable_workgroup_state_v0_t* workgroupState =
        &configuredWorkgroupState[snrt_cluster_core_idx()];
    if (workgroupState->processor_id == -1) {
      snrt_cluster_hw_barrier();
      continue;
    }

    read_csr(mcycle);
    if (configuredKernel(configuredEnvironment, configuredDispatchState,
                         workgroupState))
      error = true;

    // Required to make sure that we only read the mcycle once the FPU has
    // actually finished. Otherwise, we are measuring cycles that are too short!
    // This is only required for measurement, not in real programs.
    snrt_fpu_fence();
    read_csr(mcycle);

    // Signal being done.
    snrt_cluster_hw_barrier();
  }
  return 0;
}

void quidditch_dispatch_quit() {
  exit = true;
  snrt_cluster_hw_barrier();
}

void quidditch_dispatch_queue_workgroup(
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  configuredWorkgroupState[nextCoreToUse] = *workgroup_state;
  configuredWorkgroupState[nextCoreToUse].processor_id = nextCoreToUse;
  nextCoreToUse++;
  if (nextCoreToUse != snrt_cluster_compute_core_num()) return;

  quidditch_dispatch_execute_workgroups();
  reset_workgroup_state();
}

void quidditch_dispatch_execute_workgroups() {
  // Avoid needlessly waking any cores if no workgroups are queued.
  if (nextCoreToUse == 0) return;

  // First wake workers.
  snrt_cluster_hw_barrier();
  // Then wait for workers to be done.
  snrt_cluster_hw_barrier();
}
