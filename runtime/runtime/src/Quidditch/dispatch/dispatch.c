
#include "dispatch.h"

#include <assert.h>
#include <cluster_interrupt_decls.h>
#include <encoding.h>
#include <riscv_decls.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <team_decls.h>

#include "iree/base/alignment.h"

// TODO: This should be cluster local.
static struct worker_metadata_t {
  atomic_uint workers_waiting;
  atomic_bool exit;
} worker_metadata = {0, false};

// TODO: All of this synchronization in this file could use hardware barriers
// which might be more efficient.
static void park_worker() {
  worker_metadata.workers_waiting++;
  asm volatile("wfi");
  snrt_int_cluster_clr(1 << snrt_cluster_core_idx());
  worker_metadata.workers_waiting--;
}

static void wake_all_workers() {
  assert(snrt_is_dm_core() && "DM core is currently our host");
  uint32_t compute_cores = snrt_cluster_compute_core_num();
  // Compute cores are indices 0 to compute_cores.
  snrt_int_cluster_set((1 << compute_cores) - 1);
}

void quidditch_dispatch_wait_for_workers() {
  assert(snrt_is_dm_core() && "DM core is currently our host");
  // Spin until all compute corkers are parked.
  while (worker_metadata.workers_waiting != snrt_cluster_compute_core_num())
    ;
}

// TODO: This only works for a single cluster by using globals. Should be
// cluster local.
static iree_hal_executable_dispatch_v0_t configuredKernel;
static const iree_hal_executable_environment_v0_t* configuredEnvironment;
static const iree_hal_executable_dispatch_state_v0_t* configuredDispatchState;
static iree_alignas(64)
    iree_hal_executable_workgroup_state_v0_t configuredWorkgroupState[8];
static atomic_bool error = false;

bool quidditch_dispatch_errors_occurred() { return error; }

void quidditch_dispatch_set_kernel(
    iree_hal_executable_dispatch_v0_t kernel,
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state) {
  configuredKernel = kernel;
  configuredEnvironment = environment;
  configuredDispatchState = dispatch_state;
}

int quidditch_dispatch_enter_worker_loop() {
  snrt_interrupt_enable(IRQ_M_CLUSTER);

  while (!worker_metadata.exit) {
    park_worker();
    if (worker_metadata.exit) break;

    if (configuredKernel(configuredEnvironment, configuredDispatchState,
                         &configuredWorkgroupState[snrt_cluster_core_idx()]))
      error = true;
  }

  snrt_interrupt_disable(IRQ_M_CLUSTER);
  return 0;
}

void quidditch_dispatch_quit() {
  quidditch_dispatch_wait_for_workers();
  worker_metadata.exit = true;
  wake_all_workers();
}

void quidditch_dispatch_submit_workgroup(
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  configuredWorkgroupState[workgroup_state->processor_id] = *workgroup_state;
  snrt_int_cluster_set(1 << workgroup_state->processor_id);
}
