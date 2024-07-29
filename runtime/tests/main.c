#include <memory_decls.h>
#include <riscv_decls.h>
#include <ssr_decls.h>
#include <stdio.h>
#include <sync_decls.h>
#include <team_decls.h>

#include "/home/markus/CLionProjects/Quidditch/runtime/cmake-build-release/snitch_cluster/cluster_gen/snitch_cluster_addrmap.h"
#include "/home/markus/CLionProjects/Quidditch/runtime/cmake-build-release/snitch_cluster/cluster_gen/snitch_cluster_peripheral.h"

#define CLUSTER_PERF_COUNTER_ADDR \
  (CLUSTER_PERIPH_BASE_ADDR +     \
   SNITCH_CLUSTER_PERIPHERAL_PERF_COUNTER_ENABLE_0_REG_OFFSET)

inline uint32_t __attribute__((const)) snrt_cluster_perf_counters_addr() {
  return CLUSTER_PERF_COUNTER_ADDR;
}

#include "/home/markus/CLionProjects/Quidditch/snitch_cluster/sw/snRuntime/src/perf_cnt.h"

const unsigned REDUCTION_SIZE = 100;
const unsigned ROWS = 5;
const enum snrt_perf_cnt_type METRIC = SNRT_PERF_CNT_TCDM_CONGESTED;

// memref<1x161xf64>, memref<5x161xf64>, memref<1x5xf64>
void kernel161(double vector[], double (*weights)[REDUCTION_SIZE],
               double output[]);

// memref<1x100xf64>, memref<5x100xf64>, memref<1x5xf64>
// In NsNet2 148590 conflicts across 122 streaming regions
// (1 fill, 120 core computation and 1 elementwise post-processing).
// Roughly 1217 conflicts per region with one core performing 500 fmadds and
// 600 memory accesses using the streams.
// Measured here: 2162.
void kernel100(double vector[], double (*weights)[REDUCTION_SIZE],
               double output[]);

void kernel100_stride112(double vector[], double (*weights)[REDUCTION_SIZE],
                         double output[]);

unsigned alignTo(unsigned value, unsigned multiple) {
  unsigned int rem = value % multiple;
  if (rem == 0) return value;
  value += multiple - rem;
  return value;
}

double* padToBank(const double* ptr, unsigned bank) {
  bank &= ~1u;

  size_t address = (size_t)ptr;
  size_t current_bank = (address / 4) % 32;
  if (bank < current_bank) bank += 32;
  address += 4 * (bank - current_bank);
  return (double*)address;
}

void streamerSetup3();

int main() {
  if (snrt_is_dm_core()) {
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
    return 0;
  }

  unsigned id = snrt_cluster_core_idx();
  double* output = (double*)snrt_l1_start_addr() + id * ROWS;
  double* vector = (double*)snrt_l1_start_addr() + 8 * ROWS;
  vector = padToBank(vector, /*bank=*/0);
  double* vector_end = vector + REDUCTION_SIZE;
  vector_end = padToBank(vector_end, 16);
  // for (int i = 0; i < REDUCTION_SIZE; i++) {
  //   vector[i] = 1;
  // }

  // Contiguous layout between the different tiles.
  unsigned dim1Stride = alignTo(REDUCTION_SIZE, 16);
  unsigned dim1Index = id * ROWS;
  double(*weights)[dim1Stride] =
      (double(*)[dim1Stride])(vector_end + dim1Index * dim1Stride);
  // for (int j = 0; j < ROWS; j++) {
  //   for (int i = 0; i < REDUCTION_SIZE; i++) {
  //     weights[j][i] = id;
  //   }
  // }

  volatile uint32_t* pmtx = snrt_mutex();
  snrt_mutex_acquire(pmtx);
  if (id == 0) {
    printf("Vector preload will hit banks %d, %d, %d, %d\n",
           (((size_t)vector) / 4) % 32, ((size_t)(vector + 1) / 4) % 32,
           ((size_t)(vector + 2) / 4) % 32, ((size_t)(vector + 3) / 4) % 32);
  }
  printf("Weights[%d] preload will hit banks %d, %d, %d, %d\n", id,
         (((size_t)weights) / 4) % 32, (((size_t)weights[1]) / 4) % 32,
         (((size_t)weights[2]) / 4) % 32, (((size_t)weights[3]) / 4) % 32);
  snrt_mutex_release(pmtx);

  streamerSetup3();

  if (id == 0) {
    snrt_reset_perf_counter(SNRT_PERF_CNT0);
    snrt_start_perf_counter(SNRT_PERF_CNT0, METRIC, 0);
  }

  // Sync PC.
  uint32_t r;
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
  if (id < 8) kernel100(vector, weights, output);

  snrt_fpu_fence();
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");

  if (id == 0) snrt_stop_perf_counter(SNRT_PERF_CNT0);

  weights = (double(*)[REDUCTION_SIZE])(((double*)0x1000a5c0) +
                                        id * 5 * REDUCTION_SIZE);

  snrt_cluster_hw_barrier();

  if (id == 0) printf("%" PRId32 "\n", snrt_get_perf_counter(SNRT_PERF_CNT0));

  if (id == 0) {
    snrt_reset_perf_counter(SNRT_PERF_CNT0);
    snrt_start_perf_counter(SNRT_PERF_CNT0, METRIC, 0);
  }

  // Sync PC.
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");
  if (id < 8) kernel100(vector, weights, output);

  snrt_fpu_fence();
  asm volatile("csrr %0, mcycle" : "=r"(r) : : "memory");

  if (id == 0) snrt_stop_perf_counter(SNRT_PERF_CNT0);

  snrt_cluster_hw_barrier();

  // Old: 2118. No bubbles.
  // New: 1931, first time, 2118 second. Bubbles at the start of the first time.
  if (id == 0) printf("%" PRId32 "\n", snrt_get_perf_counter(SNRT_PERF_CNT0));

  return 0;
}
