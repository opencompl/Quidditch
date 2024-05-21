#include <string.h>
#include <sys/stat.h>

#include "alloc_decls.h"
#include "riscv_decls.h"
#include "snitch_cluster_defs.h"
#include "team_decls.h"

int close(int fd) { return -1; }

off_t lseek(int file, off_t offset, int whence) { return 0; }

ssize_t read(int fd, void *buf, size_t count) { return 0; }

extern uintptr_t volatile tohost, fromhost;

// Verilator is not able to read any stack memory, but it is capable of reading
// the bss section. Use a global buffer that any 'ptr' in '_write' is copied to.
static struct Buffer {
  uint64_t syscall_mem[8];
  char verilatorReachable[120];
} buffer[SNRT_CLUSTER_CORE_NUM];

ssize_t write(int file, const void *ptr, size_t len) {
  uint32_t id = snrt_hartid();
  int old_len = len;

  do {
    unsigned to_write = len > sizeof(buffer[id].verilatorReachable)
                            ? sizeof(buffer[id].verilatorReachable)
                            : len;
    memcpy(buffer[id].verilatorReachable, ptr, to_write);
    buffer[id].syscall_mem[0] = 64;    // sys_write
    buffer[id].syscall_mem[1] = file;  // file descriptor (1 = stdout)
    buffer[id].syscall_mem[2] =
        (uintptr_t)buffer[id].verilatorReachable;  // buffer
    buffer[id].syscall_mem[3] = to_write;          // length

    tohost = (uintptr_t)buffer[id].syscall_mem;
    while (fromhost == 0)
      ;
    fromhost = 0;

    len -= to_write;
    ptr += to_write;
  } while (len > 0);

  return old_len;
}

extern uint8_t _edram;
static uint8_t *heap = &_edram;

void *sbrk(ptrdiff_t incr) {
  uint8_t *result = heap;
  heap += incr;
  return result;
}

void _exit(int exitCode) {
  asm volatile("wfi");
  __builtin_unreachable();
}
