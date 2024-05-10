#include <stdio.h>
#include <team_decls.h>

int main() {
  if (!snrt_is_dm_core()) return 0;

  printf("Hello World\n");
  return 0;
}
