/*#include <stdio.h>*/
/*#include <team_decls.h>*/
/*#include "snax_rt.h"*/
#include <snrt.h>

int main() {
  if (!snrt_is_dm_core())
    return 0;

  printf("Hello SNAX World\n");
  return 0;
}
