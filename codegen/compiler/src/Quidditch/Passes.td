#ifndef QUIDDITCH_PASSES
#define QUIDDITCH_PASSES

include "mlir/Pass/PassBase.td"

def HoistHALOpsToFuncPass
  : Pass<"quidditch-hoist-hal-ops-to-func", "mlir::ModuleOp"> {
  let description = [{
    Pass performing HAL operation outlining.
    For every function in the module this creates another function called "<old-name>$iree_to_xDSL" and moves all HAL
    operations to that function before calling the original function.

    This allows to pass the original function to xDSL for kernel compilation while compiling the new generated function
    can be lowered by IREE.
    This pass can be removed if xDSL were to learn how to lower IREE HAL operations to RISC-V.

    The original function is additionally tagged with a "xdsl_generated" unit attribute.
  }];
}

def FilterForxDSLPass
  : Pass<"quidditch-filter-for-xdsl"> {
  let description = [{
    Pass rewriting operations to workaround xDSL operations.

    Right now these are:
    * Remove `memref.assume_alignment`.
  }];
}

#endif
