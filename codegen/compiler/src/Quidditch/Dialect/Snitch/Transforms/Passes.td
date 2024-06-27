#ifndef QUIDDITCH_DIALECT_SNITCH_TRANSFORMS_PASSES
#define QUIDDITCH_DIALECT_SNITCH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def PromoteToL1Pass : Pass<"quidditch-promote-to-l1"> {
  let description = [{
    Ensures that all tensors used within a microkernel are placed in L1 memory,
    using copies if necessary.
  }];
}

def LowerL1AllocationsPass : InterfacePass<"quidditch-lower-l1-allocations",
  "mlir::FunctionOpInterface"> {
  let description = [{

  }];

  let options = [
    Option<"l1MemoryBytes", "l1-memory-bytes", "unsigned", "112640", "">,
    Option<"assertCompiled", "assert-compiled", "bool", "false",
           "If true, errors if the kernel does not fit into L1 meory."
           "Otherwise, removes the kernel from the output and emits a warning "
           "instead.">,
  ];
}

def SpecializeDMACodePass : Pass<"quidditch-specialize-dma-code",
  "mlir::ModuleOp"> {

  let description = [{
    Pass performing code specialization for DMA and compute cores while
    inserting required synchronization primitives.

    Every function originally present in the IR will be cloned and turned into
    a "dma" version.
    DMA versions have all compute operations (i.e. `mmemref.microkernel`s)
    removed while the original version has all DMA transfer operations removed.
    Barriers are inserted where data dependencies require either transfers or
    computations to have finished.
  }];
}

#endif
