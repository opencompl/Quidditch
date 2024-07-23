#ifndef QUIDDITCH_DIALECT_SNITCH_TRANSFORMS_PASSES
#define QUIDDITCH_DIALECT_SNITCH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FormMicrokernelsPass
  : InterfacePass<"quidditch-form-microkernels", "mlir::FunctionOpInterface"> {
  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
}

def PromoteOperandsToL1Pass : Pass<"quidditch-promote-operands-to-l1"> {
  let description = [{
    TODO:
  }];

  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
}

def PromoteAllocsToL1Pass : Pass<"quidditch-promote-allocs-to-l1"> {
  let description = [{
    TODO:
  }];

  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
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

def LowerPipelineOpPass : Pass<"quidditch-lower-pipeline-op"> {
  let dependentDialects = [
    "mlir::scf::SCFDialect",
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

def LowerForallOpPass : Pass<"quidditch-lower-forall-op"> {
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
}

#endif
