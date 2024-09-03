#ifndef QUIDDITCH_DIALECT_SNITCHDMA_TRANSFORMS_PASSES
#define QUIDDITCH_DIALECT_SNITCHDMA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LegalizeDMAOperationsPass : Pass<"quidditch-snitch-legalize-dma-operations"> {
  let description = [{

  }];

  let dependentDialects = [
    "quidditch::dma::DMADialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif
