#ifndef QUIDDITCH_DIALECT_SNITCHDMA_IR_SNITCHDMAOPS
#define QUIDDITCH_DIALECT_SNITCHDMA_IR_SNITCHDMAOPS

include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMADialect.td"
include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMATypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SnitchDMA_Op<string mnemonic, list<Trait> traits = []> :
  Op<SnitchDMA_Dialect, mnemonic, traits>;

def SnitchDMA_QueueResource
  : Resource<"quidditch::SnitchDMA::QueueResource">;

def SnitchDMA_StatOp : SnitchDMA_Op<"stat",
  [MemoryEffects<[MemRead<SnitchDMA_QueueResource>]>]> {

  let description = [{
    Returns the id of the last DMA transfer that has been completed.
  }];

  let results = (outs I32:$completed_id);

  let assemblyFormat = [{
    attr-dict
  }];
}

#endif
