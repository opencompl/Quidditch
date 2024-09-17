#ifndef QUIDDITCH_DIALECT_SNITCHDMA_IR_SNITCHDMATYPES
#define QUIDDITCH_DIALECT_SNITCHDMA_IR_SNITCHDMATYPES

include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMADialect.td"
include "mlir/IR/AttrTypeBase.td"

class SnitchDMA_Type<string name, list<Trait> traits = []> :
  TypeDef<SnitchDMA_Dialect, name, traits>;

#endif
