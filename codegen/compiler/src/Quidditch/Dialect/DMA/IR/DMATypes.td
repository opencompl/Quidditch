#ifndef QUIDDITCH_DIALECT_SNITCH_DMATYPES
#define QUIDDITCH_DIALECT_SNITCH_DMATYPES

include "Quidditch/Dialect/DMA/IR/DMADialect.td"
include "mlir/IR/AttrTypeBase.td"

class DMA_Type<string name, list<Trait> traits = []> :
  TypeDef<DMA_Dialect, name, traits>;

def DMA_TokenType : DMA_Type<"Token"> {
  let mnemonic = "token";

  let description = [{
    Type representing a potentially active DMA transfer.
  }];
}

#endif
