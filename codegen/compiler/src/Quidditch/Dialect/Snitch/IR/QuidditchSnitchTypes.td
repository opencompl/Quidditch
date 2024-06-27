#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHTYPES
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHTYPES

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "mlir/IR/AttrTypeBase.td"

class QuidditchSnitch_Type<string name, list<Trait> traits = []> :
  TypeDef<QuidditchSnitch_Dialect, name, traits>;

def QuidditchSnitch_DMATokenType : QuidditchSnitch_Type<"DMAToken"> {
  let mnemonic = "dma_token";

  let description = [{
    Type representing a potentially active DMA transfer.
  }];
}

#endif
