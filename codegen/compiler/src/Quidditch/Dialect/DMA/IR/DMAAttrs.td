#ifndef QUIDDITCH_DIALECT_DMA_DMAATTRS
#define QUIDDITCH_DIALECT_DMA_DMAATTRS

include "Quidditch/Dialect/DMA/IR/DMADialect.td"
include "mlir/IR/AttrTypeBase.td"

class DMA_Attr<string name, list<Trait> traits = []> :
  AttrDef<DMA_Dialect, name, traits>;

def DMA_CompletedTokenAttr : DMA_Attr<"CompletedToken"> {

  let mnemonic = "completed_token";

  let description = [{
    Attribute representing an instance of a `!dma.token`
    signaling a complete transfer.
  }];
}

#endif
