#ifndef QUIDDITCH_DIALECT_SNITCHDMA_IR_SNITCHDMADIALECT
#define QUIDDITCH_DIALECT_SNITCHDMA_IR_SNITCHDMADIALECT

include "mlir/IR/DialectBase.td"

def SnitchDMA_Dialect : Dialect {
  let name = "snitch_dma";
  let cppNamespace = "::quidditch::SnitchDMA";

  let description = [{
    Dialect dealing with all implementation details specific to Snitch's DMA
    engine.
    Used to progressively lower and optimize the `dma` dialect.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let hasConstantMaterializer = 0;
}

#endif
