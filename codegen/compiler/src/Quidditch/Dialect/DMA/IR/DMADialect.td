#ifndef QUIDDITCH_DIALECT_SNITCH_DMADIALECT
#define QUIDDITCH_DIALECT_SNITCH_DMADIALECT

include "mlir/IR/DialectBase.td"

def DMA_Dialect : Dialect {
  let name = "dma";
  let cppNamespace = "::quidditch::dma";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

#endif
