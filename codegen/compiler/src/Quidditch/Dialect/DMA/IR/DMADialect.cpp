#include "DMADialect.h"

#include "DMAAttrs.h"
#include "DMAOps.h"
#include "DMATypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Quidditch/Dialect/DMA/IR/DMAAttrs.cpp.inc"

#include "Quidditch/Dialect/DMA/IR/DMADialect.cpp.inc"

using namespace mlir;
using namespace quidditch::dma;

//===----------------------------------------------------------------------===//
// DMADialect
//===----------------------------------------------------------------------===//

void DMADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quidditch/Dialect/DMA/IR/DMAOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Quidditch/Dialect/DMA/IR/DMAAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Quidditch/Dialect/DMA/IR/DMATypes.cpp.inc"
      >();
}

Operation *DMADialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (isa<CompletedTokenAttr>(value))
    return builder.create<CompletedTokenOp>(loc);

  return nullptr;
}
