#include "SnitchDMADialect.h"

#include "SnitchDMAAttrs.h"
#include "SnitchDMAOps.h"
#include "SnitchDMATypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMAAttrs.cpp.inc"

#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMADialect.cpp.inc"

using namespace mlir;
using namespace quidditch::SnitchDMA;

//===----------------------------------------------------------------------===//
// DMADialect
//===----------------------------------------------------------------------===//

void SnitchDMADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMAOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMAAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMATypes.cpp.inc"
      >();
}
