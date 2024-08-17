#include "QuidditchSnitchDialect.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.cpp.inc"
#include "QuidditchSnitchAttrs.h"
#include "QuidditchSnitchOps.h"
#include "QuidditchSnitchTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.cpp.inc"

using namespace mlir;
using namespace quidditch::Snitch;

static ArrayRef<int64_t> dropTrailingZeros(ArrayRef<int64_t> array) {
  while (!array.empty()) {
    if (array.back() != 0)
      return array;

    array = array.drop_back();
  }
  return array;
}

//===----------------------------------------------------------------------===//
// LoweringConfigAttr::Builders
//===----------------------------------------------------------------------===//

LoweringConfigAttr LoweringConfigAttr::get(MLIRContext *context,
                                           ArrayRef<int64_t> workgroupTiles,
                                           ArrayRef<int64_t> l1Tiles,
                                           bool dualBuffer) {
  return Base::get(context, dropTrailingZeros(workgroupTiles),
                   dropTrailingZeros(l1Tiles), dualBuffer);
}

//===----------------------------------------------------------------------===//
// QuidditchSnitchDialect
//===----------------------------------------------------------------------===//

void QuidditchSnitchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchTypes.cpp.inc"
      >();
}

Operation *QuidditchSnitchDialect::materializeConstant(OpBuilder &builder,
                                                       Attribute value,
                                                       Type type,
                                                       Location loc) {
  if (isa<CompletedTokenAttr>(value))
    return builder.create<CompletedTokenOp>(loc);

  return nullptr;
}
