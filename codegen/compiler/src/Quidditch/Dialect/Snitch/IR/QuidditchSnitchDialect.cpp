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
                                           ArrayRef<int64_t> l1TilesInterchange,
                                           bool dualBuffer) {
  l1Tiles = dropTrailingZeros(l1Tiles);
  auto interchange = llvm::to_vector(l1TilesInterchange);
  llvm::erase_if(interchange,
                 [&](int64_t value) { return value >= l1Tiles.size(); });
  return Base::get(context, dropTrailingZeros(workgroupTiles), l1Tiles,
                   interchange, dualBuffer);
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
