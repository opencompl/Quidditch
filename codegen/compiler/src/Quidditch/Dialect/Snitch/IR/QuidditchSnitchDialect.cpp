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

using namespace quidditch::Snitch;

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
