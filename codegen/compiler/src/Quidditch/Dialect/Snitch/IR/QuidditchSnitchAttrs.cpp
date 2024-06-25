#include "QuidditchSnitchAttrs.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "QuidditchSnitchDialect.h"

#define GET_ATTRDEF_CLASSES
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.cpp.inc"
