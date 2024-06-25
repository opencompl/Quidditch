#include "QuidditchSnitchDialect.h"

#include "QuidditchSnitchAttrs.h"
#include "QuidditchSnitchOps.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.cpp.inc"

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
}
