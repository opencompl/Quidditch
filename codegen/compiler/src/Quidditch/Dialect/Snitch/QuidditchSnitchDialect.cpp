#include "QuidditchSnitchDialect.h"

#include "QuidditchSnitchOps.h"

#include "Quidditch/Dialect/Snitch/QuidditchSnitchDialect.cpp.inc"

using namespace quidditch::Snitch;

void QuidditchSnitchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quidditch/Dialect/Snitch/QuidditchSnitchOps.cpp.inc"
      >();
}
