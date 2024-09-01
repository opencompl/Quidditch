#include "SnitchDMAOps.h"

#define GET_OP_CLASSES
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMAOps.cpp.inc"

using namespace mlir;
using namespace quidditch::SnitchDMA;

StringRef QueueResource::getName() {
  return "queue";
}
