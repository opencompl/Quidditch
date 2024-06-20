#include "QuidditchSnitchOps.h"

#include "mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "Quidditch/Dialect/Snitch/QuidditchSnitchOps.cpp.inc"

using namespace mlir;
using namespace quidditch::Snitch;

Block *XDSLKernelOp::createEntryBlock() {
  assert(getBody().getBlocks().empty());
  Block &block = getBody().emplaceBlock();
  block.addArguments(getInputs().getTypes(),
                     SmallVector<Location>(getInputs().size(), getLoc()));
  return &block;
}

LogicalResult XDSLKernelOp::verify() {
  // TODO: Weaken this in the future to likely only require element count and
  //  type but not the shape. TBD.
  if (getBody().getArgumentTypes() != getInputs().getTypes())
    return emitOpError("type of arguments and inputs must match");
  return success();
}
