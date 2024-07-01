#include "Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_FORMMICROKERNELSPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

using namespace mlir;
using namespace quidditch::Snitch;

namespace {
class FormMicrokernels
    : public quidditch::Snitch::impl::FormMicrokernelsPassBase<
          FormMicrokernels> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

static void outlineOpsToFunction(MutableArrayRef<linalg::LinalgOp> ops) {
  if (ops.empty())
    return;

  auto builder = OpBuilder(ops.front());

  SetVector<Value> inputs;
  for (linalg::LinalgOp computeOp : ops) {
    inputs.insert(computeOp->getOperands().begin(),
                  computeOp->getOperands().end());

    computeOp.walk([&](Operation *operation) {
      for (Value value : operation->getOperands()) {
        if (computeOp->getParentRegion()->isProperAncestor(
                value.getParentRegion()))
          continue;

        inputs.insert(value);
      }
    });
  }

  auto kernelOp = builder.create<MemRefMicrokernelOp>(ops.front()->getLoc(),
                                                      inputs.getArrayRef());

  Block *block = kernelOp.createEntryBlock();
  builder.setInsertionPointToStart(block);

  for (Operation *op : ops) {
    op->remove();
    builder.insert(op);
  }

  SmallVector<Value> vector = inputs.takeVector();
  for (auto [oldV, newV] : llvm::zip(vector, block->getArguments()))
    oldV.replaceUsesWithIf(newV, [&](OpOperand &operand) {
      return kernelOp.getBody().isAncestor(
          operand.getOwner()->getParentRegion());
    });
}

void FormMicrokernels::runOnOperation() {
  FunctionOpInterface func = getOperation();

  SmallVector<linalg::LinalgOp> outlinedOps;
  func.walk([&](Block *block) {
    for (Operation &op : *block) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp || !linalgOp.hasPureBufferSemantics()) {
        outlineOpsToFunction(outlinedOps);
        outlinedOps.clear();
        continue;
      }
      outlinedOps.push_back(linalgOp);
    }
  });
}
