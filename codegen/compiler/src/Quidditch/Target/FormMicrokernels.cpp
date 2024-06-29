#include "Passes.h"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

namespace quidditch {
#define GEN_PASS_DEF_FORMMICROKERNELSPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace quidditch::Snitch;

namespace {
class FormMicrokernels
    : public quidditch::impl::FormMicrokernelsPassBase<FormMicrokernels> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

static void outlineOpsToFunction(MutableArrayRef<linalg::LinalgOp> ops) {
  if (ops.empty())
    return;

  SetVector<Value> escapingResults;
  for (Operation *op : ops)
    for (Value result : op->getResults())
      for (OpOperand &use : result.getUses())
        if (!llvm::is_contained(ops, use.getOwner()))
          escapingResults.insert(use.get());

  auto builder = OpBuilder(ops.front());

  auto kernelOp = builder.create<TensorMicrokernelOp>(
      ops.front()->getLoc(),
      llvm::map_to_vector(escapingResults, std::mem_fn(&Value::getType)));

  Block *block = &kernelOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(block);

  for (Operation *op : ops) {
    op->remove();
    builder.insert(op);
  }

  builder.create<MicrokernelYieldOp>(ops.back().getLoc(),
                                     escapingResults.getArrayRef());

  SmallVector<Value> vector = escapingResults.takeVector();
  for (auto [index, value] : llvm::enumerate(vector))
    value.replaceUsesWithIf(kernelOp.getResult(index), [&](OpOperand &operand) {
      return !kernelOp.getBody().isAncestor(
          operand.getOwner()->getParentRegion());
    });
}

void FormMicrokernels::runOnOperation() {
  FunctionOpInterface func = getOperation();

  SmallVector<linalg::LinalgOp> outlinedOps;
  func.walk([&](Block *block) {
    for (Operation &op : *block) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp) {
        outlineOpsToFunction(outlinedOps);
        outlinedOps.clear();
        continue;
      }
      outlinedOps.push_back(linalgOp);
    }
  });
}
