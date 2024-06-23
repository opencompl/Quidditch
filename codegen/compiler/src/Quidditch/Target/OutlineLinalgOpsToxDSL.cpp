#include "Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>

#include <Quidditch/Dialect/Snitch/QuidditchSnitchDialect.h>
#include <Quidditch/Dialect/Snitch/QuidditchSnitchOps.h>

namespace quidditch {
#define GEN_PASS_DEF_OUTLINELINALGOPSTOXDSLPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace quidditch::Snitch;

namespace {
class OutlineLinalgOpsToxDSL
    : public quidditch::impl::OutlineLinalgOpsToxDSLPassBase<
          OutlineLinalgOpsToxDSL> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

void OutlineLinalgOpsToxDSL::runOnOperation() {
  OpBuilder builder(&getContext());
  builder.setInsertionPointToEnd(getOperation().getBody());

  SymbolTable symbolTable(getOperation());
  FunctionType emptyFunctionType = builder.getFunctionType({}, {});
  for (auto func : llvm::to_vector(getOperation().getOps<func::FuncOp>())) {
    // IREE functions all have empty function types with public visibility. Skip
    // over any other functions.
    if (func.getFunctionType() != emptyFunctionType || !func.isPublic())
      continue;

    // We add this suffix for tooling to know whether the kernel was xDSL
    // compiled.
    func.setSymName((func.getSymName() + "$iree_to_xdsl").str());

    auto outlineOpsToFunction =
        [&](SmallVectorImpl<DestinationStyleOpInterface> &ops) {
          if (ops.empty())
            return;

          std::reverse(ops.begin(), ops.end());

          SetVector<Value> escapingResults;
          for (Operation *op : ops)
            for (Value result : op->getResults())
              for (OpOperand &use : result.getUses())
                if (!llvm::is_contained(ops, use.getOwner()))
                  escapingResults.insert(use.get());

          OpBuilder::InsertionGuard guard{builder};
          builder.setInsertionPoint(ops.front());

          auto kernelOp =
              builder.create<quidditch::Snitch::TensorMicrokernelOp>(
                  ops.front()->getLoc(),
                  llvm::map_to_vector(escapingResults,
                                      std::mem_fn(&Value::getType)));

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
            value.replaceUsesWithIf(
                kernelOp.getResult(index), [&](OpOperand &operand) {
                  return operand.getOwner()->getParentRegion() !=
                         &kernelOp.getBody();
                });
        };

    SmallVector<DestinationStyleOpInterface> outlinedOps;
    for (Block &block : func.getBody()) {
      auto range = llvm::reverse(block.getOps<DestinationStyleOpInterface>());
      outlinedOps.assign(range.begin(), range.end());
      outlineOpsToFunction(outlinedOps);
    }
  }
}
