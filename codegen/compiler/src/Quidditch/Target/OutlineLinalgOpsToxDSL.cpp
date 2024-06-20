#include "Passes.h"

#include <llvm/ADT/ScopeExit.h>
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

static bool supportedByxDSL(Operation *operation) {
  return isa<linalg::GenericOp>(operation);
}

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

    auto outlineOpsToFunction = [&](SmallVectorImpl<Operation *> &ops) {
      if (ops.empty())
        return;

      auto exit = llvm::make_scope_exit([&] {
        for (Operation *op : ops)
          op->erase();
        ops.clear();
      });
      std::reverse(ops.begin(), ops.end());

      // TODO: Logic in all of this assumes no results of ops are used outside
      // the outlined operations.

      DenseSet<Value> resultOfOps;
      for (Operation *op : ops)
        resultOfOps.insert(op->getResults().begin(), op->getResults().end());

      SetVector<Operation *> constantsToClone;
      SetVector<Value> requiredArguments;
      for (Operation *op : ops)
        op->walk([&](Operation *operation) {
          for (Value operand : operation->getOperands()) {
            if (!operand.getParentRegion()->isAncestor(op->getParentRegion()))
              continue;

            if (matchPattern(operand, m_Constant())) {
              constantsToClone.insert(operand.getDefiningOp());
              continue;
            }
            if (!resultOfOps.contains(operand))
              requiredArguments.insert(operand);
          }
        });

      {
        OpBuilder::InsertionGuard guard{builder};
        builder.setInsertionPoint(ops.front());

        auto kernelOp = builder.create<quidditch::Snitch::XDSLKernelOp>(
            ops.front()->getLoc(), requiredArguments.getArrayRef());

        Block *block = kernelOp.createEntryBlock();
        builder.setInsertionPointToStart(block);

        IRMapping mapping;
        for (auto [old, newV] :
             llvm::zip_equal(requiredArguments, block->getArguments()))
          mapping.map(old, newV);

        for (Operation *constants : constantsToClone)
          builder.insert(constants->clone(mapping));

        for (Operation *op : ops)
          builder.insert(op->clone(mapping));

        return;
      }
    };

    for (Block &block : func.getBody()) {
      SmallVector<Operation *> outlinedOps;
      for (Operation &op : llvm::reverse(block)) {
        if (supportedByxDSL(&op))
          outlinedOps.push_back(&op);
        else
          outlineOpsToFunction(outlinedOps);
      }
      outlineOpsToFunction(outlinedOps);
    }
  }
}
