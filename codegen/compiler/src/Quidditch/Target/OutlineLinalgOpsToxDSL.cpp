#include "Passes.h"

#include <llvm/ADT/ScopeExit.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>

namespace quidditch {
#define GEN_PASS_DEF_OUTLINELINALGOPSTOXDSLPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;

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

static bool canUseBarepointerCC(Type type) {
  auto memRef = dyn_cast<MemRefType>(type);
  if (!memRef)
    return true;
  if (isa<UnrankedMemRefType>(memRef))
    return false;

  int64_t offset = 0;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memRef, strides, offset)))
    return false;

  for (int64_t stride : strides)
    if (ShapedType::isDynamic(stride))
      return false;

  return !ShapedType::isDynamic(offset);
}

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

    SmallVector<Attribute> kernelsGenerated;
    func->setAttr("xdsl_optimized", builder.getUnitAttr());
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

      auto outlinedFunction = builder.create<func::FuncOp>(
          builder.getUnknownLoc(), (func.getSymName() + "$xDSL_kernel").str(),
          builder.getFunctionType(
              llvm::map_to_vector(requiredArguments,
                                  std::mem_fn(&Value::getType)),
              {}));

      // xDSL only supports barepointer lowering right now.
      outlinedFunction->setAttr("llvm.bareptr", builder.getUnitAttr());
      outlinedFunction->setAttr("xdsl_generated", builder.getUnitAttr());
      symbolTable.insert(outlinedFunction);
      kernelsGenerated.push_back(FlatSymbolRefAttr::get(outlinedFunction));

      {
        OpBuilder::InsertionGuard guard{builder};
        builder.setInsertionPointToStart(outlinedFunction.addEntryBlock());

        IRMapping mapping;
        for (auto [old, newV] : llvm::zip_equal(
                 requiredArguments, outlinedFunction.getArguments()))
          mapping.map(old, newV);

        for (Operation *constants : constantsToClone)
          builder.insert(constants->clone(mapping));

        for (Operation *op : ops)
          builder.insert(op->clone(mapping));

        builder.create<func::ReturnOp>(builder.getUnknownLoc());
      }

      // TODO: Add support in xDSL for memrefs with dynamic components
      //  (with calling convention support if needed).
      // Need to check the types here explicitly as LLVM conversion will fail
      // later otherwise. We purposefully do this after insertion of cloned ops
      // to have the IR in the error message.
      if (!llvm::all_of(outlinedFunction.getArgumentTypes(),
                        canUseBarepointerCC)) {
        auto emit = assertCompiled ? &func::FuncOp::emitError
                                   : &func::FuncOp::emitWarning;

        (outlinedFunction.*emit)("function signature ")
            << outlinedFunction.getFunctionType()
            << " does not support bare-pointer calling convention required by "
               "xDSL.";
        // Set as if code generation failed.
        outlinedFunction->removeAttr("xdsl_generated");
        // Stop lying to make LLVM conversion succeed.
        outlinedFunction->removeAttr("llvm.bareptr");
        outlinedFunction.setPrivate();
        outlinedFunction.getBody().getBlocks().clear();
        return;
      }

      OpBuilder::InsertionGuard guard{builder};
      builder.setInsertionPoint(ops.front());
      builder.create<func::CallOp>(builder.getUnknownLoc(), outlinedFunction,
                                   requiredArguments.getArrayRef());
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

    func->setAttr("xdsl_kernels", builder.getArrayAttr(kernelsGenerated));
  }
}
