#include "Passes.h"

#include <iree/compiler/Dialect/HAL/IR/HALDialect.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/IRMapping.h>

namespace quidditch {
#define GEN_PASS_DEF_HOISTHALOPSTOFUNCPASS
#include "Quidditch/Passes.h.inc"
} // namespace quidditch

using namespace mlir;

namespace {
class HoistHALOpsToFunc
    : public quidditch::impl::HoistHALOpsToFuncPassBase<HoistHALOpsToFunc> {
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

void HoistHALOpsToFunc::runOnOperation() {
  OpBuilder builder(&getContext());
  builder.setInsertionPointToEnd(getOperation().getBody());

  Dialect *dialect =
      getContext()
          .getLoadedDialect<mlir::iree_compiler::IREE::HAL::HALDialect>();
  if (!dialect)
    return;

  FunctionType emptyFunctionType = builder.getFunctionType({}, {});
  for (auto func : llvm::to_vector(getOperation().getOps<func::FuncOp>())) {
    // IREE functions all have empty function types with public visibility. Skip
    // over any other functions.
    if (func.getFunctionType() != emptyFunctionType || !func.isPublic())
      continue;

    func->setAttr("xdsl_generated", builder.getUnitAttr());
    // xDSL only supports barepointer lowering right now.
    func->setAttr("llvm.bareptr", builder.getUnitAttr());

    // Find all HAL operations that need to be hoisted and any other operations
    // they depend on.
    SmallVector<Operation *> halOperations;
    SetVector<Operation *> toClone;
    SmallVector<std::pair<Value, Value>> queueReplacements;
    func.getFunctionBody().walk([&](Operation *operation) {
      if (operation->getDialect() != dialect)
        return;

      halOperations.push_back(operation);
      for (Value result : operation->getResults()) {
        unsigned int index = func.getNumArguments();
        func.insertArgument(index, result.getType(),
                            builder.getDictionaryAttr({}), result.getLoc());
        queueReplacements.emplace_back(result, func.getArgument(index));
      }

      // Include all operations that the HAL operation transitively depends on.
      BackwardSliceOptions options;
      options.inclusive = true;
      // This inserts into the SetVector in topological order. As we are going
      // through HAL operations in-order, it is guaranteed that any operation
      // already contained in the set appears prior to the HAL operation.
      mlir::getBackwardSlice(operation, &toClone, options);
    });

    auto wrapper = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), (func.getName() + "$iree_to_xdsl").str(),
        emptyFunctionType);

    OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(wrapper.addEntryBlock());
    // Create the clones and persist mapping between clones. This makes sure
    // all operands are remapped.
    IRMapping mapping;
    for (Operation *op : toClone) {
      if (!mlir::isPure(op)) {
        op->emitError("Pass does not expect HAL operations to depend on "
                      "impure operations");
        signalPassFailure();
        return;
      }

      builder.insert(op->clone(mapping));
    }

    SmallVector<Value> arguments;
    for (Operation *op : halOperations)
      for (Value value : op->getResults())
        arguments.push_back(mapping.lookup(value));

    // TODO: Add support in xDSL for memrefs with dynamic components
    //  (with calling convention support if needed).
    // Need to check the types here explicitly as LLVM conversion will fail
    // later otherwise.
    if (!llvm::all_of(llvm::map_range(arguments, std::mem_fn(&Value::getType)),
                      canUseBarepointerCC)) {
      auto emit = assertCompiled ? &func::FuncOp::emitError
                                 : &func::FuncOp::emitWarning;

      (func.*emit)("function signature ")
          << func.getFunctionType()
          << " does not support bare-pointer calling convention required by "
             "xDSL.";

      builder.create<func::ReturnOp>(wrapper.getLoc());

      for (auto [result, replacement] : queueReplacements)
        result.replaceAllUsesWith(replacement);

      // Set as if code generation failed.
      func.getBody().getBlocks().clear();
      func.setVisibility(SymbolTable::Visibility::Private);
      func->removeAttr("xdsl_generated");
      // Stop lying to make LLVM conversion succeed.
      func->removeAttr("llvm.bareptr");
      continue;
    }

    builder.create<func::CallOp>(wrapper.getLoc(), func, arguments);

    builder.create<func::ReturnOp>(wrapper.getLoc());

    for (auto [result, replacement] : queueReplacements)
      result.replaceAllUsesWith(replacement);
  }
}
