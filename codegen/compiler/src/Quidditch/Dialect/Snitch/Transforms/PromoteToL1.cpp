#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_PROMOTEOPERANDSTOL1PASS
#define GEN_PASS_DEF_PROMOTEALLOCSTOL1PASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class PromoteOperandsToL1
    : public quidditch::Snitch::impl::PromoteOperandsToL1PassBase<
          PromoteOperandsToL1> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

class PromoteAllocsToL1
    : public quidditch::Snitch::impl::PromoteAllocsToL1PassBase<
          PromoteAllocsToL1> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace quidditch::Snitch;

void PromoteOperandsToL1::runOnOperation() {
  // Copy all tensors used as operands to compute ops into L1 memory.
  getOperation()->walk([&](TilingInterface computeOp) {
    // Note: This can create redundant copies that must be cleaned up by CSE.
    SetVector<TypedValue<RankedTensorType>> nonL1Uses;
    for (Value operand : computeOp->getOperands())
      if (isa<RankedTensorType>(operand.getType()))
        nonL1Uses.insert(cast<TypedValue<RankedTensorType>>(operand));

    auto builder = OpBuilder(computeOp);
    for (TypedValue<RankedTensorType> value : nonL1Uses) {
      auto copyOp = builder.create<CopyTensorOp>(computeOp.getLoc(),
                                                 /*copy=*/value);
      value.replaceAllUsesExcept(copyOp, copyOp);
    }
  });
}

void PromoteAllocsToL1::runOnOperation() {
  // Change all allocas so far to be L1 allocations.
  getOperation()->walk([&](bufferization::AllocTensorOp tensorOp) {
    if (!tensorOp.getCopy()) {
      tensorOp.setMemorySpaceAttr(L1EncodingAttr::get(tensorOp.getContext()));
      return;
    }

    OpBuilder builder(tensorOp);
    Value replacement =
        builder.create<CopyTensorOp>(tensorOp.getLoc(), tensorOp);
    tensorOp.replaceAllUsesWith(replacement);
    tensorOp.erase();
  });
}
