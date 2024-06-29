#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_PROMOTETOL1PASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class PromoteToL1
    : public quidditch::Snitch::impl::PromoteToL1PassBase<PromoteToL1> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace quidditch::Snitch;

void PromoteToL1::runOnOperation() {
  // Change all allocas so far to be L1 allocations.
  getOperation()->walk([&](bufferization::AllocTensorOp tensorOp) {
    if (!tensorOp.getCopy()) {
      tensorOp.setMemorySpaceAttr(L1EncodingAttr::get(tensorOp.getContext()));
      return;
    }

    OpBuilder builder(tensorOp);
    Value replacement = builder.create<CopyTensorOp>(
        tensorOp.getLoc(), tensorOp, /*transfers_to_l1=*/true);
    tensorOp.replaceAllUsesWith(replacement);
    tensorOp.erase();
  });

  getOperation()->walk([&](TensorMicrokernelOp microkernelOp) {
    SetVector<TypedValue<RankedTensorType>> nonL1Uses;
    microkernelOp->walk([&](Operation *operation) {
      for (Value operand : operation->getOperands())
        if (isa<RankedTensorType>(operand.getType()) &&
            !microkernelOp.getBody().isAncestor(operand.getParentRegion()))
          nonL1Uses.insert(cast<TypedValue<RankedTensorType>>(operand));
    });

    if (nonL1Uses.empty())
      return;

    // Create copies into L1 for all tensors used in the kernel.
    auto builder = OpBuilder(microkernelOp);
    for (TypedValue<RankedTensorType> value : nonL1Uses) {
      auto copyOp = builder.create<CopyTensorOp>(microkernelOp.getLoc(),
                                                 /*copy=*/value,
                                                 /*transfers_to_l1=*/true);
      value.replaceUsesWithIf(copyOp, [&](OpOperand &operand) {
        return microkernelOp.getBody().isAncestor(
            operand.getOwner()->getParentRegion());
      });
    }
  });
}
