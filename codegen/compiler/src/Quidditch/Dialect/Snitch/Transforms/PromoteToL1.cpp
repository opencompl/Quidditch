#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_PROMOTEOPERANDSTOL1PASS
#define GEN_PASS_DEF_PROMOTEALLOCSTOL1PASS
#define GEN_PASS_DEF_PROMOTEPADSTOL1PASS
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

class PromotePadsToL1
    : public quidditch::Snitch::impl::PromotePadsToL1PassBase<PromotePadsToL1> {
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
    SmallVector<OpOperand *> nonL1Uses;
    for (OpOperand &operand : computeOp->getOpOperands())
      if (isa<RankedTensorType>(operand.get().getType()))
        nonL1Uses.push_back(&operand);

    auto builder = OpBuilder(computeOp);
    for (OpOperand *use : nonL1Uses) {
      auto copyOp = builder.create<StartTensorCopyOp>(computeOp.getLoc(),
                                                      /*copy=*/use->get());
      auto waitOp = builder.create<WaitForTensorCopyOp>(
          computeOp.getLoc(), copyOp.getResult(), copyOp.getToken(),
          /*copy=*/use->get());
      use->set(waitOp);
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
    auto copyOp = builder.create<StartTensorCopyOp>(tensorOp.getLoc(),
                                                    tensorOp.getCopy());
    auto waitOp = builder.create<WaitForTensorCopyOp>(
        tensorOp.getLoc(), copyOp.getResult(), copyOp.getToken(),
        /*copy=*/tensorOp.getCopy());
    tensorOp.replaceAllUsesWith(waitOp.getResult());
    tensorOp.erase();
  });
}

void PromotePadsToL1::runOnOperation() {
  getOperation()->walk([&](tensor::PadOp padOp) {
    // 'start_tensor_copy' does not yet support lower padding.
    if (!padOp.hasZeroLowPad())
      return;

    Value constant = padOp.getConstantPaddingValue();
    if (!constant)
      return;

    // 'start_tensor_copy' only supports zero-padding right now.
    // Poison (undef) can also be lowered to perform zero-padding.
    if (!matchPattern(constant, m_NonZero()) &&
        !matchPattern(constant, m_PosZeroFloat()) &&
        !matchPattern(constant, m_Constant<ub::PoisonAttr>(nullptr)))
      return;

    OpBuilder builder(padOp);
    auto copyOp = builder.create<StartTensorCopyOp>(
        padOp.getLoc(), padOp.getType(), builder.getType<DMATokenType>(),
        padOp.getSource(), padOp.getHigh(), padOp.getStaticHighAttr());
    auto waitOp = builder.create<WaitForTensorCopyOp>(
        padOp.getLoc(), copyOp.getResult(), copyOp.getToken(),
        /*copy=*/padOp.getSource());
    padOp.replaceAllUsesWith(waitOp.getResult());
    padOp.erase();
  });
}
