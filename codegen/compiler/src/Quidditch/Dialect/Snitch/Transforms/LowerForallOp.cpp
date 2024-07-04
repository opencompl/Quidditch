#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_LOWERFORALLOPPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class LowerForallOp
    : public quidditch::Snitch::impl::LowerForallOpPassBase<LowerForallOp> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace quidditch::Snitch;

void LowerForallOp::runOnOperation() {
  getOperation()->walk([&](scf::ForallOp forallOp) {
    std::optional<IntegerAttr> attr = getConfigIntegerAttr(
        IREE::HAL::ExecutableTargetAttr::lookup(forallOp), "compute_cores");
    if (!attr)
      return;

    // Nothing else supported right now.
    if (forallOp.getInductionVars().size() != 1)
      return;

    // No longer needed in any of the below code.
    forallOp.getTerminator().erase();

    OpBuilder builder(forallOp);
    Value id = builder.create<ClusterIndexOp>(forallOp.getLoc());

    Value lb = forallOp.getLowerBound(builder).front();
    Value ub = forallOp.getUpperBound(builder).front();
    Value step = forallOp.getStep(builder).front();
    lb = builder.create<affine::AffineApplyOp>(
        forallOp.getLoc(),
        AffineMap::get(
            3, 0,
            {builder.getAffineDimExpr(0) +
             (builder.getAffineDimExpr(1) * builder.getAffineDimExpr(2))},
            &getContext()),
        ValueRange{lb, id, step});
    step = builder.create<affine::AffineApplyOp>(
        forallOp.getLoc(),
        AffineMap::get(
            1, 0,
            {builder.getAffineConstantExpr(attr->getValue().getSExtValue()) *
             builder.getAffineDimExpr(0)},
            &getContext()),
        step);
    auto forOp = builder.create<scf::ForOp>(forallOp.getLoc(), lb, ub, step);

    forOp.getRegion().takeBody(forallOp.getRegion());
    builder.setInsertionPointToEnd(forOp.getBody());
    builder.create<scf::YieldOp>(forallOp.getLoc());

    forallOp.erase();
  });
}
