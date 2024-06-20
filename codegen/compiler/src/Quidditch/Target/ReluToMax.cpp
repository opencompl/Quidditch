#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_RELUTOMAXPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;

namespace {
class ReluToMax : public quidditch::impl::ReluToMaxPassBase<ReluToMax> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

// Apparently arith.cmpf do not have a canonical representation for either LT
// or GT.

struct ReluSelectToMaxLTPattern : OpRewritePattern<arith::SelectOp> {

  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cmpFOp = op.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmpFOp)
      return failure();
    if (cmpFOp.getPredicate() != arith::CmpFPredicate::ULT)
      return failure();
    if (cmpFOp.getRhs() != op.getTrueValue())
      return failure();
    if (cmpFOp.getLhs() != op.getFalseValue())
      return failure();

    rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, cmpFOp.getLhs(),
                                                   cmpFOp.getRhs());
    return success();
  }
};

struct ReluSelectToMaxGTPattern : OpRewritePattern<arith::SelectOp> {

  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cmpFOp = op.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmpFOp)
      return failure();
    if (cmpFOp.getPredicate() != arith::CmpFPredicate::UGT)
      return failure();
    if (cmpFOp.getRhs() != op.getFalseValue())
      return failure();
    if (cmpFOp.getLhs() != op.getTrueValue())
      return failure();

    rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, cmpFOp.getLhs(),
                                                   cmpFOp.getRhs());
    return success();
  }
};

} // namespace

void ReluToMax::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<ReluSelectToMaxLTPattern, ReluSelectToMaxGTPattern>(
      &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
