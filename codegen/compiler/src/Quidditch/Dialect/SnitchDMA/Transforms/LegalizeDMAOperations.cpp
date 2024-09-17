#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quidditch/Dialect/DMA/IR/DMADialect.h"
#include "Quidditch/Dialect/DMA/IR/DMAOps.h"

namespace quidditch::SnitchDMA {
#define GEN_PASS_DEF_LEGALIZEDMAOPERATIONSPASS
#include "Quidditch/Dialect/SnitchDMA/Transforms/Passes.h.inc"
} // namespace quidditch::SnitchDMA

namespace {
class LegalizeDMAOperations
    : public quidditch::SnitchDMA::impl::LegalizeDMAOperationsPassBase<
          LegalizeDMAOperations> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace quidditch::dma;

/// Returns the number of potentially non-contiguous outer dimensions of
/// 'memRefType'. The remaining inner dimensions (i.e. all dimensions at index
/// 'NonContiguousOuterDims' to the MemRef rank) are known to be contiguous.
/// Returns failure if the layout attribute of the MemRef is unsupported.
static FailureOr<size_t> getNumNonContiguousOuterDims(MemRefType memRefType) {
  auto stridesAttr =
      dyn_cast_or_null<StridedLayoutAttr>(memRefType.getLayout());
  if (!stridesAttr) {
    if (memRefType.getLayout() && !memRefType.getLayout().isIdentity())
      return failure();

    // No layout or identity layouts are by definition fully contiguous.
    return 0;
  }

  int64_t innerSize = 1;
  ArrayRef<int64_t> shape = memRefType.getShape();
  ArrayRef<int64_t> strides = stridesAttr.getStrides();
  for (; !shape.empty();
       shape = shape.drop_back(), strides = strides.drop_back()) {
    int64_t dim = shape.back();
    // Unit dims can be dropped alongside the corresponding stride of that dim.
    if (dim == 1)
      continue;

    int64_t stride = strides.back();
    if (ShapedType::isDynamic(stride))
      break;

    if (innerSize != stride)
      break;

    // Note: Dim may be dynamic with the value -1. This intentionally will only
    // fail the 'if' above later if the outer dims are non-zero.
    innerSize *= dim;
  }

  return shape.size();
}

/// Returns true if this MemRef type is known to have a fully contiguous layout.
/// TODO: Could be upstreamed next to
/// 'memref::isStaticShapeAndContiguousRowMajor'
static bool isContiguous(MemRefType memRefType) {
  return getNumNonContiguousOuterDims(memRefType) == 0;
}

/// Returns true if the given transfer can naively be lowered to Snitch's DMA.
/// Snitch's DMA may be either one or two dimensional where only the innermost
/// dimension is contiguous. The outermost may have arbitrary strides.
static bool isLegal(StartTransferOp transferOp) {
  MemRefType memRef = transferOp.getSource().getType();
  // Only 1 or 2D.
  if (memRef.getRank() > 2)
    return false;

  FailureOr<size_t> sourceNonCont = getNumNonContiguousOuterDims(memRef);
  FailureOr<size_t> destNonCont =
      getNumNonContiguousOuterDims(transferOp.getDest().getType());
  if (failed(sourceNonCont) || failed(destNonCont))
    return false;

  return std::max(*sourceNonCont, *destNonCont) == memRef.getRank() - 1;
}

/// Removes outer unit dimensions from the result type of the subview.
static void rankReduce(memref::SubViewOp op) {
  MemRefType resultType = op.getResult().getType();
  ArrayRef<int64_t> shape = resultType.getShape();
  shape = shape.drop_while([](int64_t value) { return value == 1; });
  MemRefLayoutAttrInterface layout;
  if (auto strided =
          dyn_cast_or_null<StridedLayoutAttr>(resultType.getLayout()))
    layout =
        StridedLayoutAttr::get(op.getContext(), strided.getOffset(),
                               strided.getStrides().take_back(shape.size()));

  op.getResult().setType(MemRefType::get(shape, resultType.getElementType(),
                                         layout, resultType.getMemorySpace()));
};

namespace {
/// Remove outer unit dimensions.
struct RankReduceStartTransferOp : OpRewritePattern<StartTransferOp> {

  using OpRewritePattern<StartTransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StartTransferOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType type = op.getSource().getType();
    if (type.getShape().size() <= 1 || type.getShape().front() != 1)
      return failure();

    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, op->getLoc(), op.getSource());
    SmallVector<OpFoldResult> offsets(sizes.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(sizes.size(), rewriter.getIndexAttr(1));
    auto source = rewriter.create<memref::SubViewOp>(
        op->getLoc(), op.getSource(), offsets, sizes, strides);
    rankReduce(source);
    auto dest = rewriter.create<memref::SubViewOp>(op->getLoc(), op.getDest(),
                                                   offsets, sizes, strides);
    rankReduce(dest);
    rewriter.replaceOpWithNewOp<StartTransferOp>(op, source, dest);
    return success();
  }
};

/// Collapse inner contiguous inner dimensions and remove all but possibly one
/// outer dimension by tiling with size 1 and performing rank-reduction.
struct CollapseStartTransferOp : OpRewritePattern<StartTransferOp> {

  using OpRewritePattern<StartTransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StartTransferOp op,
                                PatternRewriter &rewriter) const override {
    // Must be handled by the rank reduce pattern.
    if (op.getSource().getType().getShape().front() == 1)
      return failure();

    FailureOr<size_t> sourceNonContiguous =
        getNumNonContiguousOuterDims(op.getSource().getType());
    FailureOr<size_t> destNonContiguous =
        getNumNonContiguousOuterDims(op.getDest().getType());
    if (failed(sourceNonContiguous) || failed(destNonContiguous))
      return failure();

    size_t sharedNonContiguous =
        std::max(*sourceNonContiguous, *destNonContiguous);
    // A missing contiguous dimension is handled by the expansion pattern.
    if (sharedNonContiguous == op.getSource().getType().getRank())
      return failure();

    TypedValue<MemRefType> source = op.getSource();
    TypedValue<MemRefType> dest = op.getDest();
    // Collapse multiple contiguous inner dims into one.
    if (sharedNonContiguous + 1 < source.getType().getRank()) {
      SmallVector<ReassociationIndices> reAssociation(sharedNonContiguous + 1);
      for (unsigned index : llvm::seq(sharedNonContiguous))
        reAssociation[index].push_back(index);

      llvm::append_range(
          reAssociation.back(),
          llvm::seq<int64_t>(sharedNonContiguous, source.getType().getRank()));

      source = rewriter.create<memref::CollapseShapeOp>(op->getLoc(), source,
                                                        reAssociation);
      dest = rewriter.create<memref::CollapseShapeOp>(op->getLoc(), dest,
                                                      reAssociation);
    }

    // No, or one outer dim is already legal.
    if (sharedNonContiguous <= 1) {
      rewriter.replaceOpWithNewOp<StartTransferOp>(op, source, dest);
      return success();
    }

    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, op->getLoc(), source);

    // Build a loop nest iterating over all outer dimensions - 1.
    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    for (size_t index : llvm::seq(sharedNonContiguous - 1)) {
      lowerBounds.push_back(zeroIndex);
      steps.push_back(oneIndex);
      upperBounds.push_back(getValueOrCreateConstantIndexOp(
          rewriter, op->getLoc(), sizes[index]));
    }

    Value completedToken = rewriter.create<CompletedTokenOp>(op->getLoc());
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, op->getLoc(), lowerBounds, upperBounds, steps, completedToken,
        [&](OpBuilder &builder, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          SmallVector<OpFoldResult> offsets = ivs;
          SmallVector<OpFoldResult> subSizes(sharedNonContiguous - 1,
                                             rewriter.getIndexAttr(1));
          for (auto index : llvm::seq<unsigned>(sharedNonContiguous - 1,
                                                source.getType().getRank())) {
            offsets.push_back(rewriter.getIndexAttr(0));
            subSizes.push_back(sizes[index]);
          }

          SmallVector<OpFoldResult> strides(source.getType().getRank(),
                                            rewriter.getIndexAttr(1));
          auto sourceMemRefSlice = rewriter.create<memref::SubViewOp>(
              loc, source, offsets, subSizes, strides);
          rankReduce(sourceMemRefSlice);
          auto destMemRefSlice = rewriter.create<memref::SubViewOp>(
              loc, dest, offsets, subSizes, strides);
          rankReduce(destMemRefSlice);
          Value token = rewriter.create<StartTransferOp>(
              op->getLoc(), sourceMemRefSlice, destMemRefSlice);
          return {rewriter.create<CombineTokensOp>(
              op->getLoc(), ValueRange{token, iterArgs.front()})};
        });

    rewriter.replaceOp(op, loopNest.results.front());
    return success();
  }
};

/// Add a contiguous inner unit dim if none exists.
struct ExpandStartTransferOp : OpRewritePattern<StartTransferOp> {

  using OpRewritePattern<StartTransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StartTransferOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<size_t> sourceNonContiguous =
        getNumNonContiguousOuterDims(op.getSource().getType());
    FailureOr<size_t> destNonContiguous =
        getNumNonContiguousOuterDims(op.getDest().getType());
    if (failed(sourceNonContiguous) || failed(destNonContiguous))
      return failure();

    size_t sharedNonContiguous =
        std::max(*sourceNonContiguous, *destNonContiguous);

    TypedValue<MemRefType> source = op.getSource();
    TypedValue<MemRefType> dest = op.getDest();
    // Nothing to do if contiguous inner dims exist.
    if (sharedNonContiguous != source.getType().getRank())
      return failure();

    SmallVector<ReassociationIndices> reAssociation(sharedNonContiguous);
    for (unsigned index : llvm::seq(sharedNonContiguous))
      reAssociation[index].push_back(index);
    reAssociation.back().push_back(sharedNonContiguous);

    auto resultShape = llvm::to_vector(source.getType().getShape());
    resultShape.push_back(1);

    source = rewriter.create<memref::ExpandShapeOp>(op->getLoc(), resultShape,
                                                    source, reAssociation);
    dest = rewriter.create<memref::ExpandShapeOp>(op->getLoc(), resultShape,
                                                  dest, reAssociation);

    rewriter.replaceOpWithNewOp<StartTransferOp>(op, source, dest);
    return success();
  }
};

} // namespace

void LegalizeDMAOperations::runOnOperation() {
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto &&...) { return true; });
  target.addDynamicallyLegalOp<StartTransferOp>(
      [](StartTransferOp op) { return isLegal(op); });

  RewritePatternSet patterns(&getContext());
  patterns.insert<CollapseStartTransferOp, RankReduceStartTransferOp,
                  ExpandStartTransferOp>(&getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
