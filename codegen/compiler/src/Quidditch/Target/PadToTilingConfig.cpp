#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_PADTOTILINGCONFIGPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace quidditch::Snitch;
using namespace mlir::iree_compiler;

namespace {
class PadToTilingConfig
    : public quidditch::impl::PadToTilingConfigPassBase<PadToTilingConfig> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

/// Returns true if it is legal to zero-pad the given linalg operation.
/// Legal is defined as being able to extend the iteration space and the
/// corresponding operands using zero-padding without changing the values
/// in the slice corresponding to the ops original result.
static bool canZeroPad(linalg::LinalgOp linalgOp) {
  // Elementwise operations can be padded with any value as there are no cross
  // dimension data dependencies.
  if (linalgOp.getNumParallelLoops() == linalgOp.getNumLoops())
    return true;

  // Contractions can be zero padded.
  if (linalg::isaContractionOpInterface(linalgOp))
    return true;

  // Convolutions can be zero padded.
  return linalg::isaConvolutionOpInterface(linalgOp);
}

static LogicalResult padToMultipleOf(linalg::LinalgOp &linalgOp,
                                     SmallVector<int64_t> config) {
  for (int64_t &value : config)
    if (value == 0)
      value = 1;

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(
              llvm::to_vector(llvm::seq<int64_t>(config.size())))
          .setPadToMultipleOf(config)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

  auto loweringConfig = getLoweringConfig<LoweringConfigAttr>(linalgOp);

  auto builder = IRRewriter(linalgOp);
  SmallVector<tensor::PadOp> padOps;
  linalg::LinalgOp oldOp = linalgOp;
  SmallVector<Value> replacements;
  if (failed(linalg::rewriteAsPaddedOp(builder, linalgOp, options, linalgOp,
                                       replacements, padOps)))
    return failure();
  builder.replaceOp(oldOp, replacements);

  if (loweringConfig)
    setLoweringConfig(linalgOp, loweringConfig);
  return success();
}

static LogicalResult padToTileSize(linalg::LinalgOp &linalgOp,
                                   std::optional<IntegerAttr> computeCores) {
  SmallVector<int64_t> tileSize;
  if (auto loweringConfig = getLoweringConfig<LoweringConfigAttr>(linalgOp)) {
    for (auto getTileSizeMem : {&LoweringConfigAttr::getWorkgroupTiles,
                                &LoweringConfigAttr::getL1Tiles}) {
      tileSize = llvm::to_vector((loweringConfig.*getTileSizeMem)());
      size_t numLoops = linalgOp.getNumLoops();
      while (tileSize.size() < numLoops)
        tileSize.push_back(0);

      if (failed(padToMultipleOf(linalgOp, tileSize)))
        return failure();
    }
  }

  if (!computeCores)
    return success();

  // TODO: This duplicates the logic for thread tiling risking them to go out of
  //       sync.
  //       We probably want 'LoweringConfigAttr' to include these tile sizes
  //       in the future as well.
  if (tileSize.empty())
    tileSize = linalgOp.getStaticLoopRanges();

  std::optional<unsigned> largestParallelDim;
  std::optional<int64_t> largestParallelSize;
  for (auto [index, iterType, range] :
       llvm::enumerate(linalgOp.getIteratorTypesArray(), tileSize)) {
    // Not doing reduction tiling.
    if (iterType == utils::IteratorType::reduction) {
      range = 0;
      continue;
    }

    // Not tileable.
    if (range <= 1) {
      range = 0;
      continue;
    }

    // Not tiling dynamic dimensions right now.
    if (range == ShapedType::kDynamic) {
      range = 0;
      continue;
    }

    if (!largestParallelSize || range > largestParallelSize) {
      largestParallelDim = index;
      largestParallelSize = range;
    }
  }

  if (largestParallelDim) {
    assert(largestParallelSize);
    tileSize[*largestParallelDim] = llvm::divideCeil(
        *largestParallelSize, computeCores->getValue().getSExtValue());
  }

  return padToMultipleOf(linalgOp, std::move(tileSize));
}

/// Returns true if the given pad operation uses an undefined value as padding
/// value.
static bool hasUndefPadding(tensor::PadOp padOp) {
  Value constant = padOp.getConstantPaddingValue();
  return constant &&
         matchPattern(constant, m_Constant<ub::PoisonAttrInterface>(nullptr));
}

/// Clones 'padOp' using 'rewriter' and replaces its padding value with an
/// undefined value.
static tensor::PadOp cloneWithUndefPad(PatternRewriter &rewriter,
                                       tensor::PadOp padOp) {
  auto newPadOp = cast<tensor::PadOp>(rewriter.cloneWithoutRegions(*padOp));
  {
    OpBuilder::InsertionGuard guard{rewriter};
    rewriter.setInsertionPointToEnd(&newPadOp.getRegion().emplaceBlock());
    for (unsigned _ : llvm::seq(padOp.getSource().getType().getRank()))
      newPadOp.getRegion().addArgument(rewriter.getIndexType(),
                                       padOp->getLoc());

    // TODO: This is very wrong as poison is stronger than undef as there are
    //       operations where a poison value will cause immediate undefined
    //       behaviour where an undef value wouldn't.
    //       Our lowering does the equivalent of using an undef value for now
    //       but things like folding won't respect it.
    //       The correct fix would be to have `ub.freeze` upstream.
    Value poison = rewriter.create<ub::PoisonOp>(
        newPadOp->getLoc(), padOp.getType().getElementType());
    rewriter.create<tensor::YieldOp>(padOp->getLoc(), poison);
  }
  return newPadOp;
}

namespace {

// Patterns applied on linalg operations to turn zero pads created by the
// padding rewriter into undef pads.
// Note that these assume that padding of results are not consumed in a way
// where its semantics have any impact on the final output.
// In simplified words: The linalg's result is only used by the `extract_slice`
// which extracts the slice corresponding to the original unpadded computation.
//
// TODO: This might be cleaner to implement right after or during the padding
//       rewrite.

/// Optimizes every operand of an elementwise operation to be undef padded.
struct OptimizeElementwisePad : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Only elementwise operations.
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
      return failure();

    bool changed = false;
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      auto padOp = operand.get().getDefiningOp<tensor::PadOp>();
      if (!padOp)
        continue;

      if (hasUndefPadding(padOp))
        continue;

      auto newPadOp = cloneWithUndefPad(rewriter, padOp);
      rewriter.modifyOpInPlace(linalgOp, [&] { operand.set(newPadOp); });
      changed = true;
    }
    return success(changed);
  }
};

/// Optimizes the init operand of a contraction operation to be undef padded.
struct OptimizeContractionOutputPad
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(linalgOp))
      return failure();

    bool changed = false;
    for (OpOperand &operand : linalgOp.getDpsInitsMutable()) {
      auto padOp = operand.get().getDefiningOp<tensor::PadOp>();
      if (!padOp)
        continue;

      if (hasUndefPadding(padOp))
        continue;

      auto newPadOp = cloneWithUndefPad(rewriter, padOp);
      rewriter.modifyOpInPlace(linalgOp, [&] { operand.set(newPadOp); });
      changed = true;
    }
    return success(changed);
  }
};

/// Optimizes operands that are padded but only contribute to an output tensor
/// that is undef padded to also be undef padded.
struct PropagateUnusedOutputPads : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();

    bool changed = false;
    // For every operand, go through all inits tensors and find which of its
    // dimensions have been undef padded. The corresponding dimensions of the
    // operands can then also be undef padded.
    // If true for every dim of the operand, for every output, the entire
    // operand can be undef padded.
    for (auto [operandMap, operand] :
         llvm::zip_equal(indexingMaps, linalgOp->getOpOperands())) {
      auto operandPadOp = operand.get().getDefiningOp<tensor::PadOp>();
      // Already undef padded.
      if (!operandPadOp || hasUndefPadding(operandPadOp))
        continue;

      bool canUndefPad = true;
      for (OpOperand &inits : linalgOp.getDpsInitsMutable()) {
        auto padOp = inits.get().getDefiningOp<tensor::PadOp>();
        if (!padOp || !hasUndefPadding(padOp)) {
          canUndefPad = false;
          break;
        }

        // Initially, all padded dims need non-undef padding.
        llvm::SmallBitVector needNonUndefPadding = operandPadOp.getPaddedDims();

        llvm::SmallBitVector initsPaddedDims = padOp.getPaddedDims();
        // Create a mapping from the init dims to the operand dims.
        // The init dims affine map must be invertible for this to be possible.
        AffineMap initsMap = linalgOp.getMatchingIndexingMap(&inits);
        AffineMap inverted = inverseAndBroadcastProjectedPermutation(initsMap);
        if (!inverted) {
          canUndefPad = false;
          break;
        }

        AffineMap initsToOperand = operandMap.compose(inverted);
        for (unsigned index : initsPaddedDims.set_bits())
          if (std::optional<unsigned> position =
                  initsToOperand.getResultPosition(
                      rewriter.getAffineDimExpr(index)))
            needNonUndefPadding[*position] = false;

        if (needNonUndefPadding.any()) {
          canUndefPad = false;
          break;
        }
      }
      if (!canUndefPad)
        continue;

      rewriter.modifyOpInPlace(linalgOp, [&, &operand = operand] {
        operand.set(cloneWithUndefPad(rewriter, operandPadOp));
      });
      changed = true;
    }
    return success(changed);
  }
};

} // namespace

namespace {

/// Expands a 'pad_undef(empty)' to a larger empty.
struct ExpandPaddedEmptyOp : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!hasUndefPadding(padOp))
      return failure();

    auto emptyOp = padOp.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return failure();

    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyResultShapes(rewriter, padOp, dims)))
      return failure();

    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        padOp, dims.front(), getElementTypeOrSelf(padOp.getType()));
    return success();
  }
};

/// Expands a 'pad_undef(extract_slice)' to a larger extract_slice if possible.
struct ExpandPaddedSlice : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Lower padding and dynamic padding values are currently unsupported.
    if (!hasUndefPadding(padOp) || !padOp.hasZeroLowPad() ||
        !padOp.getHigh().empty())
      return failure();

    auto extractSlice =
        padOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    // Only zero-offset supported right now to simplify the logic.
    if (!extractSlice || !extractSlice.hasZeroOffset())
      return failure();

    TypedValue<RankedTensorType> source = extractSlice.getSource();
    if (extractSlice.getType().getRank() != source.getType().getRank())
      return failure();

    // Check that the pad does not make the tensor larger than the original
    // source of the slice.
    ArrayRef<int64_t> high = padOp.getStaticHigh();
    for (auto [highPad, sourceShape, resultShape] : llvm::zip_equal(
             high, source.getType().getShape(), padOp.getType().getShape())) {
      if (highPad == 0)
        continue;

      if (sourceShape < resultShape)
        return failure();
    }

    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyResultShapes(rewriter, padOp, dims)))
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        padOp, source, /*offsets=*/
        SmallVector<OpFoldResult>(source.getType().getRank(),
                                  rewriter.getIndexAttr(0)),
        /*sizes=*/dims.front(),
        /*strides=*/
        SmallVector<OpFoldResult>(source.getType().getRank(),
                                  rewriter.getIndexAttr(1)));
    return success();
  }
};
} // namespace

void PadToTilingConfig::runOnOperation() {
  // TODO: This is seems like a horrible restriction and should be fixed.
  if (getOperation()
          ->walk([&](tensor::PadOp padOp) {
            padOp.emitError(
                "pass does not handle pre-existing padding operations");
            return WalkResult::interrupt();
          })
          .wasInterrupted())
    return signalPassFailure();

  SmallVector<linalg::LinalgOp> workList;
  getOperation()->walk([&](linalg::LinalgOp linalgOp) {
    if (!canZeroPad(linalgOp))
      return;
    workList.push_back(linalgOp);
  });

  std::optional<IntegerAttr> attr = getConfigIntegerAttr(
      IREE::HAL::ExecutableTargetAttr::lookup(getOperation()), "compute_cores");

  // Pad every linalg op to a multiple of all applied tile sizes.
  for (linalg::LinalgOp &linalgOp : workList)
    if (failed(padToTileSize(linalgOp, attr)))
      return signalPassFailure();

  // First perform just the conversion of zero-pads to undef-pads.
  // These must run separately from later patterns that may erase pad ops
  // entirely which discards information required by these patterns.
  {
    RewritePatternSet patterns(&getContext());
    patterns.insert<OptimizeElementwisePad, PropagateUnusedOutputPads,
                    OptimizeContractionOutputPad>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

  {
    RewritePatternSet patterns(&getContext());
    linalg::populateSwapExtractSliceWithFillPatterns(patterns);
    linalg::FillOp::getCanonicalizationPatterns(patterns, &getContext());
    getContext()
        .getLoadedDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    tensor::PadOp::getCanonicalizationPatterns(patterns, &getContext());
    patterns.insert<ExpandPaddedEmptyOp, ExpandPaddedSlice>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
}
