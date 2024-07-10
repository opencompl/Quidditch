#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_TENSORTILEPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

namespace {
class TensorTile : public quidditch::impl::TensorTilePassBase<TensorTile> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace quidditch;
using namespace mlir;
using namespace mlir::iree_compiler;

// Adapted from GPUApplyTilingLevel.cpp.

/// This collects the set of operations to tile + fuse starting from the given
/// root |op| and walking up to its producers. Stops at operations given by
/// |exclude| which are expected to receive their own independent tiling for the
/// given level.
static llvm::SmallDenseSet<Operation *>
collectTiledAndFusedOps(Operation *op,
                        const llvm::SmallDenseSet<TilingInterface> &exclude) {
  SmallVector<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> producers;
  worklist.push_back(op);
  producers.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      auto producer = operand.get().getDefiningOp<TilingInterface>();
      if (!producer || producers.contains(producer) ||
          exclude.contains(producer))
        continue;
      worklist.push_back(producer);
      producers.insert(producer);
    }
  }
  return producers;
}

static SmallVector<OpFoldResult>
threadTileSizeComputation(OpBuilder &builder, Operation *operation) {
  SmallVector<OpFoldResult> result;

  std::optional<IntegerAttr> attr = getConfigIntegerAttr(
      IREE::HAL::ExecutableTargetAttr::lookup(operation), "compute_cores");
  if (!attr)
    return result;

  auto computeOp = cast<TilingInterface>(operation);
  std::optional<unsigned> largestParallelDim;
  std::optional<int64_t> largestParallelSize;
  for (auto [iterType, range] :
       llvm::zip_equal(computeOp.getLoopIteratorTypes(),
                       computeOp.getIterationDomain(builder))) {
    // Not doing reduction tiling.
    if (iterType == utils::IteratorType::reduction) {
      result.push_back(builder.getIndexAttr(0));
      continue;
    }

    // Not tileable.
    if (getConstantIntValue(range.size) == 1) {
      result.push_back(builder.getIndexAttr(0));
      continue;
    }

    // Not tiling dynamic dimensions right now.
    std::optional<int64_t> size = getConstantIntValue(range.size);
    if (!size) {
      result.push_back(builder.getIndexAttr(0));
      continue;
    }

    if (!largestParallelSize || size > largestParallelSize) {
      largestParallelDim = result.size();
      largestParallelSize = size;
    }

    // Placeholder for later.
    result.push_back(builder.getIndexAttr(0));
  }

  if (largestParallelDim) {
    assert(largestParallelSize);
    result[*largestParallelDim] = builder.getIndexAttr(llvm::divideCeil(
        *largestParallelSize, attr->getValue().getSExtValue()));
  }
  return result;
}

/// Apply a tile and fuse transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult
applyTileAndFuseToEachRoot(RewriterBase &rewriter,
                           llvm::SmallDenseSet<TilingInterface> &payloadOps,
                           TilingLevel tilingLevel) {
  for (TilingInterface tilingInterfaceOp : payloadOps) {

    DominanceInfo dominanceInfo(tilingInterfaceOp);

    llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
        collectTiledAndFusedOps(tilingInterfaceOp, payloadOps);
    DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    rewriter.setInsertionPoint(tilingInterfaceOp);

    scf::SCFTilingOptions tilingOptions;
    switch (tilingLevel) {
    case TilingLevel::Thread:
      tilingOptions.setTileSizeComputationFunction(threadTileSizeComputation);
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      break;
    case TilingLevel::Reduction:
      tilingOptions.setTileSizeComputationFunction(
          [&](OpBuilder &builder, auto &&...) {
            SmallVector<OpFoldResult> result;

            // Reapply the workgroup tiling config where the reduction dimension
            // was not applied.
            SmallVector<int64_t> workgroupSize =
                getLoweringConfig(tilingInterfaceOp).getWorkgroupTileSizes();
            for (int64_t value : workgroupSize)
              result.push_back(builder.getIndexAttr(value));

            size_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
            while (result.size() < numLoops)
              result.push_back(builder.getIndexAttr(0));

            return result;
          });
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      break;
    case TilingLevel::L1:
      tilingOptions.setTileSizeComputationFunction(
          [&](OpBuilder &builder, auto &&...) {
            SmallVector<OpFoldResult> result;

            SmallVector<int64_t> l1Tiles(
                getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(
                    tilingInterfaceOp)
                    .getL1Tiles());
            for (int64_t value : l1Tiles)
              result.push_back(builder.getIndexAttr(value));

            size_t numLoops = tilingInterfaceOp.getLoopIteratorTypes().size();
            while (result.size() < numLoops)
              result.push_back(builder.getIndexAttr(0));

            return result;
          });
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      break;
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand) {
          Operation *owner = originalProducer.getOwner();
          bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
          bool shouldFuse = false;
          if (auto tilingOwner = dyn_cast<TilingInterface>(owner)) {
            shouldFuse = !payloadOps.contains(tilingOwner);
          }
          // Do not fuse destination operands.
          shouldFuse &= !isDestinationOperand;
          return std::make_tuple(shouldFuse, yieldProducerReplacement);
        };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                  tileAndFuseOptions);
    if (failed(tiledResults)) {
      return failure();
    }

    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{tilingInterfaceOp};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(res, replacement, [&](OpOperand &use) {
            Operation *user = use.getOwner();
            return dominanceInfo.properlyDominates(replacementOp, user);
          });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }
  }
  return success();
}

void TensorTile::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  llvm::SmallDenseSet<TilingInterface> targetOps;
  switch (tilingLevel) {
  case TilingLevel::Thread:
    funcOp->walk([&](TilingInterface target) { targetOps.insert(target); });
    break;
  case TilingLevel::L1:
  case TilingLevel::Reduction:
    funcOp->walk([&](TilingInterface target) {
      if (auto loweringConfig =
              getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(target))
        targetOps.insert(target);
    });
    break;
  }

  IRRewriter rewriter(funcOp);
  if (failed(applyTileAndFuseToEachRoot(rewriter, targetOps, tilingLevel)))
    return signalPassFailure();

  MLIRContext *context = &getContext();

  // Apply cleanup patterns.
  {
    RewritePatternSet patterns(context);
    // Merge consecutive insert/extract slice ops to simplify later loop
    // hoisting patterns.
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "tiling cleanup failed\n";
      return signalPassFailure();
    }
  }
}
