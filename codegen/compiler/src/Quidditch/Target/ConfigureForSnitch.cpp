#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_CONFIGUREFORSNITCHPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
class ConfigureForSnitch
    : public quidditch::impl::ConfigureForSnitchPassBase<ConfigureForSnitch> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

static LogicalResult setTranslationInfo(FunctionOpInterface funcOp) {
  return setTranslationInfo(
      funcOp,
      IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None, SymbolRefAttr()));
}

static LogicalResult setRootConfig(FunctionOpInterface funcOp,
                                   Operation *rootOp) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        // [0]: Always one in our matvec case.

        // [1]: How many rows we are processing. Should fit in L1.
        // Should be as high as possible for subgroup distribution.
        // Should be a multiple of 8 to be further distributed to compute cores.

        // [2]: Reduction dimension. How many columns are we
        // processing at once? Cannot be distributed but has a few effects:
        // * It allows us to make [1] larger by fitting more rows into L1.
        //   This therefore also gives us more parallelism compute core wise.
        // * It makes our workgroups larger, reducing dispatch overhead and
        //   memory bandwidth (by only needing to copy loop invariant memory
        //   once + needing to copy back the result fewer times). This could
        //   come at the cost of concurrency for distributing workgroups but is
        //   only applicable once on Occamy.
        SmallVector<int64_t> workgroupTiles(3, 0);
        SmallVector<int64_t> l1Tiles(3, 0);

        if (funcOp.getName() ==
            "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64") {
          l1Tiles[1] = 40;
          l1Tiles[2] = 0;
        }
        if (funcOp.getName() ==
            "main$async_dispatch_7_matmul_transpose_b_1x600x400_f64") {
          workgroupTiles[2] = 200;

          l1Tiles[0] = 0;
          l1Tiles[1] = 40;
          l1Tiles[2] = 0;
        }
        if (funcOp.getName() ==
            "main$async_dispatch_8_matmul_transpose_b_1x600x600_f64") {
          workgroupTiles[2] = 200;

          l1Tiles[0] = 0;
          l1Tiles[1] = 40;
        }
        if (funcOp.getName() ==
            "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64") {
          workgroupTiles[2] = 200;

          l1Tiles[0] = 0;
          l1Tiles[1] = 40;
          l1Tiles[2] = 0;
        }

        setLoweringConfig(rootOp,
                          quidditch::Snitch::LoweringConfigAttr::get(
                              rootOp->getContext(), workgroupTiles, l1Tiles));
        return success();
      })
      .Default(success());
}

void ConfigureForSnitch::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  if (getTranslationInfo(funcOp))
    return;

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp))
    return signalPassFailure();
  Operation *rootOperation = rootOp.value();
  if (!rootOperation)
    return;

  // Set the same translation info for all functions right now.
  // This should move into 'setRootConfig' if we gain different pass pipelines
  // for different kernels.
  if (failed(setTranslationInfo(funcOp)))
    return signalPassFailure();

  if (failed(setRootConfig(funcOp, rootOperation)))
    return signalPassFailure();

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
