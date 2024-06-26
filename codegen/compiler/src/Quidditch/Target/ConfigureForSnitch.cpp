#include "Passes.h"

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

static LogicalResult setRootConfig(FunctionOpInterface funcOp,
                                   Operation *rootOp) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        if (funcOp.getName() ==
            "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64") {
          SmallVector<int64_t> bounds(3, 0);
          // Future subgroup distribution.
          bounds[0] = 1;
          // How many rows we are processing (0 to 400). Should fit in L1.
          // Should be as high as possible for subgroup distribution.
          // (Could almost be 40).
          bounds[1] = 50;

          // Reduction dimension (0 to 161). How many columns are we processing
          // at once?
          // Cannot be distributed. As wide as possible for FPU utilization of a
          // single core.
          bounds[2] = 0;

          TileSizesListType tileSizes = {bounds};
          return setOpConfigAndEntryPointFnTranslation(
              funcOp, rootOp, tileSizes,
              IREE::Codegen::DispatchLoweringPassPipeline::None);
        }
        if (funcOp.getName() ==
            "main$async_dispatch_7_matmul_transpose_b_1x600x400_f64") {
          SmallVector<int64_t> bounds(3, 0);
          // Future subgroup distribution.
          bounds[0] = 1;
          // How many rows we are processing (0 to 600). Should fit in L1.
          // Should be as high as possible for subgroup distribution.
          // (Could almost be 40).
          bounds[1] = 25;

          // Reduction dimension (0 to 400). How many columns are we processing
          // at once?
          // Cannot be distributed. As wide as possible for FPU utilization of a
          // single core.
          bounds[2] = 0;

          TileSizesListType tileSizes = {bounds};
          return setOpConfigAndEntryPointFnTranslation(
              funcOp, rootOp, tileSizes,
              IREE::Codegen::DispatchLoweringPassPipeline::None);
        }
        if (funcOp.getName() ==
            "main$async_dispatch_8_matmul_transpose_b_1x600x600_f64") {
          SmallVector<int64_t> bounds(3, 0);
          // Future subgroup distribution.
          bounds[0] = 1;
          // How many rows we are processing (0 to 600). Should fit in L1.
          // Should be as high as possible for subgroup distribution.
          bounds[1] = 15;

          // Reduction dimension (0 to 600). How many columns are we processing
          // at once?
          // Cannot be distributed. As wide as possible for FPU utilization of a
          // single core.
          bounds[2] = 0;

          TileSizesListType tileSizes = {bounds};
          return setOpConfigAndEntryPointFnTranslation(
              funcOp, rootOp, tileSizes,
              IREE::Codegen::DispatchLoweringPassPipeline::None);
        }
        if (funcOp.getName() ==
            "main$async_dispatch_1_matmul_transpose_b_1x1200x400_f64") {
          SmallVector<int64_t> bounds(3, 0);
          // Future subgroup distribution.
          bounds[0] = 0;
          // How many rows we are processing (0 to 1200). Should fit in L1.
          // Should be as high as possible for subgroup distribution.
          bounds[1] = 25;
          // Reduction dimension (0 to 400). How many columns we are processing
          // at once?
          // Cannot be distributed. As wide as possible for FPU utilization of a
          // single core.
          bounds[2] = 0;

          TileSizesListType tileSizes = {bounds};
          return setOpConfigAndEntryPointFnTranslation(
              funcOp, rootOp, tileSizes,
              IREE::Codegen::DispatchLoweringPassPipeline::None);
        }

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

  if (failed(setRootConfig(funcOp, rootOperation)))
    return signalPassFailure();

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
