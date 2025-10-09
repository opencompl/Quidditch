#include "Passes.h"

#include <sys/wait.h>
#include <unistd.h>
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "TilingScheme.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace quidditch {
#define GEN_PASS_DEF_CONFIGURETILES
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

class ConfigureTiles
    : public quidditch::impl::ConfigureTilesBase<ConfigureTiles> {
public:
  using Base::Base;
  ConfigureTiles(const quidditch::ConfigureTilesOptions &options) {
    this->importTiles = options.importTiles;
    this->tbl = (quidditch::TileInfoTbl *)options.tablePointer;
  }

protected:
  void runOnOperation() override;

private:
  std::string importTiles = "";
  quidditch::TileInfoTbl *tbl;
};
} // namespace

static LogicalResult setTranslationInfo(FunctionOpInterface funcOp) {
  return setTranslationInfo(
      funcOp,
      IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None, SymbolRefAttr()));
}

static LogicalResult
setRootConfig(FunctionOpInterface funcOp, Operation *rootOp,
              quidditch::TileInfoTbl *tbl) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::LinalgOp op) {
        // Assume tiling scheme passed in with --iree-quidditch-import-tiles
        SmallVector<int64_t> workgroupTiles(3, 0);
        SmallVector<int64_t> l1Tiles(3, 0);
        SmallVector<int64_t> l1Interchange = {2, 0, 1};
        bool dualBuffer = false;
        // if table of tiling schemes is invalid, throw an error
        if (tbl == 0) {
          funcOp.emitWarning() << "\nConfigureTiles: Table pointer is zero!!";
          return failure();
        }
        
        // Look up the tile size, interchange, and double buffering settings
        // from table
        auto search = tbl->find(funcOp.getName().str());
        if (search == tbl->end()) {
          funcOp.emitWarning()
              << "\nConfigureTiles: Root operation of this dispatch "
                 "is a missing tiling scheme";
          return failure();
        }
        quidditch::TilingScheme &ts = search->second;
        if (!ts.getTiles_flat(l1Tiles)) {
          funcOp.emitWarning() << "\nConfigureTiles: Found tiling scheme, but "
                                  "couldn't get l1 tile list";
          return failure();
        }
        if (!ts.getOrder_flat(l1Interchange)) {
          funcOp.emitWarning() << "\nConfigureTiles: Found tiling scheme, but "
                                  "couldn't get l1 interchange";
          return failure();
        }
        dualBuffer = ts.getDualBuffer();
        // set lowering config according to info in table
        setLoweringConfig(rootOp, quidditch::Snitch::LoweringConfigAttr::get(
                                      rootOp->getContext(), workgroupTiles,
                                      l1Tiles, l1Interchange, dualBuffer));
        return success();
      })
      .Default(success());
}

void ConfigureTiles::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  if (getTranslationInfo(funcOp))
    return;

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) {
    return signalPassFailure();
  }
  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    return;
  }

  // Set the same translation info for all functions right now.
  // This should move into 'setRootConfig' if we gain different pass pipelines
  // for different kernels.
  if (failed(setTranslationInfo(funcOp))) {
    return signalPassFailure();
  }

  // Annotate root linalg ops with tile sizes
  auto loweringConfig =
      getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(rootOperation);
  if (!loweringConfig) {
    if (failed(setRootConfig(funcOp, rootOperation, tbl))) {
      funcOp.emitWarning()
          << "\nConfigureTiles: set root config failed\n";
      return signalPassFailure();
    }
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    funcOp.emitWarning() << "\nConfigureTiles: apply patterns and "
                            "fold greedily failed\n";
    signalPassFailure();
  }
}
