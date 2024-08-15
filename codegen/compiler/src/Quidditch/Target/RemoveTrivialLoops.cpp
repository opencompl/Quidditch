#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

// Adapted from IREE's upstream RemoveTrivialLoops pass.
// TODO: Could make an interface (or reuse an interface from upstream?) to make
// it more generic and useable for us downstream.

namespace quidditch {
#define GEN_PASS_DEF_REMOVETRIVIALLOOPSPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace iree_compiler;

/// If the value is a threadID return the range [0, workgroupSize-1].
/// If the number of workgroup is known also return the range of workgroupId ad
/// workgroupCount.
static std::optional<std::pair<AffineExpr, AffineExpr>>
getWorkgroupRange(Value processorValue,
                  std::optional<IntegerAttr> numComputeCores,
                  ArrayRef<int64_t> workgroupCount) {
  if (auto idOp = processorValue
                      .getDefiningOp<quidditch::Snitch::ComputeCoreIndexOp>()) {
    if (!numComputeCores)
      return std::nullopt;

    OpBuilder b(processorValue.getContext());
    AffineExpr zero = b.getAffineConstantExpr(0);
    AffineExpr ubExpr =
        b.getAffineConstantExpr(numComputeCores->getValue().getZExtValue());
    return std::make_pair(zero, ubExpr - 1);
  }

  if (workgroupCount.empty() ||
      llvm::any_of(workgroupCount, ShapedType::isDynamic))
    return std::nullopt;

  if (auto idOp =
          processorValue.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>()) {
    OpBuilder builder(processorValue.getContext());

    // Can't infer the range when workroupCount is unknown.
    unsigned index = idOp.getDimension().getZExtValue();
    if (!workgroupCount[index])
      return std::nullopt;

    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  if (auto dimOp = processorValue
                       .getDefiningOp<IREE::HAL::InterfaceWorkgroupCountOp>()) {
    OpBuilder builder(processorValue.getContext());

    // Can't infer the range when workroupCount is unknown.
    unsigned index = dimOp.getDimension().getZExtValue();
    if (!workgroupCount[index])
      return std::nullopt;

    AffineExpr bound = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(bound, bound);
  }
  return std::nullopt;
}

static LogicalResult removeOneTripTiledLoops(FunctionOpInterface funcOp,
                                             ArrayRef<int64_t> numWorkgroups) {
  std::optional<IntegerAttr> attr = getConfigIntegerAttr(
      IREE::HAL::ExecutableTargetAttr::lookup(funcOp), "compute_cores");

  auto getWorkgroupRangeFn = [&](Value processorValue, SmallVectorImpl<Value> &,
                                 SmallVectorImpl<Value> &) {
    return getWorkgroupRange(processorValue, attr, numWorkgroups);
  };
  RewritePatternSet patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns, getWorkgroupRangeFn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

namespace {

class RemoveTrivialLoops final
    : public quidditch::impl::RemoveTrivialLoopsPassBase<RemoveTrivialLoops> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);
    if (failed(removeOneTripTiledLoops(funcOp, numWorkgroups))) {
      return signalPassFailure();
    }
  }
};
} // namespace
