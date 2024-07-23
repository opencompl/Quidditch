#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_LOWERPIPELINEOPPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class LowerPipelineOp
    : public quidditch::Snitch::impl::LowerPipelineOpPassBase<LowerPipelineOp> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace quidditch::Snitch;

/// Duplicate resources required by a stage to support concurrent execution of
/// the stage and switches on it. Currently only supports `memref.alloca`.
static void makeResourceUsageExplicit(PipelineOp pipelineOp) {
  pipelineOp->walk([&](memref::AllocaOp allocation) {
    // TODO: Pipeline independent dynamic allocations would be fine.
    assert(allocation.getDynamicSizes().empty() &&
           "dynamic allocations not supported");
    Location loc = allocation->getLoc();

    // Clone right before the pipeline op as many times as we have stages.
    SmallVector<memref::AllocaOp> replacements;
    OpBuilder builder(pipelineOp);
    for ([[maybe_unused]] unsigned i : llvm::seq(pipelineOp.getStages().size()))
      builder.insert(replacements.emplace_back(allocation.clone()));

    // Find the region the allocation is contained in to find the IV we need to
    // use for the resource cycling.
    Region *parentRegion = allocation->getParentRegion();
    while (parentRegion->getParentOp() != pipelineOp)
      parentRegion = parentRegion->getParentRegion();
    Value iv = parentRegion->getArgument(0);

    // The cycle can be computed by normalizing the IV first and then
    // performing a mod operation using our number of stages.
    builder.setInsertionPoint(allocation);
    Value cycle = affine::makeComposedAffineApply(
        builder, loc,
        builder.getAffineDimExpr(0).floorDiv(builder.getAffineDimExpr(1)) %
            pipelineOp.getStages().size(),
        {iv, pipelineOp.getStep()});

    // Switch on the cycle number and return the corresponding resource.
    auto switchOp = builder.create<scf::IndexSwitchOp>(
        loc, allocation.getType(), cycle,
        llvm::to_vector(llvm::seq<int64_t>(pipelineOp.getStages().size() - 1)),
        pipelineOp.getStages().size() - 1);
    for (unsigned i : llvm::seq(pipelineOp.getRegions().size())) {
      if (i == pipelineOp.getRegions().size() - 1)
        builder.setInsertionPointToStart(
            &switchOp.getDefaultRegion().emplaceBlock());
      else
        builder.setInsertionPointToStart(
            &switchOp.getCaseRegions()[i].emplaceBlock());

      builder.create<scf::YieldOp>(loc, replacements[i].getMemref());
    }

    allocation->replaceAllUsesWith(switchOp);
    allocation->erase();
  });
}

/// Clones the given pipeline stage at the insertion point
/// of builder without the yield operation.
/// 'mapping' can be used to remap the block arguments of the block
/// and will additionally be used to contain the result mapping.
///
/// Returns the operands of the yield operation had it been cloned.
static SmallVector<Value> insertStage(OpBuilder &builder, Region &stage,
                                      IRMapping &mapping) {
  Block *block = &stage.front();
  for (Operation &op : block->without_terminator())
    builder.clone(op, mapping);

  SmallVector<Value> result;
  for (Value value : cast<PipelineYieldOp>(block->getTerminator()).getResults())
    result.push_back(mapping.lookupOrDefault(value));

  return result;
}

/// Returns the local induction value for a given stage.
/// We separate between two kinds of induction values:
/// * The global induction value
/// * The local induction value of a stage
/// The global induction value is the induction value of the first pipeline
/// stage only.
/// Since the second stage with IV = 0 executes at the same time as the first
/// stage with IV = 1, a negative offset has to be applied to the global IV
/// to map it to a given stage. This returned value is the local induction
/// value.
/// This function generates the code necessary that maps the global IV
/// 'inductionVar' to the local IV of 'stage' and returns the local IV.
/// 'step' is the step value of the pipeline op.
static Value getInductionVarForStage(OpBuilder &builder, Region &stage,
                                     Value inductionVar, Value step) {
  return affine::makeComposedAffineApply(
      builder, inductionVar.getLoc(),
      builder.getAffineDimExpr(0) -
          builder.getAffineDimExpr(1) *
              builder.getAffineConstantExpr(stage.getRegionNumber()),
      {inductionVar, step});
}

void LowerPipelineOp::runOnOperation() {
  getOperation()->walk([&](PipelineOp pipelineOp) {
    if (pipelineOp.hasTensorSemantics())
      return;

    makeResourceUsageExplicit(pipelineOp);

    IRRewriter builder(pipelineOp);
    Value currentIV = pipelineOp.getLowerBound();
    SmallVector<SmallVector<Value>> oldResultsOfStages(
        pipelineOp.getRegions().size() - 1);

    // Build on-ramp that executes all stages except the last.
    // As soon as we have started executing the last stage our pipeline is fully
    // utilized and we can switch to the 'scf.for'.
    for (unsigned i = 0; i < pipelineOp.getRegions().size() - 1; i++) {
      SmallVector<SmallVector<Value>> newResultsOfStages(
          pipelineOp.getRegions().size() - 1);

      // Go over one more stage every iteration as the result of the previous
      // stage gets ready.
      for (Region &stage : pipelineOp->getRegions().take_front(i + 1)) {
        IRMapping mapping;
        mapping.map(stage.getArguments().front(),
                    getInductionVarForStage(builder, stage, currentIV,
                                            pipelineOp.getStep()));
        if (stage != pipelineOp.getStages().front())
          for (auto [oldBlockArg, newBlockArg] :
               llvm::zip_equal(stage.getArguments().drop_front(),
                               oldResultsOfStages[stage.getRegionNumber() - 1]))
            mapping.map(oldBlockArg, newBlockArg);

        newResultsOfStages[stage.getRegionNumber()] =
            insertStage(builder, stage, mapping);
      }

      oldResultsOfStages = std::move(newResultsOfStages);
      currentIV = builder.create<arith::AddIOp>(pipelineOp->getLoc(), currentIV,
                                                pipelineOp.getStep());
    }

    SmallVector<Value> oldResultsOfStagesFlattened;
    // Maps from the index of a stage to the index of the first value in
    // 'oldResultsOfStagesFlattened' that is no longer part of this stages
    // result.
    SmallVector<size_t> endOfStageIndex{0};
    for (ArrayRef<Value> results : oldResultsOfStages) {
      llvm::append_range(oldResultsOfStagesFlattened, results);
      endOfStageIndex.push_back(endOfStageIndex.back() + results.size());
    }

    auto forOp = builder.create<scf::ForOp>(
        pipelineOp->getLoc(), currentIV, pipelineOp.getUpperBound(),
        pipelineOp.getStep(), oldResultsOfStagesFlattened);
    {
      OpBuilder::InsertionGuard guard{builder};
      builder.setInsertionPointToStart(forOp.getBody());

      SmallVector<Value> newResultsOfStagesFlattened;
      for (Region &stage : pipelineOp.getStages()) {
        Value iv = getInductionVarForStage(
            builder, stage, forOp.getInductionVar(), pipelineOp.getStep());

        IRMapping mapping;
        mapping.map(stage.getArguments().front(), iv);
        if (stage != pipelineOp.getStages().front()) {
          size_t begin = endOfStageIndex[stage.getRegionNumber() - 1];
          size_t end = endOfStageIndex[stage.getRegionNumber()];
          MutableArrayRef<BlockArgument> argument =
              forOp.getRegionIterArgs().slice(begin, end - begin);
          for (auto [oldBlockArg, newBlockArg] :
               llvm::zip_equal(stage.getArguments().drop_front(), argument))
            mapping.map(oldBlockArg, newBlockArg);
        }
        llvm::append_range(newResultsOfStagesFlattened,
                           insertStage(builder, stage, mapping));
      }
      builder.create<scf::YieldOp>(pipelineOp->getLoc(),
                                   newResultsOfStagesFlattened);
    }

    // Off-ramp.
    for (unsigned i = 0; i < endOfStageIndex.size() - 1; i++) {
      uint64_t begin = endOfStageIndex[i];
      uint64_t end = endOfStageIndex[i + 1];
      oldResultsOfStages[i] = forOp.getResults().slice(begin, end - begin);
    }

    // Compute the real inclusive UB of the IV as it'd be at the end of the
    // 'scf.for'.
    currentIV = affine::makeComposedAffineApply(
        builder, pipelineOp->getLoc(),
        builder.getAffineDimExpr(0).floorDiv(builder.getAffineDimExpr(1)) *
            builder.getAffineDimExpr(1),
        {forOp.getUpperBound(), forOp.getStep()});

    // Paste the stages until the last stage has done its last iteration.
    for (unsigned i = 1; i < pipelineOp.getRegions().size(); i++) {
      SmallVector<SmallVector<Value>> newResultsOfStages(
          pipelineOp.getRegions().size() - 1);

      // One fewer stage every iteration, first stage already completed.
      for (Region &stage : pipelineOp->getRegions().drop_front(i)) {
        Value iv = getInductionVarForStage(builder, stage, currentIV,
                                           pipelineOp.getStep());

        IRMapping mapping;
        mapping.map(stage.getArguments().front(), iv);
        for (auto [oldBlockArg, newBlockArg] :
             llvm::zip_equal(stage.getArguments().drop_front(),
                             oldResultsOfStages[stage.getRegionNumber() - 1]))
          mapping.map(oldBlockArg, newBlockArg);

        if (stage != pipelineOp.getStages().back())
          newResultsOfStages[stage.getRegionNumber()] =
              insertStage(builder, stage, mapping);
        else
          insertStage(builder, stage, mapping);
      }
      oldResultsOfStages = std::move(newResultsOfStages);
      currentIV = builder.create<arith::AddIOp>(pipelineOp->getLoc(), currentIV,
                                                pipelineOp.getStep());
    }

    pipelineOp->erase();
  });
}
