#include "QuidditchSnitchOps.h"

#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "QuidditchSnitchAttrs.h"

static mlir::ParseResult parseRISCVAssembly(mlir::OpAsmParser &opAsmParser,
                                            mlir::StringAttr &assembly);

static void printRISCVAssembly(mlir::OpAsmPrinter &opAsmPrinter,
                               mlir::Operation *, mlir::StringAttr assembly);

#define GET_OP_CLASSES
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.cpp.inc"

using namespace mlir;
using namespace mlir::bufferization;
using namespace quidditch::Snitch;

static ParseResult parseRISCVAssembly(OpAsmParser &opAsmParser,
                                      StringAttr &assembly) {
  std::string result;
  std::string line;
  while (succeeded(opAsmParser.parseOptionalString(&line))) {
    if (!result.empty())
      result += "\n";
    result += line;
  }
  assembly = StringAttr::get(opAsmParser.getContext(), result);
  return success();
}

static void printRISCVAssembly(OpAsmPrinter &opAsmPrinter, Operation *,
                               StringAttr assembly) {
  opAsmPrinter.increaseIndent();
  auto onExit = llvm::make_scope_exit([&] {
    opAsmPrinter.decreaseIndent();
    opAsmPrinter.printNewline();
  });

  SmallVector<StringRef> split;
  assembly.getValue().split(split, "\n");
  for (StringRef line : split) {
    opAsmPrinter.printNewline();
    opAsmPrinter.printString(line);
  }
}

//===----------------------------------------------------------------------===//
// TensorMicrokernelOp::RegionBranchOpInterface
//===----------------------------------------------------------------------===//

void TensorMicrokernelOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody());
    return;
  }
  regions.emplace_back(getResults());
}

void TensorMicrokernelOp::getRegionInvocationBounds(
    ArrayRef<Attribute>, SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.push_back({1, 1});
}

//===----------------------------------------------------------------------===//
// TensorMicrokernelOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

AliasingOpOperandList
TensorMicrokernelOp::getAliasingOpOperands(Value value,
                                           const AnalysisState &state) {
  return {{
      &getYieldOp()
           .getResultsMutable()[cast<OpResult>(value).getResultNumber()],
      BufferRelation::Equivalent,
      /*isDefinite=*/true,
  }};
}

FailureOr<BaseMemRefType>
TensorMicrokernelOp::getBufferType(Value value,
                                   const BufferizationOptions &options,
                                   SmallVector<Value> &invocationStack) {
  Value corresponding =
      getYieldOp().getResults()[cast<OpResult>(value).getResultNumber()];
  if (auto memRefType = dyn_cast<BaseMemRefType>(corresponding.getType()))
    return memRefType;

  return bufferization::getBufferType(corresponding, options, invocationStack);
}

LogicalResult
TensorMicrokernelOp::bufferize(RewriterBase &rewriter,
                               const BufferizationOptions &options) {
  SmallVector<Value> newYields;
  for (Value result : getYieldOp().getResults()) {
    if (!isa<TensorType>(result.getType())) {
      newYields.push_back(result);
      continue;
    }
    auto bufferType = bufferization::getBuffer(rewriter, result, options);
    if (failed(bufferType))
      return failure();
    newYields.push_back(*bufferType);
  }

  SetVector<Value> inputs;
  WalkResult walkResult = walk([&](Operation *operation) {
    for (Value value : operation->getOperands()) {
      if (isa<TensorType>(value.getType())) {
        FailureOr<Value> newInput = getBuffer(rewriter, value, options);
        if (failed(newInput))
          return WalkResult::interrupt();
        value = *newInput;
      }

      if (getBody().isAncestor(value.getParentRegion()))
        continue;
      inputs.insert(value);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  auto replacement =
      rewriter.create<MemRefMicrokernelOp>(getLoc(), inputs.getArrayRef());
  Block *newBlock = replacement.createEntryBlock();
  {
    OpBuilder::InsertionGuard guard{rewriter};
    rewriter.setInsertionPointToStart(newBlock);

    rewriter.mergeBlocks(&getBody().front(), newBlock);
    rewriter.eraseOp(newBlock->getTerminator());

    SmallVector<Value> vector = inputs.takeVector();
    rewriter.setInsertionPointToStart(newBlock);
    for (auto [oldV, newV] : llvm::zip(vector, newBlock->getArguments()))
      rewriter.replaceUsesWithIf(oldV, newV, [&](OpOperand &operand) {
        return replacement.getBody().isAncestor(
            operand.getOwner()->getParentRegion());
      });
  }

  replaceOpWithBufferizedValues(rewriter, *this, newYields);
  return success();
}

//===----------------------------------------------------------------------===//
// MicrokernelYieldOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

bool MicrokernelYieldOp::bufferizesToMemoryRead(
    OpOperand &, const bufferization::AnalysisState &) {
  return false;
}

bool MicrokernelYieldOp::bufferizesToMemoryWrite(OpOperand &,
                                                 const AnalysisState &) {
  return false;
}

AliasingValueList MicrokernelYieldOp::getAliasingValues(OpOperand &opOperand,
                                                        const AnalysisState &) {
  return {{getParentOp()->getResult(opOperand.getOperandNumber()),
           BufferRelation::Equivalent, /*isDefinite=*/true}};
}

bool MicrokernelYieldOp::mustBufferizeInPlace(OpOperand &,
                                              const AnalysisState &) {
  // Yield operands always bufferize inplace. Otherwise, an alloc + copy
  // may be generated inside the block. We should not return/yield allocations
  // when possible.
  return true;
}

LogicalResult
MicrokernelYieldOp::bufferize(RewriterBase &rewriter,
                              const BufferizationOptions &options) {
  SmallVector<Value> newResults;
  for (auto &&[index, value] : llvm::enumerate(getResults())) {
    if (!isa<TensorType>(value.getType())) {
      newResults.push_back(value);
      continue;
    }

    FailureOr<Value> maybeBuffer = getBuffer(rewriter, value, options);
    if (failed(maybeBuffer))
      return failure();

    newResults.push_back(*maybeBuffer);
  }
  replaceOpWithNewBufferizedOp<MicrokernelYieldOp>(rewriter, *this, newResults);
  return success();
}

//===----------------------------------------------------------------------===//
// SyncTensorOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

bool SyncTensorOp::bufferizesToMemoryRead(
    OpOperand &, const bufferization::AnalysisState &) {
  return false;
}

bool SyncTensorOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                           const AnalysisState &) {
  assert(opOperand == getInputMutable());
  // The op making the asynchronous result of the microkernel available is
  // effectively a write operation to the MemRef.
  return true;
}

AliasingValueList SyncTensorOp::getAliasingValues(OpOperand &opOperand,
                                                  const AnalysisState &) {
  assert(opOperand == getInputMutable());
  return {{getResult(), BufferRelation::Equivalent, /*isDefinite=*/true}};
}

bool SyncTensorOp::mustBufferizeInPlace(OpOperand &opOperand,
                                        const AnalysisState &) {
  assert(opOperand == getInputMutable());
  // The operation must bufferize in place as a copy inserted by the
  // bufferization framework would be inserted prior to the
  // `microkernel_fence` operation and not semantically equivalent.
  return true;
}

LogicalResult SyncTensorOp::bufferize(RewriterBase &rewriter,
                                      const BufferizationOptions &options) {
  FailureOr<Value> inputTensorBuffer = getBuffer(rewriter, getInput(), options);
  if (failed(inputTensorBuffer))
    return failure();

  rewriter.create<MicrokernelFenceOp>(getLoc());
  replaceOpWithBufferizedValues(rewriter, *this, *inputTensorBuffer);
  return success();
}

//===----------------------------------------------------------------------===//
// MemRefMicrokernelOp
//===----------------------------------------------------------------------===//

Block *MemRefMicrokernelOp::createEntryBlock() {
  assert(getBody().getBlocks().empty());
  Block &block = getBody().emplaceBlock();
  block.addArguments(getInputs().getTypes(),
                     SmallVector<Location>(getInputs().size(), getLoc()));
  return &block;
}

LogicalResult MemRefMicrokernelOp::verify() {
  if (getBody().getArgumentTypes() != getInputs().getTypes())
    return emitOpError("type of arguments and inputs must match");
  return success();
}

//===----------------------------------------------------------------------===//
// MemRefMicrokernelOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {

struct SinkConstantArguments : OpRewritePattern<MemRefMicrokernelOp> {
  using OpRewritePattern<MemRefMicrokernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefMicrokernelOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<std::pair<BlockArgument, Operation *>> constantOps;
    for (auto [input, arg] :
         llvm::zip(op.getInputs(), op.getBody().getArguments()))
      if (matchPattern(input, m_Constant()))
        constantOps.emplace_back(arg, input.getDefiningOp());

    if (constantOps.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      rewriter.setInsertionPointToStart(&op.getBody().front());
      for (auto [repl, constantOp] : constantOps) {
        Operation *clone = rewriter.clone(*constantOp);
        repl.replaceAllUsesWith(clone->getResult(0));
      }
    });
    return success();
  }
};

struct RemoveDeadArguments : OpRewritePattern<MemRefMicrokernelOp> {
  using OpRewritePattern<MemRefMicrokernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefMicrokernelOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<bool> deadArguments(op.getInputs().size());
    for (BlockArgument arg : op.getBody().getArguments())
      if (arg.use_empty())
        deadArguments[arg.getArgNumber()] = true;

    if (llvm::none_of(deadArguments, [](auto value) { return value; }))
      return failure();

    SmallVector<Value> newInputs;
    for (auto [index, value] : llvm::enumerate(op.getInputs()))
      if (!deadArguments[index])
        newInputs.push_back(value);

    auto replacement =
        rewriter.create<MemRefMicrokernelOp>(op.getLoc(), newInputs);
    rewriter.inlineRegionBefore(op.getBody(), replacement.getBody(),
                                replacement.getBody().end());
    rewriter.modifyOpInPlace(replacement, [&] {
      replacement.getBody().front().eraseArguments([&](BlockArgument argument) {
        return deadArguments[argument.getArgNumber()];
      });
    });

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct ReplaceIdenticalArguments : OpRewritePattern<MemRefMicrokernelOp> {
  using OpRewritePattern<MemRefMicrokernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefMicrokernelOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    llvm::SmallDenseMap<Value, BlockArgument> seenPreviously;
    for (auto [input, blockArg] :
         llvm::zip_equal(op.getInputs(), op.getBody().getArguments())) {
      auto [iter, inserted] = seenPreviously.insert({input, blockArg});
      if (inserted)
        continue;

      changed = true;
      rewriter.replaceAllUsesWith(blockArg, iter->second);
    }
    return success(changed);
  }
};
} // namespace

void MemRefMicrokernelOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<RemoveDeadArguments, SinkConstantArguments,
                 ReplaceIdenticalArguments>(context);
}

//===----------------------------------------------------------------------===//
// MemRefMicrokernelOp::ComputeCoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

void MemRefMicrokernelOp::replaceWithNoop(RewriterBase &rewriter) {
  rewriter.eraseOp(*this);
}

//===----------------------------------------------------------------------===//
// CallMicrokernelOp
//===----------------------------------------------------------------------===//

bool CallMicrokernelOp::supportsArgumentType(mlir::Type type) {
  auto memRef = dyn_cast<MemRefType>(type);
  if (!memRef)
    return true;
  if (isa<UnrankedMemRefType>(memRef))
    return false;

  int64_t offset = 0;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memRef, strides, offset)))
    return false;

  return llvm::none_of(strides, ShapedType::isDynamic);
}

LogicalResult CallMicrokernelOp::verify() {
  if (!llvm::all_of(getInputs().getTypes(), supportsArgumentType))
    return emitOpError("do not support functions with signature ")
           << FunctionType::get(getContext(), getInputs().getTypes(), {});
  return success();
}

//===----------------------------------------------------------------------===//
// MicrokernelFenceOp::ComputeCoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

void MicrokernelFenceOp::replaceWithNoop(RewriterBase &rewriter) {
  rewriter.eraseOp(*this);
}

//===----------------------------------------------------------------------===//
// ComputeCoreIndexOp::ComputeCoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

void ComputeCoreIndexOp::replaceWithNoop(RewriterBase &rewriter) {
  // Make the DMA core follow the control flow of the first compute core.
  // This whole pass runs under the assumption that any operation that is
  // run on either the DMA core or compute cores are in non-divergent
  // control flow. Making the DMA core follow any compute cores control
  // flow is therefore safe to do.
  // This is mainly required for barriers within a `scf.forall`.
  rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(*this, 0);
}

//===----------------------------------------------------------------------===//
// PipelineOp
//===----------------------------------------------------------------------===//

LogicalResult PipelineOp::verify() {
  if (getStages().empty())
    return emitOpError("must have at least one stage");

  for (Region &region : getStages())
    if (region.getArguments().empty() ||
        !isa<IndexType>(region.getArgument(0).getType()))
      return emitOpError(
          "first block argument of every stage must be of type 'index'");

  if (!hasTensorSemantics())
    if (!getResults().empty() || !getInitArgs().empty())
      return emitOpError("bufferized pipeline cannot have any iterables");

  return success();
}

BlockArgument PipelineOp::getTiedEntryIterArg(OpOperand &operand) {
  return getTiedEntryIterArg(getTiedResult(operand));
}

BlockArgument PipelineOp::getTiedEntryIterArg(OpResult opResult) {
  return getStages().front().getArgument(1 + opResult.getResultNumber());
}

OpResult PipelineOp::getTiedResult(OpOperand &operand) {
  return getResults()[operand.getOperandNumber() -
                      getInitArgs().getBeginOperandIndex()];
}

mlir::OpOperand &PipelineOp::getTiedYielded(OpResult result) {
  return getTiedYielded(getTiedEntryIterArg(result));
}

mlir::OpOperand &PipelineOp::getTiedYielded(BlockArgument argument) {
  Region *region = argument.getParentRegion();
  unsigned index = region->getRegionNumber();
  if (index == 0)
    index = getStages().size() - 1;
  else
    index--;

  return cast<PipelineYieldOp>(getStages()[index].back().getTerminator())
      .getResultsMutable()[argument.getArgNumber() - 1];
}

OpOperand &PipelineOp::getTiedInit(BlockArgument argument) {
  return getInitArgsMutable()[argument.getArgNumber() - 1];
}

namespace {
/// Removes block arguments with no uses (excluding the entry block).
struct DeadBlockArgRemoval : OpRewritePattern<PipelineOp> {
  using OpRewritePattern<PipelineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PipelineOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (Region &region : op.getStages().drop_front())
      for (BlockArgument arg :
           llvm::reverse(region.getArguments().drop_front()))
        if (arg.use_empty()) {
          changed = true;
          OpOperand &operand = op.getTiedYielded(arg);
          rewriter.modifyOpInPlace(operand.getOwner(), [&] {
            cast<PipelineYieldOp>(operand.getOwner())
                .getResultsMutable()
                .erase(operand.getOperandNumber());
          });
          rewriter.modifyOpInPlace(
              op, [&] { region.eraseArgument(arg.getArgNumber()); });
        }

    return success(changed);
  }
};

/// Replace uses of arguments if they are loop invariant.
struct InvariantArgumentReplacement : OpRewritePattern<PipelineOp> {
  using OpRewritePattern<PipelineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PipelineOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (Region &region : op.getStages())
      for (BlockArgument arg :
           llvm::reverse(region.getArguments().drop_front())) {
        Value yielded = op.getTiedYielded(arg).get();
        if (!yielded.getParentRegion()->isAncestor(op->getParentRegion()))
          continue;

        if (&region == &op.getStages().front()) {
          OpOperand &init = op.getTiedInit(arg);
          if (yielded != init.get())
            continue;

          OpResult result = op.getTiedResult(init);
          if (arg.use_empty() && result.use_empty())
            continue;

          rewriter.replaceAllUsesWith(result, yielded);
          changed = true;
        }

        if (arg.use_empty())
          continue;

        rewriter.replaceAllUsesWith(arg, yielded);
        changed = true;
      }

    return success(changed);
  }
};

/// Remove unused results.
struct DeadResultRemoval : OpRewritePattern<PipelineOp> {

  using OpRewritePattern<PipelineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PipelineOp op,
                                PatternRewriter &rewriter) const override {
    llvm::BitVector alive(op.getNumResults(), true);
    for (OpResult result : llvm::reverse(op.getResults())) {
      if (!result.use_empty())
        continue;

      BlockArgument argument = op.getTiedEntryIterArg(result);
      if (!argument.use_empty())
        continue;

      alive[result.getResultNumber()] = false;
      OpOperand &operand = op.getTiedYielded(argument);
      OpOperand &init = op.getTiedInit(argument);
      rewriter.modifyOpInPlace(operand.getOwner(), [&] {
        cast<PipelineYieldOp>(operand.getOwner())
            .getResultsMutable()
            .erase(operand.getOperandNumber());
      });
      rewriter.modifyOpInPlace(op, [&] {
        argument.getOwner()->eraseArgument(argument.getArgNumber());
        op.getInitArgsMutable().erase(init.getOperandNumber() -
                                      op.getInitArgs().getBeginOperandIndex());
      });
    }

    if (alive.all())
      return failure();

    auto newPipeline = rewriter.create<PipelineOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        op.getInitArgs(), op.getStages().size());
    for (auto [oldRegion, newRegion] :
         llvm::zip_equal(op.getStages(), newPipeline.getStages()))
      newRegion.takeBody(oldRegion);

    auto current = newPipeline.getResults().begin();
    for (OpResult result : op.getResults())
      if (alive[result.getResultNumber()]) {
        rewriter.replaceAllUsesWith(result, *current);
        current++;
      }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Move operations from one stage into a later stage if profitable.
struct MoveSideEffectFreeComputation : OpRewritePattern<PipelineOp> {
  using OpRewritePattern<PipelineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PipelineOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (Region &region : op.getStages().drop_front()) {
      rewriter.setInsertionPointToStart(&region.front());
      for (BlockArgument arg : region.getArguments().drop_front()) {
        if (arg.use_empty())
          continue;

        // If an implementation is side effect free, only used in a later stage,
        // and only depends on the index as input operand, move it to the later
        // stage.
        // TODO: As is often the case with code motion, this might be very
        //       opinionated.
        OpOperand &yieldOperand = op.getTiedYielded(arg);
        Operation *yield = yieldOperand.getOwner();
        Value yielded = yieldOperand.get();
        Operation *def = yielded.getDefiningOp();
        if (!def || !isPure(def) ||
            def->getParentRegion() != yield->getParentRegion() ||
            !llvm::all_of(def->getUsers(),
                          [&](Operation *user) { return user == yield; }))
          continue;

        if (!llvm::all_of(def->getOperands(), [&](Value value) {
              return value == yielded.getParentRegion()->getArgument(0) ||
                     value.getParentRegion()->isAncestor(op->getParentRegion());
            }))
          continue;

        IRMapping mapping;
        mapping.map(yielded.getParentRegion()->getArgument(0),
                    arg.getParentRegion()->getArgument(0));
        rewriter.clone(*def, mapping);
        rewriter.replaceAllUsesWith(arg, mapping.lookupOrDefault(yielded));
        changed = true;
      }
    }

    return success(changed);
  }
};
} // namespace

void PipelineOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<DeadBlockArgRemoval, InvariantArgumentReplacement,
                 DeadResultRemoval, MoveSideEffectFreeComputation>(context);
}

//===----------------------------------------------------------------------===//
// PipelineOp::InferTypeOpAdaptor
//===----------------------------------------------------------------------===//

LogicalResult
PipelineOp::inferReturnTypes(MLIRContext *context,
                             std::optional<Location> location, Adaptor adaptor,
                             SmallVectorImpl<Type> &inferredReturnTypes) {
  llvm::append_range(inferredReturnTypes, adaptor.getInitArgs().getTypes());
  return success();
}

//===----------------------------------------------------------------------===//
// PipelineOp::RegionBranchOpInterface
//===----------------------------------------------------------------------===//

void PipelineOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  auto addRegion = [&](Region &region) {
    regions.emplace_back(&region, region.getArguments().drop_front());
  };

  if (point.isParent()) {
    addRegion(getStages().front());
    regions.emplace_back(getResults());
    return;
  }

  unsigned index = point.getRegionOrNull()->getRegionNumber();
  if (index + 1 == getStages().size()) {
    addRegion(getStages().front());
    regions.emplace_back(getResults());
    return;
  }

  addRegion(getStages()[index + 1]);
}

OperandRange PipelineOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  return getInitArgs();
}

//===----------------------------------------------------------------------===//
// PipelineOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

bool PipelineOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  assert(llvm::is_contained(getInitArgsMutable(), opOperand));

  return state.isValueRead(getTiedEntryIterArg(opOperand));
}

bool PipelineOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  assert(llvm::is_contained(getInitArgsMutable(), opOperand));

  // TODO: This is cargo-culted from 'scf.for'. The impact is unclear.
  return true;
}

AliasingValueList
PipelineOp::getAliasingValues(OpOperand &opOperand,
                              const bufferization::AnalysisState &state) {
  assert(llvm::is_contained(getInitArgsMutable(), opOperand));

  return {{getTiedResult(opOperand), BufferRelation::Equivalent}};
}

FailureOr<BaseMemRefType>
PipelineOp::getBufferType(Value value,
                          const bufferization::BufferizationOptions &options,
                          SmallVector<Value> &invocationStack) {
  if (auto opResult = dyn_cast<OpResult>(value))
    return bufferization::getBufferType(getTiedEntryIterArg(opResult), options,
                                        invocationStack);

  auto argument = cast<BlockArgument>(value);
  if (argument.getParentRegion()->getRegionNumber() != 0) {
    Value yielded = getTiedYielded(argument).get();
    if (auto memRef = dyn_cast<BaseMemRefType>(yielded.getType()))
      return memRef;

    return bufferization::getBufferType(yielded, options, invocationStack);
  }

  FailureOr<BaseMemRefType> initType = bufferization::getBufferType(
      getTiedInit(argument).get(), options, invocationStack);
  if (failed(initType))
    return failure();

  if (llvm::is_contained(invocationStack, argument))
    return initType;

  Value yieldedValue = getTiedYielded(argument).get();
  BaseMemRefType yieldedType = dyn_cast<BaseMemRefType>(yieldedValue.getType());
  if (!yieldedType) {
    FailureOr<BaseMemRefType> maybeYieldedType =
        bufferization::getBufferType(yieldedValue, options, invocationStack);
    if (failed(maybeYieldedType))
      return failure();

    yieldedType = *maybeYieldedType;
  }

  if (yieldedType == initType)
    return yieldedType;

  return getMemRefTypeWithFullyDynamicLayout(
      cast<TensorType>(argument.getType()), yieldedType.getMemorySpace());
}

static FailureOr<SmallVector<Value>>
getBuffers(ValueRange values, RewriterBase &rewriter,
           const bufferization::BufferizationOptions &options) {
  SmallVector<Value> result;
  for (Value old : values) {
    if (!isa<TensorType>(old.getType())) {
      result.push_back(old);
      continue;
    }

    FailureOr<Value> maybeBuffer =
        bufferization::getBuffer(rewriter, old, options);
    if (failed(maybeBuffer))
      return failure();

    // TODO: MemRef types may not exactly match and require a cast.
    result.push_back(*maybeBuffer);
  }
  return std::move(result);
}

LogicalResult
PipelineOp::bufferize(RewriterBase &rewriter,
                      const bufferization::BufferizationOptions &options) {
  FailureOr<SmallVector<Value>> inits =
      getBuffers(getInitArgs(), rewriter, options);
  if (failed(inits))
    return failure();

  // TODO: Cast inits to buffer type of block arg if necessary.

  SmallVector<SmallVector<Value>> perRegionReplacements;
  auto newPipelineOp = rewriter.create<PipelineOp>(
      getLoc(), getLowerBound(), getUpperBound(), getStep(),
      /*inits=*/ValueRange(), getStages().size());
  for (auto [oldRegion, newRegion] :
       llvm::zip_equal(getStages(), newPipelineOp.getStages())) {
    Block &newBlock = newRegion.emplaceBlock();
    rewriter.setInsertionPointToStart(&newBlock);

    SmallVector<Value> replacements;
    for (BlockArgument arg : oldRegion.getArguments()) {
      if (!isa<TensorType>(arg.getType())) {
        replacements.push_back(
            newBlock.addArgument(arg.getType(), arg.getLoc()));
        continue;
      }

      // Entry regions are special: They can only have tensor block arguments
      // that are filled from inits and the last stage. After bufferization,
      // they must disappear entirely.
      if (oldRegion.getRegionNumber() == 0) {
        Value init = (*inits)[arg.getArgNumber() - 1];
        replacements.push_back(
            rewriter.create<bufferization::ToTensorOp>(init.getLoc(), init));
        continue;
      }

      FailureOr<BaseMemRefType> memRef =
          bufferization::getBufferType(arg, options);
      if (failed(memRef))
        return failure();

      BlockArgument newArg = newBlock.addArgument(*memRef, arg.getLoc());
      replacements.push_back(
          rewriter.create<bufferization::ToTensorOp>(newArg.getLoc(), newArg));
    }
    perRegionReplacements.push_back(std::move(replacements));
  }

  for (auto [oldRegion, newRegion, replacements] : llvm::zip_equal(
           getStages(), newPipelineOp.getStages(), perRegionReplacements))
    rewriter.mergeBlocks(&oldRegion.front(), &newRegion.front(), replacements);

  // Last stage cannot yield anything anymore once bufferized.
  cast<PipelineYieldOp>(newPipelineOp.getStages().back().back().getTerminator())
      .getResultsMutable()
      .clear();

  replaceOpWithBufferizedValues(rewriter, *this, *inits);
  return success();
}

bool PipelineOp::isWritable(Value value,
                            const bufferization::AnalysisState &state) {
  return true;
}

bool PipelineOp::mustBufferizeInPlace(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return true;
}

AliasingOpOperandList
PipelineOp::getAliasingOpOperands(Value value,
                                  const bufferization::AnalysisState &state) {
  if (auto result = dyn_cast<OpResult>(value))
    return {{&getTiedInit(getTiedEntryIterArg(result)),
             BufferRelation::Equivalent}};
  return {};
}

//===----------------------------------------------------------------------===//
// PipelineOp::LoopLikeInterface
//===----------------------------------------------------------------------===//

SmallVector<Region *> PipelineOp::getLoopRegions() {
  return llvm::map_to_vector(getStages(),
                             [](Region &region) { return &region; });
}

void PipelineOp::moveOutOfLoop(Operation *op) {
  Block *block = op->getBlock();
  auto yieldOp = cast<PipelineYieldOp>(block->getTerminator());
  if (op->getParentRegion() != getStages().back())
    for (OpOperand &operand : yieldOp.getResultsMutable())
      if (operand.get().getDefiningOp() == op)
        yieldOp.getTiedBlockArgument(operand).replaceAllUsesWith(operand.get());

  op->moveBefore(*this);
}

//===----------------------------------------------------------------------===//
// PipelineYieldOp
//===----------------------------------------------------------------------===//

BlockArgument PipelineYieldOp::getTiedBlockArgument(OpOperand &operand) {
  Region *region = (*this)->getParentRegion();
  unsigned index = region->getRegionNumber();
  unsigned next = index + 1;
  if (next == getParentOp().getStages().size())
    next = 0;

  return getParentOp().getStages()[next].getArgument(
      1 + (operand.getOperandNumber() - getResults().getBeginOperandIndex()));
}

//===----------------------------------------------------------------------===//
// PipelineYieldOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

bool PipelineYieldOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return true;
}

bool PipelineYieldOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return false;
}

bool PipelineYieldOp::mustBufferizeInPlace(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return true;
}

AliasingValueList
PipelineYieldOp::getAliasingValues(OpOperand &opOperand,
                                   const bufferization::AnalysisState &state) {
  return {};
}

LogicalResult
PipelineYieldOp::bufferize(RewriterBase &rewriter,
                           const bufferization::BufferizationOptions &options) {
  FailureOr<SmallVector<Value>> buffers =
      getBuffers(getResults(), rewriter, options);
  if (failed(buffers))
    return failure();

  replaceOpWithNewBufferizedOp<PipelineYieldOp>(rewriter, *this, *buffers);
  return success();
}
