#include "QuidditchSnitchOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "QuidditchSnitchAttrs.h"

#define GET_OP_CLASSES
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.cpp.inc"

using namespace mlir;
using namespace mlir::bufferization;
using namespace quidditch::Snitch;

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

  auto replacement = rewriter.create<MemRefMicrokernelOp>(
      getLoc(), llvm::map_to_vector(newYields, std::mem_fn(&Value::getType)),
      inputs.getArrayRef());
  Block *newBlock = replacement.createEntryBlock();
  {
    OpBuilder::InsertionGuard guard{rewriter};
    rewriter.setInsertionPointToStart(newBlock);

    rewriter.mergeBlocks(&getBody().front(), newBlock);
    rewriter.modifyOpInPlace(replacement.getYieldOp(), [&] {
      replacement.getYieldOp().getResultsMutable().assign(newYields);
    });

    SmallVector<Value> vector = inputs.takeVector();
    rewriter.setInsertionPointToStart(newBlock);
    for (auto [oldV, newV] : llvm::zip(vector, newBlock->getArguments()))
      rewriter.replaceUsesWithIf(oldV, newV, [&](OpOperand &operand) {
        return replacement.getBody().isAncestor(
            operand.getOwner()->getParentRegion());
      });
  }

  replaceOpWithBufferizedValues(rewriter, *this, replacement.getResults());
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
  return {{(*this)->getParentOp()->getResult(opOperand.getOperandNumber()),
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
// MemRefMicrokernelOp
//===----------------------------------------------------------------------===//

MicrokernelYieldOp MemRefMicrokernelOp::getYieldOp() {
  return llvm::cast<MicrokernelYieldOp>(getBody().back().getTerminator());
}

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
struct RemoveDeadResults : OpRewritePattern<MemRefMicrokernelOp> {
  using OpRewritePattern<MemRefMicrokernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefMicrokernelOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<bool> deadResults(op.getNumResults());
    for (OpResult result : op.getResults())
      if (result.use_empty())
        deadResults[result.getResultNumber()] = true;

    if (llvm::none_of(deadResults, [](auto value) { return value; }))
      return failure();

    SmallVector<Type> newResults;
    for (auto [index, type] : llvm::enumerate(op.getResults().getTypes()))
      if (!deadResults[index])
        newResults.push_back(type);

    auto replacement = rewriter.create<MemRefMicrokernelOp>(
        op.getLoc(), newResults, op.getInputs());
    rewriter.inlineRegionBefore(op.getBody(), replacement.getBody(),
                                replacement.getBody().end());
    MicrokernelYieldOp yieldOp = replacement.getYieldOp();
    for (auto index :
         llvm::reverse(llvm::seq<std::size_t>(0, deadResults.size())))
      if (deadResults[index])
        rewriter.modifyOpInPlace(yieldOp, [&, index = index] {
          yieldOp.getResultsMutable().erase(index);
        });

    unsigned nextAliveIndex = 0;
    for (auto [index, dead] : llvm::enumerate(deadResults)) {
      if (dead)
        continue;
      rewriter.replaceAllUsesWith(op.getResult(index),
                                  replacement.getResult(nextAliveIndex));
      nextAliveIndex++;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

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

struct ReplaceInvariantResults : OpRewritePattern<MemRefMicrokernelOp> {
  using OpRewritePattern<MemRefMicrokernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefMicrokernelOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (auto [result, yielded] :
         llvm::zip_equal(op.getResults(), op.getYieldOp().getResults())) {
      auto arg = dyn_cast<BlockArgument>(yielded);
      if (!arg || !arg.getParentBlock()->isEntryBlock())
        continue;
      changed = true;
      rewriter.replaceAllUsesWith(result, op.getInputs()[arg.getArgNumber()]);
    }
    return success(changed);
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

    auto replacement = rewriter.create<MemRefMicrokernelOp>(
        op.getLoc(), op.getResults().getTypes(), newInputs);
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
  results.insert<RemoveDeadResults, RemoveDeadArguments, SinkConstantArguments,
                 ReplaceInvariantResults, ReplaceIdenticalArguments>(context);
}

//===----------------------------------------------------------------------===//
// MemRefMicrokernelOp::RegionBranchOpInterface
//===----------------------------------------------------------------------===//

void MemRefMicrokernelOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody(), getBody().getArguments());
    return;
  }
  regions.emplace_back(getResults());
}

OperandRange MemRefMicrokernelOp::getEntrySuccessorOperands(RegionBranchPoint) {
  return getInputsMutable();
}

void MemRefMicrokernelOp::getRegionInvocationBounds(
    ArrayRef<Attribute>, SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.push_back({1, 1});
}

//===----------------------------------------------------------------------===//
// CopyL1TensorOp
//===----------------------------------------------------------------------===//

OpFoldResult CopyTensorOp::fold(FoldAdaptor adaptor) {
  if (auto source = getCopy().getDefiningOp<CopyTensorOp>()) {
    getCopyMutable().set(source.getCopy());
    return getResult();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CopyL1TensorOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

static std::optional<bool>
addressSpaceMatches(bool toL1Memory, Value copy,
                    const bufferization::BufferizationOptions &options = {},
                    SmallVector<Value> *invocationStack = nullptr) {
  FailureOr<BaseMemRefType> copyType =
      invocationStack
          ? bufferization::getBufferType(copy, options, *invocationStack)
          : bufferization::getBufferType(copy, options);
  if (failed(copyType))
    return std::nullopt;

  return toL1Memory ==
         isa_and_nonnull<L1EncodingAttr>(copyType->getMemorySpace());
}

bool CopyTensorOp::resultBufferizesToMemoryWrite(
    OpResult opResult, const bufferization::AnalysisState &state) {
  assert(opResult == getResult() && "no other result");

  // No copy is performed unless the address space does not match.
  // Copy in this context implies that we are writing to the result.
  return addressSpaceMatches(getTransfersToL1(), getCopy(), state.getOptions())
      .value_or(true);
}

bool CopyTensorOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &) {
  assert(opOperand == getCopyMutable() && "have only one operand");

  // We read from the buffer we are copying.
  return true;
}

bool CopyTensorOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &) {
  assert(opOperand == getCopyMutable() && "have only one operand");

  // We do not write into the buffer we are copying.
  return false;
}

AliasingValueList
CopyTensorOp::getAliasingValues(OpOperand &opOperand,
                                const bufferization::AnalysisState &state) {
  assert(opOperand == getCopyMutable() && "have only one operand");

  std::optional<bool> matches =
      addressSpaceMatches(getTransfersToL1(), getCopy(), state.getOptions());
  if (!matches)
    // Assume the worst case.
    return {{getResult(), BufferRelation::Equivalent, /*isDefinite=*/false}};

  // Always a brand-new allocation unless the address space matches and we elide
  // the copy, in which case operand and result alias.
  if (!*matches)
    return {{getResult(), BufferRelation::Equivalent}};
  return {};
}

bool CopyTensorOp::bufferizesToAllocation(Value value) {
  assert(value == getResult() && "have only one result");

  // True is the conservative reply according to the docs.
  return addressSpaceMatches(getTransfersToL1(), getCopy()).value_or(true);
}

FailureOr<BaseMemRefType>
CopyTensorOp::getBufferType(Value value, const BufferizationOptions &options,
                            SmallVector<Value> &invocationStack) {
  assert(value == getResult() && "have only one result");

  bool contained = llvm::is_contained(invocationStack, value);
  if (!contained)
    if (addressSpaceMatches(getTransfersToL1(), getCopy(), options,
                            &invocationStack)
            .value_or(false))
      return bufferization::getBufferType(getCopy(), options, invocationStack);

  // Unless contained in the invocation stack (where we are free to impose the
  // most optimal layout), we do not really impose a specific layout on the
  // result. Contiguous is a good bet for now.
  Attribute memorySpace =
      getTransfersToL1() ? L1EncodingAttr::get(getContext()) : nullptr;
  return getMemRefTypeWithStaticIdentityLayout(getResult().getType(),
                                               memorySpace);
}

LogicalResult CopyTensorOp::bufferize(RewriterBase &rewriter,
                                      const BufferizationOptions &options) {
  if (use_empty()) {
    rewriter.eraseOp(*this);
    return success();
  }

  FailureOr<BaseMemRefType> copyType =
      bufferization::getBufferType(getCopy(), options);
  if (failed(copyType))
    return failure();

  FailureOr<Value> copyBuffer = getBuffer(rewriter, getCopy(), options);
  if (failed(copyBuffer))
    return failure();

  std::optional<bool> matches =
      addressSpaceMatches(getTransfersToL1(), getCopy(), options);
  if (!matches)
    return failure();

  if (*matches) {
    replaceOpWithBufferizedValues(rewriter, getOperation(), *copyBuffer);
    return success();
  }

  FailureOr<BaseMemRefType> allocType =
      bufferization::getBufferType(getResult(), options);
  if (failed(allocType))
    return failure();

  // TODO: Add operands to the op representing the dynamic dimensions of the
  //  result tensor and use them below.
  if (!allocType->hasStaticShape())
    return emitOpError(
        "Bufferizing results with dynamic dimensions is not yet implemented");

  FailureOr<Value> alloc = options.createAlloc(
      rewriter, getLoc(), llvm::cast<MemRefType>(*allocType),
      /*dynShape=*/ValueRange());
  if (failed(alloc))
    return failure();

  if (failed(options.createMemCpy(rewriter, getLoc(), *copyBuffer, *alloc)))
    return failure();

  // Replace op.
  replaceOpWithBufferizedValues(rewriter, getOperation(), *alloc);
  return success();
}
