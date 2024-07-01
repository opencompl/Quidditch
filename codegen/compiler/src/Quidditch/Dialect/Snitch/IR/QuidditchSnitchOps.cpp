#include "QuidditchSnitchOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "QuidditchSnitchAttrs.h"

#define GET_OP_CLASSES
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.cpp.inc"

using namespace mlir;
using namespace mlir::bufferization;
using namespace quidditch::Snitch;

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
