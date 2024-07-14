#include "QuidditchSnitchOps.h"

#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
// StartTensorCopyOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

/// Returns whether 'copy' is already in L1 memory.
/// Returns an empty optional if it was not possible to determine.
static std::optional<bool>
isInL1Memory(Value copy,
             const bufferization::BufferizationOptions &options = {},
             SmallVector<Value> *invocationStack = nullptr) {
  FailureOr<BaseMemRefType> copyType =
      invocationStack
          ? bufferization::getBufferType(copy, options, *invocationStack)
          : bufferization::getBufferType(copy, options);
  if (failed(copyType))
    return std::nullopt;

  return isa_and_nonnull<L1EncodingAttr>(copyType->getMemorySpace());
}

bool StartTensorCopyOp::resultBufferizesToMemoryWrite(
    OpResult opResult, const bufferization::AnalysisState &state) {
  assert(opResult == getResult() && "no other result");

  std::optional<bool> matches = isInL1Memory(getCopy(), state.getOptions());
  // Conservative answer.
  if (!matches)
    return true;

  // No copy is performed unless the address space does not match.
  // Copy in this context implies that we are writing to the result.
  return !*matches;
}

bool StartTensorCopyOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  assert(opOperand == getCopyMutable() && "have only one operand");

  std::optional<bool> result = isInL1Memory(getCopy(), state.getOptions());
  // Conservative answer.
  if (!result)
    return true;

  // We only read from the buffer if we are copying.
  return !*result;
}

bool StartTensorCopyOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &) {
  assert(opOperand == getCopyMutable() && "have only one operand");

  // We do not write into the buffer we are copying ever.
  return false;
}

AliasingValueList StartTensorCopyOp::getAliasingValues(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  assert(opOperand == getCopyMutable() && "have only one operand");

  std::optional<bool> result = isInL1Memory(getCopy(), state.getOptions());
  if (!result)
    // Assume the worst case.
    return {{getResult(), BufferRelation::Equivalent, /*isDefinite=*/false}};

  // Always a brand-new allocation unless the input buffer is already in L1 and
  // we elide the copy, in which case operand and result alias.
  if (*result)
    return {{getResult(), BufferRelation::Equivalent, /*isDefinite=*/true}};

  return {};
}

bool StartTensorCopyOp::bufferizesToAllocation(Value value) {
  assert(value == getResult() && "have only one result");

  if (isInL1Memory(getCopy()) == true)
    return false;

  // True is the conservative reply, according to the docs.
  return true;
}

FailureOr<BaseMemRefType>
StartTensorCopyOp::getBufferType(Value value,
                                 const BufferizationOptions &options,
                                 SmallVector<Value> &invocationStack) {
  assert(value == getResult() && "have only one result");

  bool contained = llvm::is_contained(invocationStack, value);
  if (!contained)
    if (isInL1Memory(getCopy(), options, &invocationStack) == true)
      return bufferization::getBufferType(getCopy(), options, invocationStack);

  // Unless contained in the invocation stack (where we are free to impose the
  // most optimal layout), we do not really impose a specific layout on the
  // result. Contiguous is a good bet for now.
  return getMemRefTypeWithStaticIdentityLayout(
      getResult().getType(), L1EncodingAttr::get(getContext()));
}

LogicalResult
StartTensorCopyOp::bufferize(RewriterBase &rewriter,
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

  std::optional<bool> result = isInL1Memory(getCopy(), options);
  if (!result)
    return failure();

  if (*result) {
    Value token = rewriter.create<CompletedTokenOp>(getLoc());
    replaceOpWithBufferizedValues(rewriter, getOperation(),
                                  {*copyBuffer, token});
    return success();
  }

  FailureOr<BaseMemRefType> allocType =
      bufferization::getBufferType(getResult(), options);
  if (failed(allocType))
    return failure();

  SmallVector<Value> dynamicDims;
  for (auto [index, shape] : llvm::enumerate(allocType->getShape())) {
    if (!ShapedType::isDynamic(shape))
      continue;
    dynamicDims.push_back(
        rewriter.create<memref::DimOp>(getLoc(), *copyBuffer, index));
  }

  FailureOr<Value> alloc = options.createAlloc(
      rewriter, getLoc(), llvm::cast<MemRefType>(*allocType),
      /*dynShape=*/dynamicDims);
  if (failed(alloc))
    return failure();

  Value token =
      rewriter.create<StartDMATransferOp>(getLoc(), *copyBuffer, *alloc);

  // Replace op.
  replaceOpWithBufferizedValues(rewriter, getOperation(), {*alloc, token});
  return success();
}

//===----------------------------------------------------------------------===//
// WaitForTensorCopyOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

bool WaitForTensorCopyOp::mustBufferizeInPlace(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  return true;
}

bool WaitForTensorCopyOp::bufferizesToMemoryRead(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  if (opOperand == getTransferTensorMutable())
    return false;

  if (opOperand == getCopyMutable())
    return true;

  llvm_unreachable("unknown operand");
}

bool WaitForTensorCopyOp::bufferizesToMemoryWrite(
    OpOperand &opOperand, const bufferization::AnalysisState &) {
  if (opOperand == getTransferTensorMutable())
    return true;

  if (opOperand == getCopyMutable())
    return false;

  llvm_unreachable("unknown operand");
}

AliasingValueList WaitForTensorCopyOp::getAliasingValues(
    OpOperand &opOperand, const bufferization::AnalysisState &state) {
  if (opOperand == getCopyMutable())
    return {};

  if (opOperand == getTransferTensorMutable())
    return {{getResult(), BufferRelation::Equivalent, /*isDefinite=*/true}};

  llvm_unreachable("unknown operand");
}

LogicalResult
WaitForTensorCopyOp::bufferize(RewriterBase &rewriter,
                               const BufferizationOptions &options) {
  FailureOr<Value> transferTensorBuffer =
      getBuffer(rewriter, getTransferTensor(), options);
  if (failed(transferTensorBuffer))
    return failure();

  rewriter.create<WaitForDMATransfersOp>(getLoc(), getToken());
  replaceOpWithBufferizedValues(rewriter, getOperation(),
                                *transferTensorBuffer);
  return success();
}

bool WaitForTensorCopyOp::isNotConflicting(
    OpOperand *uRead, OpOperand *uWrite,
    const bufferization::AnalysisState &state) {
  if (*uRead == getCopyMutable() && *uWrite == getTransferTensorMutable())
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// StartDMATransferOp
//===----------------------------------------------------------------------===//

LogicalResult StartDMATransferOp::canonicalize(StartDMATransferOp op,
                                               PatternRewriter &rewriter) {
  if (op.getSource() != op.getDest())
    return failure();

  rewriter.replaceOpWithNewOp<CompletedTokenOp>(op);
  return success();
}

//===----------------------------------------------------------------------===//
// WaitForDMATransfersOp
//===----------------------------------------------------------------------===//

LogicalResult
WaitForDMATransfersOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  bool changed = false;
  MutableOperandRange tokens = getTokensMutable();
  for (int i = tokens.size() - 1; i >= 0; i--) {
    if (tokens[i].get().getDefiningOp<CompletedTokenOp>()) {
      changed = true;
      tokens.erase(i);
    }
  }
  return success(changed);
}

LogicalResult WaitForDMATransfersOp::canonicalize(WaitForDMATransfersOp op,
                                                  PatternRewriter &rewriter) {
  if (!op.getTokens().empty())
    return failure();

  rewriter.eraseOp(op);
  return success();
}
