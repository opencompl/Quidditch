#include "DMAOps.h"

#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "DMAAttrs.h"

static mlir::ParseResult
parseTensorCopyTypes(mlir::OpAsmParser &parser,
                     mlir::DenseI64ArrayAttr staticHighPad,
                     mlir::Type &copyType, mlir::Type &resultType);

static void printTensorCopyTypes(mlir::OpAsmPrinter &printer, mlir::Operation *,
                                 mlir::DenseI64ArrayAttr staticHighPad,
                                 mlir::Type copyType, mlir::Type resultType);

#define GET_OP_CLASSES
#include "Quidditch/Dialect/DMA/IR/DMAOps.cpp.inc"

using namespace mlir;
using namespace mlir::bufferization;
using namespace quidditch::dma;

//===----------------------------------------------------------------------===//
// StartTensorCopyOp
//===----------------------------------------------------------------------===//

ParseResult parseTensorCopyTypes(OpAsmParser &parser,
                                 DenseI64ArrayAttr staticHighPad,
                                 Type &copyType, Type &resultType) {
  if (staticHighPad && !staticHighPad.empty()) {
    if (parser.parseColon() || parser.parseType(copyType))
      return failure();
  }
  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();
  if (!staticHighPad || staticHighPad.empty())
    copyType = resultType;
  return success();
}

static void printTensorCopyTypes(OpAsmPrinter &printer, mlir::Operation *,
                                 DenseI64ArrayAttr staticHighPad, Type copyType,
                                 Type resultType) {
  if (staticHighPad && !staticHighPad.empty())
    printer << ": " << copyType;
  printer << " -> " << resultType;
}

LogicalResult StartTensorCopyOp::verify() {
  if (getStaticHighPadAttr())
    if (getStaticHighPadAttr().size() != getCopy().getType().getRank())
      return emitOpError("expected padding number for every dimension");

  unsigned numDynamicPads = llvm::count(
      getStaticHighPad().value_or(std::nullopt), ShapedType::kDynamic);
  if (numDynamicPads != getHighPad().size())
    return emitOpError("expected ")
           << numDynamicPads << " dynamic padding values";

  return success();
}

LogicalResult StartTensorCopyOp::fold(FoldAdaptor adaptor,
                                      SmallVectorImpl<OpFoldResult> &results) {
  if (hasPadding()) {
    // Remove noop padding.
    if (llvm::all_of(getStaticHighPadAttr().asArrayRef(),
                     [](int64_t value) { return value == 0; })) {
      removeStaticHighPadAttr();
      return success();
    }

    // Fold dynamic indices with constant values into the static list.
    {
      bool changed = false;
      SmallVector<int64_t> padding =
          llvm::to_vector(getStaticHighPadAttr().asArrayRef());
      unsigned dynamicIndex = 0;
      for (int64_t &value : padding) {
        if (!ShapedType::isDynamic(value))
          continue;

        if (auto integer = dyn_cast_or_null<IntegerAttr>(
                adaptor.getHighPad()[dynamicIndex])) {
          value = integer.getValue().getZExtValue();
          getHighPadMutable().erase(dynamicIndex);
          changed = true;
        } else {
          dynamicIndex++;
        }
      }
      if (changed) {
        setStaticHighPad(padding);
        return success();
      }
    }
  }

  auto waitOp = getCopy().getDefiningOp<WaitForTensorCopyOp>();
  if (!waitOp)
    return failure();
  auto copyOp = waitOp.getTransferTensor().getDefiningOp<StartTensorCopyOp>();
  if (!copyOp)
    return failure();

  if (hasPadding() &&
      (copyOp.getStaticHighPadAttr() != getStaticHighPadAttr() ||
       copyOp.getHighPad() != getHighPad()))
    return failure();

  results.emplace_back(waitOp);
  results.emplace_back(CompletedTokenAttr::get(getContext()));
  return success();
}

SmallVector<OpFoldResult> StartTensorCopyOp::getMixedHighPad() {
  Builder builder(getContext());
  if (!hasPadding())
    return SmallVector<OpFoldResult>(getResult().getType().getRank(),
                                     builder.getIndexAttr(0));

  return getMixedValues(getStaticHighPadAttr().asArrayRef(), getHighPad(),
                        builder);
}

//===----------------------------------------------------------------------===//
// StartTensorCopyOp::BufferizableOpInterface
//===----------------------------------------------------------------------===//

/// Returns whether the allocation can be elided entirely.
/// Returns an empty optional if it was not possible to determine.
std::optional<bool> StartTensorCopyOp::elidesAllocation(
    const bufferization::BufferizationOptions &options,
    SmallVector<Value> *invocationStack) {
  // Padding cannot be elided in general, even if the copied buffer is in L1.
  if (hasPadding())
    return false;

  FailureOr<BaseMemRefType> copyType =
      invocationStack
          ? bufferization::getBufferType(getCopy(), options, *invocationStack)
          : bufferization::getBufferType(getCopy(), options);
  if (failed(copyType))
    return std::nullopt;

  return copyType->getMemorySpace() == getMemorySpaceAttr();
}

bool StartTensorCopyOp::resultBufferizesToMemoryWrite(
    OpResult opResult, const bufferization::AnalysisState &state) {
  assert(opResult == getResult() && "no other result");

  std::optional<bool> matches = elidesAllocation(state.getOptions());
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

  std::optional<bool> result = elidesAllocation(state.getOptions());
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

  std::optional<bool> result = elidesAllocation(state.getOptions());
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

  if (elidesAllocation() == true)
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
    if (elidesAllocation(options, &invocationStack) == true)
      return bufferization::getBufferType(getCopy(), options, invocationStack);

  // Unless contained in the invocation stack (where we are free to impose the
  // most optimal layout), we do not really impose a specific layout on the
  // result. Contiguous is a good bet for now.
  return getMemRefTypeWithStaticIdentityLayout(getResult().getType(),
                                               getMemorySpaceAttr());
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

  std::optional<bool> result = elidesAllocation(options);
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

  SmallVector<OpFoldResult> copyBufferSizes =
      memref::getMixedSizes(rewriter, getLoc(), *copyBuffer);

  // Compute the dynamic dimensions for the allocation.
  SmallVector<Value> dynamicDims;
  for (auto [index, shape, pad] :
       llvm::enumerate(allocType->getShape(), getMixedHighPad())) {
    if (!ShapedType::isDynamic(shape))
      continue;

    dynamicDims.push_back(affine::makeComposedAffineApply(
        rewriter, getLoc(),
        rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1),
        ArrayRef<OpFoldResult>{copyBufferSizes[index], pad}));
  }

  FailureOr<Value> alloc = options.createAlloc(
      rewriter, getLoc(), llvm::cast<MemRefType>(*allocType),
      /*dynShape=*/dynamicDims);
  if (failed(alloc))
    return failure();

  // Zero out the entire buffer prior to overwriting it with the copied values.
  // TODO: This could be optimized to only zero regions that won't be filled
  //  with the copied values at the cost of 2^rank transfers instead of two.
  if (hasPadding() && !getUndefPadding())
    rewriter.create<StartZeroMemTransferOp>(getLoc(), *alloc);

  // Subview into the original memory without any padding.
  // As we only add padding at the end of the dimensions, the offsets are always
  // zero.
  Value destination = rewriter.create<memref::SubViewOp>(
      getLoc(), *alloc,
      /*offsets=*/
      SmallVector<OpFoldResult>(allocType->getRank(), rewriter.getIndexAttr(0)),
      copyBufferSizes,
      /*strides=*/
      SmallVector<OpFoldResult>(allocType->getRank(),
                                rewriter.getIndexAttr(1)));
  Value token =
      rewriter.create<StartTransferOp>(getLoc(), *copyBuffer, destination);

  // Replace op.
  replaceOpWithBufferizedValues(rewriter, getOperation(), {*alloc, token});
  return success();
}

//===----------------------------------------------------------------------===//
// WaitForTensorCopyOp
//===----------------------------------------------------------------------===//

OpFoldResult WaitForTensorCopyOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getToken())
    return getTransferTensor();

  return nullptr;
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

  rewriter.create<WaitForTransfersOp>(getLoc(), getToken());
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
// CompletedTokenOp
//===----------------------------------------------------------------------===//

OpFoldResult CompletedTokenOp::fold(FoldAdaptor adaptor) {
  return CompletedTokenAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// StartTransferOp
//===----------------------------------------------------------------------===//

OpFoldResult StartTransferOp::fold(FoldAdaptor adaptor) {
  if (getSource() != getDest())
    return nullptr;

  return CompletedTokenAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// WaitForTransfersOp
//===----------------------------------------------------------------------===//

LogicalResult WaitForTransfersOp::fold(FoldAdaptor adaptor,
                                       SmallVectorImpl<OpFoldResult> &results) {
  bool changed = false;
  MutableOperandRange tokens = getTokensMutable();
  for (int i = tokens.size() - 1; i >= 0; i--) {
    if (adaptor.getTokens()[i]) {
      changed = true;
      tokens.erase(i);
    }
  }
  return success(changed);
}

LogicalResult WaitForTransfersOp::canonicalize(WaitForTransfersOp op,
                                               PatternRewriter &rewriter) {
  if (!op.getTokens().empty())
    return failure();

  rewriter.eraseOp(op);
  return success();
}
