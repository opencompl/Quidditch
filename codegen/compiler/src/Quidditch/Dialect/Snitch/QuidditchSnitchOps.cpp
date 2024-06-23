#include "QuidditchSnitchOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "Quidditch/Dialect/Snitch/QuidditchSnitchOps.cpp.inc"

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
