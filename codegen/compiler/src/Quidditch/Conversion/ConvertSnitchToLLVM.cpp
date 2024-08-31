#include "ConvertSnitchToLLVM.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

using namespace mlir;
using namespace quidditch::Snitch;

namespace {
struct L1MemoryViewOpLowering : ConvertOpToLLVMPattern<L1MemoryViewOp> {
  using ConvertOpToLLVMPattern<L1MemoryViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(L1MemoryViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size;

    this->getMemRefDescriptorSizes(op->getLoc(), op.getType(),
                                   adaptor.getOperands(), rewriter, sizes,
                                   strides, size);

    // TODO: This is horribly hardcoded when it shouldn't be.
    Value l1Address = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(0x10000000));
    Value allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(
        op->getLoc(), rewriter.getType<LLVM::LLVMPointerType>(), l1Address);

    auto memRefDescriptor =
        this->createMemRefDescriptor(op->getLoc(), op.getType(), allocatedPtr,
                                     allocatedPtr, sizes, strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }
};
} // namespace

/// Returns the number of potentially non-contiguous outer dimensions of
/// 'memRefType'. The remaining inner dimensions (i.e. all dimensions at index
/// 'NonContiguousOuterDims' to the MemRef rank) are known to be contiguous.
/// Returns failure if the layout attribute of the MemRef is unsupported.
static FailureOr<size_t> getNumNonContiguousOuterDims(MemRefType memRefType) {
  auto stridesAttr =
      dyn_cast_or_null<StridedLayoutAttr>(memRefType.getLayout());
  if (!stridesAttr) {
    if (memRefType.getLayout() && !memRefType.getLayout().isIdentity())
      return failure();

    // No layout or identity layouts are by definition fully contiguous.
    return 0;
  }

  int64_t innerSize = 1;
  ArrayRef<int64_t> shape = memRefType.getShape();
  ArrayRef<int64_t> strides = stridesAttr.getStrides();
  for (; !shape.empty();
       shape = shape.drop_back(), strides = strides.drop_back()) {
    int64_t dim = shape.back();
    // Unit dims can be dropped alongside the corresponding stride of that dim.
    if (dim == 1)
      continue;

    int64_t stride = strides.back();
    if (ShapedType::isDynamic(stride))
      break;

    if (innerSize != stride)
      break;

    // Note: Dim may be dynamic with the value -1. This intentionally will only
    // fail the 'if' above later if the outer dims are non-zero.
    innerSize *= dim;
  }

  return shape.size();
}

/// Returns true if this MemRef type is known to have a fully contiguous layout.
/// TODO: Could be upstreamed next to
/// 'memref::isStaticShapeAndContiguousRowMajor'
static bool isContiguous(MemRefType memRefType) {
  return getNumNonContiguousOuterDims(memRefType) == 0;
}

namespace {
struct StartDMATransferOp1DLowering
    : ConvertOpToLLVMPattern<StartDMATransferOp> {

  LLVM::LLVMFuncOp dmaStart1DFunc;

  StartDMATransferOp1DLowering(LLVM::LLVMFuncOp dmaStart1DFunc,
                               const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter, /*benefit=*/2),
        dmaStart1DFunc(dmaStart1DFunc) {}

  LogicalResult match(StartDMATransferOp op) const override {
    return success(isContiguous(op.getSource().getType()) &&
                   isContiguous(op.getDest().getType()));
  }

  void rewrite(StartDMATransferOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    MemRefDescriptor sourceDescriptor(adaptor.getSource());
    MemRefDescriptor destDescriptor(adaptor.getDest());

    Value source = sourceDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getSource().getType());
    Value dest = destDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getDest().getType());

    MemRefType sourceMemRef = op.getSource().getType();
    SmallVector<Value> dynamicSizes;
    for (auto [index, dim] : llvm::enumerate(sourceMemRef.getShape()))
      if (ShapedType::isDynamic(dim))
        dynamicSizes.push_back(
            sourceDescriptor.size(rewriter, op->getLoc(), index));

    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    Value totalSize;
    getMemRefDescriptorSizes(
        op->getLoc(),
        // Offsets are not considered an identity layout.
        // Get rid of the layout entirely for the size calculation.
        MemRefType::get(sourceMemRef.getShape(), sourceMemRef.getElementType(),
                        nullptr, sourceMemRef.getMemorySpace()),
        dynamicSizes, rewriter, sizes, strides, totalSize);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, dmaStart1DFunc,
                                              ValueRange{
                                                  dest,
                                                  source,
                                                  totalSize,
                                              });
  }
};

struct StartDMATransferOp2DLowering
    : ConvertOpToLLVMPattern<StartDMATransferOp> {

  LLVM::LLVMFuncOp dmaStart2DFunc;

  StartDMATransferOp2DLowering(LLVM::LLVMFuncOp dmaStart2DFunc,
                               const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), dmaStart2DFunc(dmaStart2DFunc) {}

  LogicalResult
  matchAndRewrite(StartDMATransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType sourceMemRef = op.getSource().getType();
    MemRefType destMemRef = op.getDest().getType();

    // Compute the size of the contiguous inner loop common to both MemRefs and
    // "shave" it off the ends of the shapes and strides. The remaining shapes
    // and strides are considered our outer dimensions.
    FailureOr<size_t> sourceNonContiguous =
        getNumNonContiguousOuterDims(sourceMemRef);
    FailureOr<size_t> destNonContiguous =
        getNumNonContiguousOuterDims(destMemRef);
    if (failed(sourceNonContiguous) || failed(destNonContiguous))
      return failure();
    size_t sharedNonContiguous =
        std::max(*sourceNonContiguous, *destNonContiguous);
    if (sharedNonContiguous == 0)
      return failure();

    Value elementSize = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(),
        rewriter.getI32IntegerAttr(llvm::divideCeil(
            op.getSource().getType().getElementTypeBitWidth(), 8)));
    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, op->getLoc(), op.getSource());

    // Build a loop nest iterating over all outer dimensions - 1 and adjusts the
    // source and destination pointers accordingly. The inner-most outer
    // dimension is used in the DMA call for the repetition count and strides.
    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    for (size_t index : llvm::seq(sharedNonContiguous - 1)) {
      lowerBounds.push_back(zeroIndex);
      steps.push_back(oneIndex);
      upperBounds.push_back(getValueOrCreateConstantIndexOp(
          rewriter, op->getLoc(), sizes[index]));
    }

    Value contiguousSize;
    for (auto index :
         llvm::seq<size_t>(sharedNonContiguous, sourceMemRef.getRank())) {
      Value dim =
          getValueOrCreateConstantIndexOp(rewriter, op->getLoc(), sizes[index]);
      if (!contiguousSize) {
        contiguousSize = dim;
        continue;
      }
      contiguousSize =
          rewriter.create<arith::MulIOp>(op->getLoc(), contiguousSize, dim);
    }
    contiguousSize = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), getIndexType(), contiguousSize);
    contiguousSize =
        rewriter.create<LLVM::MulOp>(op->getLoc(), contiguousSize, elementSize);

    Value completedToken = rewriter.create<CompletedTokenOp>(op->getLoc());

    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, op->getLoc(), lowerBounds, upperBounds, steps, completedToken,
        [&](OpBuilder &builder, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          SmallVector<OpFoldResult> offsets = ivs;
          SmallVector<OpFoldResult> subSizes(sharedNonContiguous - 1,
                                             rewriter.getIndexAttr(1));
          for (unsigned i : llvm::seq<unsigned>(sharedNonContiguous - 1,
                                                sourceMemRef.getRank())) {
            offsets.push_back(rewriter.getIndexAttr(0));
            subSizes.push_back(sizes[i]);
          }
          SmallVector<OpFoldResult> strides(sourceMemRef.getRank(),
                                            rewriter.getIndexAttr(1));

          TypedValue<MemRefType> sourceMemRefSlice =
              rewriter.create<memref::SubViewOp>(loc, op.getSource(), offsets,
                                                 subSizes, strides);
          TypedValue<MemRefType> destMemRefSlice =
              rewriter.create<memref::SubViewOp>(loc, op.getDest(), offsets,
                                                 subSizes, strides);

          auto sourceDescriptor =
              MemRefDescriptor(typeConverter->materializeTargetConversion(
                  rewriter, op->getLoc(),
                  typeConverter->convertType(sourceMemRefSlice.getType()),
                  sourceMemRefSlice));
          auto destDescriptor =
              MemRefDescriptor(typeConverter->materializeTargetConversion(
                  rewriter, op->getLoc(),
                  typeConverter->convertType(destMemRefSlice.getType()),
                  destMemRefSlice));

          Value sourceAdjusted = sourceDescriptor.bufferPtr(
              rewriter, op->getLoc(), *getTypeConverter(),
              sourceMemRefSlice.getType());
          Value destAdjusted = destDescriptor.bufferPtr(
              rewriter, op->getLoc(), *getTypeConverter(),
              destMemRefSlice.getType());

          Value sourceStride =
              sourceDescriptor.stride(builder, loc, sharedNonContiguous - 1);
          sourceStride = rewriter.create<LLVM::MulOp>(
              op->getLoc(), sourceStride, elementSize);
          Value destStride =
              destDescriptor.stride(builder, loc, sharedNonContiguous - 1);
          destStride = rewriter.create<LLVM::MulOp>(op->getLoc(), destStride,
                                                    elementSize);

          Value outerLoopSize =
              sourceDescriptor.size(builder, loc, sharedNonContiguous - 1);
          return {builder
                      .create<LLVM::CallOp>(loc, dmaStart2DFunc,
                                            ValueRange{
                                                destAdjusted,
                                                sourceAdjusted,
                                                contiguousSize,
                                                destStride,
                                                sourceStride,
                                                outerLoopSize,
                                            })
                      .getResult()};
        });

    Type tokenType = typeConverter->convertType(op.getType());
    rewriter.replaceOp(
        op, typeConverter->materializeTargetConversion(
                rewriter, op->getLoc(), tokenType, loopNest.results.front()));
    return success();
  }
};

// TODO: These should not be hardcoded.
constexpr unsigned zeroMemSize = 0x10000;
constexpr unsigned zeroMemAddress = 0x10030000;

struct StartContiguousZeroMemTransferOpOpLowering
    : ConvertOpToLLVMPattern<StartZeroMemTransferOp> {

  LLVM::LLVMFuncOp dmaStart1DFunc;
  LLVM::LLVMFuncOp dmaStart2DFunc;

  StartContiguousZeroMemTransferOpOpLowering(LLVM::LLVMFuncOp dmaStart1DFunc,
                                             LLVM::LLVMFuncOp dmaStart2DFunc,
                                             const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter, /*benefit=*/2),
        dmaStart1DFunc(dmaStart1DFunc), dmaStart2DFunc(dmaStart2DFunc) {}

  LogicalResult
  matchAndRewrite(StartZeroMemTransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isContiguous(op.getFilled().getType()))
      return failure();

    Value zeroPointer = rewriter.create<LLVM::IntToPtrOp>(
        op->getLoc(), rewriter.getType<LLVM::LLVMPointerType>(),
        rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), rewriter.getI32IntegerAttr(zeroMemAddress)));
    Value zeroMemSizeValue = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(zeroMemSize));

    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size;

    auto filledDesc = MemRefDescriptor(adaptor.getFilled());

    MemRefType memRefType = op.getFilled().getType();
    SmallVector<Value> dynamicSizes;
    for (auto [index, shape] : llvm::enumerate(memRefType.getShape()))
      if (ShapedType::isDynamic(shape))
        dynamicSizes.push_back(filledDesc.size(rewriter, op->getLoc(), index));

    // Function does not support strided layout, even if it is contiguous.
    // Lie about it and remove it.
    // TODO: Consider fixing this upstream.
    // TODO: Make a clone method of `MemRefType` that changes just the layout.
    this->getMemRefDescriptorSizes(
        op->getLoc(),
        MemRefType::get(memRefType.getShape(), memRefType.getElementType()),
        dynamicSizes, rewriter, sizes, strides, size);

    Value zero =
        createIndexAttrConstant(rewriter, op->getLoc(), getIndexType(), 0);
    Value bufferPointer = filledDesc.bufferPtr(rewriter, op->getLoc(),
                                               *getTypeConverter(), memRefType);
    Value times2D =
        rewriter.create<LLVM::UDivOp>(op->getLoc(), size, zeroMemSizeValue);
    // Note: This call would not be legal as a 'start_dma_transfer' call as
    // MemRefs do not allow internal aliasing, which the below does via the
    // stride of 0.
    rewriter.create<LLVM::CallOp>(op->getLoc(), dmaStart2DFunc,
                                  ValueRange{bufferPointer, zeroPointer,
                                             zeroMemSizeValue, zeroMemSizeValue,
                                             zero, times2D});
    Value offset =
        rewriter.create<LLVM::MulOp>(op->getLoc(), times2D, zeroMemSizeValue);
    bufferPointer = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), bufferPointer.getType(), rewriter.getI8Type(),
        bufferPointer, offset);
    Value rest =
        rewriter.create<LLVM::URemOp>(op->getLoc(), size, zeroMemSizeValue);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, dmaStart1DFunc, ValueRange{bufferPointer, zeroPointer, rest});
    return success();
  }
};

struct StartZeroMemTransferOpOpLowering
    : ConvertOpToLLVMPattern<StartZeroMemTransferOp> {

  using ConvertOpToLLVMPattern<StartZeroMemTransferOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StartZeroMemTransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType = op.getFilled().getType();

    FailureOr<size_t> nonContiguousDims =
        getNumNonContiguousOuterDims(memRefType);
    if (failed(nonContiguousDims) || nonContiguousDims == 0)
      return failure();

    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, op->getLoc(), op.getFilled());

    OpFoldResult contiguousSize = rewriter.getIndexAttr(1);
    for (OpFoldResult contiguousDim :
         ArrayRef(sizes).drop_front(*nonContiguousDims)) {
      contiguousSize = affine::makeComposedFoldedAffineApply(
          rewriter, op->getLoc(),
          rewriter.getAffineDimExpr(0) * rewriter.getAffineDimExpr(1),
          {contiguousSize, contiguousDim});
    }
    contiguousSize = affine::makeComposedFoldedAffineApply(
        rewriter, op->getLoc(),
        rewriter.getAffineDimExpr(0) *
            rewriter.getAffineConstantExpr(memRefType.getElementTypeBitWidth() /
                                           8),
        {contiguousSize});

    OpFoldResult tileSize = affine::makeComposedFoldedAffineApply(
        rewriter, op->getLoc(),
        rewriter.getAffineDimExpr(0).floorDiv(
            rewriter.getAffineConstantExpr(zeroMemSize)),
        {contiguousSize});

    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    for (size_t index : llvm::seq(*nonContiguousDims - 1)) {
      lowerBounds.push_back(zeroIndex);
      steps.push_back(oneIndex);
      upperBounds.push_back(getValueOrCreateConstantIndexOp(
          rewriter, op->getLoc(), sizes[index]));
    }
    lowerBounds.push_back(zeroIndex);
    steps.push_back(
        getValueOrCreateConstantIndexOp(rewriter, op->getLoc(), tileSize));
    upperBounds.push_back(getValueOrCreateConstantIndexOp(
        rewriter, op->getLoc(), sizes[*nonContiguousDims - 1]));

    // Loop over every non-contiguous dimension to zero every contiguous
    // inner subview.
    Value completedToken = rewriter.create<CompletedTokenOp>(op->getLoc());
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, op->getLoc(), lowerBounds, upperBounds, steps, completedToken,
        [&](OpBuilder &builder, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          SmallVector<OpFoldResult> offsets = ivs;
          SmallVector<OpFoldResult> subSizes(*nonContiguousDims - 1,
                                             rewriter.getIndexAttr(1));
          subSizes.push_back(affine::makeComposedFoldedAffineMin(
              rewriter, loc,
              AffineMap::get(
                  3, 0,
                  {rewriter.getAffineDimExpr(0),
                   rewriter.getAffineDimExpr(1) - rewriter.getAffineDimExpr(2)},
                  getContext()),
              {tileSize, sizes[*nonContiguousDims - 1], offsets.back()}));
          for (unsigned i :
               llvm::seq<unsigned>(*nonContiguousDims, memRefType.getRank())) {
            offsets.push_back(rewriter.getIndexAttr(0));
            subSizes.push_back(sizes[i]);
          }
          SmallVector<OpFoldResult> strides(memRefType.getRank(),
                                            rewriter.getIndexAttr(1));

          TypedValue<MemRefType> subMemRef = rewriter.create<memref::SubViewOp>(
              loc, op.getFilled(), offsets, subSizes, strides);

          auto zeroContMemRefType =
              MemRefType::get(subMemRef.getType().getShape(),
                              subMemRef.getType().getElementType());

          Value zeroPointer = rewriter.create<LLVM::IntToPtrOp>(
              op->getLoc(), rewriter.getType<LLVM::LLVMPointerType>(),
              rewriter.create<LLVM::ConstantOp>(
                  op->getLoc(), rewriter.getI32IntegerAttr(zeroMemAddress)));

          SmallVector<Value> llvmConvertedSizes;
          for (OpFoldResult size : sizes) {
            llvmConvertedSizes.push_back(
                typeConverter->materializeTargetConversion(
                    rewriter, loc, getIndexType(),
                    getValueOrCreateConstantIndexOp(rewriter, loc, size)));
          }
          SmallVector<Value> llvmConvertedStrides;
          llvmConvertedStrides.push_back(
              rewriter.create<LLVM::ConstantOp>(loc, getIndexType(), 1));
          for (Value size :
               llvm::reverse(ArrayRef(llvmConvertedSizes).drop_front())) {
            llvmConvertedStrides.push_back(rewriter.create<LLVM::MulOp>(
                loc, llvmConvertedStrides.back(), size));
          }
          std::reverse(llvmConvertedStrides.begin(),
                       llvmConvertedStrides.end());
          Value llvmMemRef = createMemRefDescriptor(
              loc, zeroContMemRefType, zeroPointer, zeroPointer,
              llvmConvertedSizes, llvmConvertedStrides, rewriter);

          Value memRef = typeConverter->materializeSourceConversion(
              rewriter, loc, zeroContMemRefType, llvmMemRef);
          return {builder.create<StartDMATransferOp>(loc, memRef, subMemRef)};
        });

    Type tokenType = typeConverter->convertType(op.getType());
    rewriter.replaceOp(
        op, typeConverter->materializeTargetConversion(
                rewriter, op->getLoc(), tokenType, loopNest.results.front()));
    return success();
  }
};

struct WaitForDMATransfersOpLowering
    : ConvertOpToLLVMPattern<WaitForDMATransfersOp> {

  using ConvertOpToLLVMPattern<WaitForDMATransfersOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(WaitForDMATransfersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getTokens().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    Value current = adaptor.getTokens().front();
    for (Value iter : llvm::drop_begin(adaptor.getTokens()))
      current = rewriter.create<LLVM::UMaxOp>(op->getLoc(), current, iter);

    Block *prev = op->getBlock();
    Block *body = rewriter.splitBlock(prev, op->getIterator());
    Block *after = rewriter.splitBlock(body, op->getNextNode()->getIterator());
    rewriter.setInsertionPointToEnd(prev);
    rewriter.create<LLVM::BrOp>(op->getLoc(), body);

    rewriter.setInsertionPointToEnd(body);
    Value lastCompleted =
        rewriter
            .create<LLVM::InlineAsmOp>(
                op->getLoc(), /*res=*/rewriter.getI32Type(),
                /*operands=*/ValueRange(),
                // dmstati $0, 0
                // opcode6=0x2b, func3=0, func7=0b100, rd=$0, rs1=zero,
                // rs2=imm5(0)
                ".insn r 0x2b, 0, 0b100, $0, zero, zero\n",
                /*constraints=*/"=r",
                /*has_side_effects=*/true, /*is_align_stack=*/false,
                /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr)
            .getRes();
    Value notDone = rewriter.create<LLVM::ICmpOp>(
        op->getLoc(), LLVM::ICmpPredicate::ult, lastCompleted, current);
    rewriter.create<LLVM::CondBrOp>(op->getLoc(), notDone, body, after);

    rewriter.setInsertionPointToStart(after);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CompletedTokenOpLowering : ConvertOpToLLVMPattern<CompletedTokenOp> {

  using ConvertOpToLLVMPattern<CompletedTokenOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CompletedTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, typeConverter->convertType(op.getType()), 0);
    return success();
  }
};

struct BarrierOpLowering : ConvertOpToLLVMPattern<BarrierOp> {

  using ConvertOpToLLVMPattern<BarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Effectively clobbers all memory by being synchronization point
    // (kind of like atomics).
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op, /*res=*/TypeRange(),
        /*operands=*/ValueRange(), "csrr x0, 0x7C2",
        /*constraints=*/"~{memory}",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr);
    return success();
  }
};

struct MicrokernelFenceOpLowering : ConvertOpToLLVMPattern<MicrokernelFenceOp> {

  using ConvertOpToLLVMPattern<MicrokernelFenceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MicrokernelFenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Sync up the FPU and integer pipelines by creating a (fake) data
    // dependency between an FPU and integer register.
    rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), /*res=*/rewriter.getI32Type(),
        /*operands=*/ValueRange(),
        "fmv.x.w $0, fa0\n"
        "mv $0, $0",
        // Tell the register allocator to allocate `$0` as if returning a
        // register. Also consider it to clob memory given that all
        // side effects of the FPU pipeline only become visible after this
        // instruction.
        /*constraints=*/"=r,~{memory}",
        // 'has_side_effects' is currently set to true due to a bug in MLIR
        // DCEing despite the memory clobber.
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CallMicrokernelOpLowering : ConvertOpToLLVMPattern<CallMicrokernelOp> {
  mutable SymbolTable symbolTable;

  CallMicrokernelOpLowering(SymbolTable symbolTable,
                            const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), symbolTable(std::move(symbolTable)) {
  }

  LogicalResult
  matchAndRewrite(CallMicrokernelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::LLVMFuncOp kernelDecl;
    {
      OpBuilder::InsertionGuard guard{rewriter};
      rewriter.setInsertionPointToEnd(
          &symbolTable.getOp()->getRegion(0).back());

      SmallVector<Type> types;
      for (Type type : op.getInputs().getTypes()) {
        if (auto memRefType = dyn_cast<MemRefType>(type)) {
          // Pretend layouts don't exit.
          type = MemRefType::get(
              memRefType.getShape(), memRefType.getElementType(),
              /*layout=*/nullptr, memRefType.getMemorySpace());
        }
        Type converted = getTypeConverter()->convertCallingConventionType(
            type, /*useBarePointerCallConv=*/true);
        if (!converted)
          return failure();

        types.push_back(converted);
      }

      kernelDecl = rewriter.create<LLVM::LLVMFuncOp>(
          op.getLoc(), op.getName(),
          LLVM::LLVMFunctionType::get(rewriter.getType<LLVM::LLVMVoidType>(),
                                      types));
      symbolTable.insert(kernelDecl);

      // Required to tell the conversion pass to LLVM that this is actually a
      // call into the same linkage unit, and does not have to be rewritten to a
      // HAL module call.
      kernelDecl->setAttr("hal.import.bitcode", rewriter.getUnitAttr());

      cast<QuidditchSnitchDialect>(op->getDialect())
          ->getRiscvAssemblyAttrHelper()
          .setAttr(kernelDecl, op.getRiscvAssemblyAttr());
    }

    SmallVector<Value> inputs;
    for (auto [value, oldType] :
         llvm::zip_equal(adaptor.getInputs(), op.getInputs().getType())) {
      auto memRefType = dyn_cast<MemRefType>(oldType);
      if (!memRefType) {
        inputs.push_back(value);
        continue;
      }
      auto descriptor = MemRefDescriptor(value);
      inputs.push_back(descriptor.bufferPtr(rewriter, op->getLoc(),
                                            *getTypeConverter(), memRefType));
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, kernelDecl, inputs);
    return success();
  }
};

struct ComputeCoreIndexOpLowering : ConvertOpToLLVMPattern<ComputeCoreIndexOp> {

  LLVM::LLVMFuncOp computeCoreIndexFunc;

  ComputeCoreIndexOpLowering(LLVM::LLVMFuncOp computeCoreIndexFunc,
                             const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter),
        computeCoreIndexFunc(computeCoreIndexFunc) {}

  LogicalResult
  matchAndRewrite(ComputeCoreIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, computeCoreIndexFunc,
                                              ValueRange());
    return success();
  }
};

} // namespace

void quidditch::populateSnitchToLLVMConversionPatterns(
    mlir::ModuleOp moduleOp, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {

  typeConverter.addConversion([](DMATokenType token) {
    return IntegerType::get(token.getContext(), 32);
  });

  auto builder = OpBuilder::atBlockEnd(moduleOp.getBody());
  auto ptrType = builder.getType<LLVM::LLVMPointerType>();
  IntegerType i32 = builder.getI32Type();
  IntegerType sizeT = i32;
  auto dmaStart1D = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_dma_start_1d",
      LLVM::LLVMFunctionType::get(i32,
                                  ArrayRef<Type>{ptrType, ptrType, sizeT}));
  dmaStart1D->setAttr("hal.import.bitcode", builder.getUnitAttr());

  auto dmaStart2D = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_dma_start_2d",
      LLVM::LLVMFunctionType::get(
          i32, ArrayRef<Type>{ptrType, ptrType, sizeT, sizeT, sizeT, sizeT}));
  dmaStart2D->setAttr("hal.import.bitcode", builder.getUnitAttr());

  auto computeCoreIndex = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_cluster_core_idx",
      LLVM::LLVMFunctionType::get(i32, ArrayRef<Type>{}));
  computeCoreIndex->setAttr("hal.import.bitcode", builder.getUnitAttr());

  patterns
      .insert<L1MemoryViewOpLowering, CompletedTokenOpLowering,
              BarrierOpLowering, MicrokernelFenceOpLowering,
              WaitForDMATransfersOpLowering, StartZeroMemTransferOpOpLowering>(
          typeConverter);
  patterns.insert<StartDMATransferOp1DLowering>(dmaStart1D, typeConverter);
  patterns.insert<StartDMATransferOp2DLowering>(dmaStart2D, typeConverter);
  patterns.insert<StartContiguousZeroMemTransferOpOpLowering>(
      dmaStart1D, dmaStart2D, typeConverter);
  patterns.insert<ComputeCoreIndexOpLowering>(computeCoreIndex, typeConverter);
  patterns.insert<CallMicrokernelOpLowering>(SymbolTable(moduleOp),
                                             typeConverter);
}
