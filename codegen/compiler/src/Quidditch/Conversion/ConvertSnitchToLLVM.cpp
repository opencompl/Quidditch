#include "Passes.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

namespace quidditch {
#define GEN_PASS_DEF_CONVERTSNITCHTOLLVMPASS
#include "Quidditch/Conversion/Passes.h.inc"
} // namespace quidditch

namespace {
class ConvertSnitchToLLVM
    : public quidditch::impl::ConvertSnitchToLLVMPassBase<ConvertSnitchToLLVM> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

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

/// Returns true if this MemRef type is known to have a fully contiguous layout.
/// TODO: Could be upstreamed next to
/// 'memref::isStaticShapeAndContiguousRowMajor'
bool isContiguous(MemRefType memRefType) {
  MemRefLayoutAttrInterface layout = memRefType.getLayout();
  if (!layout || layout.isIdentity())
    return true;

  // It is impossible to statically determine contiguity with dynamic strides.
  auto strided = dyn_cast<StridedLayoutAttr>(layout);
  if (!strided || llvm::any_of(strided.getStrides(), ShapedType::isDynamic))
    return false;

  // Calculate what the strides would be if it had an identity layout and check
  // that they match.
  ArrayRef<int64_t> shape = memRefType.getShape();
  ArrayRef<int64_t> strides = strided.getStrides();
  std::uint64_t currentIdentityStride = 1;
  for (auto [dim, stride] : llvm::zip_equal(llvm::reverse(shape.drop_front()),
                                            strides.drop_front())) {
    // Unit dimensions are noops in regards to strides.
    if (dim == 1)
      continue;

    if (currentIdentityStride != stride)
      return false;

    if (ShapedType::isDynamic(dim))
      return false;
    currentIdentityStride *= dim;
  }

  // Unit dimensions are noops in regards to strides.
  if (shape.front() == 1)
    return true;

  return currentIdentityStride == strides.front();
}

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
    for (std::int64_t dim : sourceMemRef.getShape())
      if (ShapedType::isDynamic(dim))
        dynamicSizes.push_back(
            sourceDescriptor.size(rewriter, op->getLoc(), dim));

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

  static StridedLayoutAttr identityStride(MemRefType type) {
    SmallVector<int64_t> strides{1};
    for (int64_t dim : llvm::reverse(type.getShape().drop_back())) {
      if (ShapedType::isDynamic(dim))
        break;
      strides.push_back(strides.back() * dim);
    }

    while (strides.size() < type.getShape().size())
      strides.push_back(ShapedType::kDynamic);

    std::reverse(strides.begin(), strides.end());
    return StridedLayoutAttr::get(type.getContext(), 0, strides);
  }

  LogicalResult
  matchAndRewrite(StartDMATransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType sourceMemRef = op.getSource().getType();
    MemRefType destMemRef = op.getDest().getType();

    StridedLayoutAttr sourceStridesAttr =
        dyn_cast_or_null<StridedLayoutAttr>(sourceMemRef.getLayout());
    if (!sourceStridesAttr) {
      if (sourceMemRef.getLayout() && !sourceMemRef.getLayout().isIdentity())
        return failure();

      sourceStridesAttr = identityStride(sourceMemRef);
    }

    StridedLayoutAttr destStridesAttr =
        dyn_cast_or_null<StridedLayoutAttr>(destMemRef.getLayout());
    if (!destStridesAttr) {
      if (destMemRef.getLayout() && !destMemRef.getLayout().isIdentity())
        return failure();

      destStridesAttr = identityStride(destMemRef);
    }

    // Compute the size of the contiguous inner loop common to both MemRefs and
    // "shave" it off the ends of the shapes and strides. The remaining shapes
    // and strides are considered our outer dimensions.
    int64_t innerSize = 1;
    ArrayRef<int64_t> shape = sourceMemRef.getShape();
    ArrayRef<int64_t> sourceStrides = sourceStridesAttr.getStrides();
    ArrayRef<int64_t> destStrides = destStridesAttr.getStrides();
    assert(shape.size() == sourceStrides.size() &&
           sourceStrides.size() == destStrides.size());
    for (; shape.size() > 1; shape = shape.drop_back(),
                             sourceStrides = sourceStrides.drop_back(),
                             destStrides = destStrides.drop_back()) {
      int64_t dim = shape.back();
      if (dim == 1)
        continue;

      int64_t sourceStride = sourceStrides.back();
      int64_t destStride = destStrides.back();
      if (sourceStride != destStride)
        break;

      if (innerSize != sourceStride)
        break;

      if (ShapedType::isDynamic(dim))
        break;

      innerSize *= dim;
    }

    MemRefDescriptor sourceDescriptor(adaptor.getSource());
    MemRefDescriptor destDescriptor(adaptor.getDest());

    Value source = sourceDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getSource().getType());
    Value dest = destDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getDest().getType());

    Value elementSize = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(),
        rewriter.getI32IntegerAttr(llvm::divideCeil(
            op.getSource().getType().getElementTypeBitWidth(), 8)));
    Value contiguousSize = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(innerSize));
    contiguousSize =
        rewriter.create<LLVM::MulOp>(op->getLoc(), contiguousSize, elementSize);

    // Build a loop nest iterating over all outer dimensions - 1 and adjusts the
    // source and destination pointers accordingly. The inner-most outer
    // dimension is used in the DMA call for the repetition count and strides.
    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    for (size_t index : llvm::seq(shape.size() - 1)) {
      lowerBounds.push_back(zeroIndex);
      steps.push_back(oneIndex);
      Value dim = typeConverter->materializeSourceConversion(
          rewriter, op->getLoc(), rewriter.getIndexType(),
          sourceDescriptor.size(rewriter, op->getLoc(), index));
      upperBounds.push_back(dim);
    }

    Type tokenType = typeConverter->convertType(op.getType());
    Value completedToken = rewriter.create<CompletedTokenOp>(op->getLoc());
    completedToken = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), tokenType, completedToken);

    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, op->getLoc(), lowerBounds, upperBounds, steps, completedToken,
        [&](OpBuilder &builder, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          auto linearizeOffset = [&](MemRefDescriptor descriptor) {
            Value offset =
                rewriter.create<LLVM::ZeroOp>(loc, rewriter.getI32Type());
            for (auto [index, iv] : llvm::enumerate(ivs)) {
              Value increment = rewriter.create<LLVM::MulOp>(
                  loc,
                  typeConverter->materializeTargetConversion(
                      builder, op->getLoc(),
                      typeConverter->convertType(iv.getType()), iv),
                  descriptor.stride(builder, loc, index));
              offset = rewriter.create<LLVM::AddOp>(loc, offset, increment);
            }
            return offset;
          };

          Value sourceAdjusted = rewriter.create<LLVM::GEPOp>(
              loc, source.getType(),
              typeConverter->convertType(sourceMemRef.getElementType()), source,
              linearizeOffset(sourceDescriptor));
          Value destAdjusted = rewriter.create<LLVM::GEPOp>(
              loc, dest.getType(),
              typeConverter->convertType(destMemRef.getElementType()), dest,
              linearizeOffset(destDescriptor));

          Value sourceStride =
              sourceDescriptor.stride(builder, loc, sourceStrides.size() - 1);
          sourceStride = rewriter.create<LLVM::MulOp>(
              op->getLoc(), sourceStride, elementSize);
          Value destStride =
              destDescriptor.stride(builder, loc, destStrides.size() - 1);
          destStride = rewriter.create<LLVM::MulOp>(op->getLoc(), destStride,
                                                    elementSize);

          Value outerLoopSize =
              sourceDescriptor.size(builder, loc, shape.size() - 1);
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

    rewriter.replaceOp(op, loopNest.results.front());
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

    // Creating yummy dialect soup here. This hackily relies on IREE's
    // ConvertToLLVM lowering the scf in one shot.
    rewriter.create<scf::WhileOp>(
        op->getLoc(), /*resultTypes=*/TypeRange(),
        /*operands=*/ValueRange(),
        [&](OpBuilder &builder, Location loc, ValueRange) {
          Value lastCompleted =
              builder
                  .create<LLVM::InlineAsmOp>(
                      loc, /*res=*/builder.getI32Type(),
                      /*operands=*/ValueRange(),
                      // dmstati $0, 0
                      // opcode6=0x2b, func3=0, func7=0b100, rd=$0, rs1=zero,
                      // rs2=imm5(0)
                      ".insn r 0x2b, 0, 0b100, $0, zero, zero\n",
                      /*constraints=*/"=r",
                      /*has_side_effects=*/true, /*is_align_stack=*/false,
                      /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr)
                  .getRes();
          Value notDone = builder.create<LLVM::ICmpOp>(
              loc, LLVM::ICmpPredicate::ult, lastCompleted, current);
          builder.create<scf::ConditionOp>(loc, notDone, ValueRange());
        },
        [](OpBuilder &builder, Location loc, ValueRange) {
          builder.create<scf::YieldOp>(loc);
        });

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
  SymbolTable &symbolTable;

  CallMicrokernelOpLowering(SymbolTable &symbolTable,
                            const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), symbolTable(symbolTable) {}

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

struct ClusterIndexOpLowering : ConvertOpToLLVMPattern<ClusterIndexOp> {

  LLVM::LLVMFuncOp clusterIndexFunc;

  ClusterIndexOpLowering(LLVM::LLVMFuncOp clusterIndexFunc,
                         const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), clusterIndexFunc(clusterIndexFunc) {}

  LogicalResult
  matchAndRewrite(ClusterIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, clusterIndexFunc,
                                              ValueRange());
    return success();
  }
};

} // namespace

void ConvertSnitchToLLVM::runOnOperation() {

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(&getContext(),
                             dataLayoutAnalysis.getAtOrAbove(getOperation()));
  // TODO: This is horribly hardcoded when it shouldn't be.
  options.overrideIndexBitwidth(32);
  LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
  typeConverter.addConversion([](DMATokenType token) {
    return IntegerType::get(token.getContext(), 32);
  });

  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
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

  auto clusterCoreIndex = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_cluster_core_idx",
      LLVM::LLVMFunctionType::get(i32, ArrayRef<Type>{}));
  clusterCoreIndex->setAttr("hal.import.bitcode", builder.getUnitAttr());

  SymbolTable symbolTable(getOperation());
  RewritePatternSet patterns(&getContext());
  patterns.insert<L1MemoryViewOpLowering, CompletedTokenOpLowering,
                  BarrierOpLowering, MicrokernelFenceOpLowering,
                  WaitForDMATransfersOpLowering>(typeConverter);
  patterns.insert<StartDMATransferOp1DLowering>(dmaStart1D, typeConverter);
  patterns.insert<StartDMATransferOp2DLowering>(dmaStart2D, typeConverter);
  patterns.insert<ClusterIndexOpLowering>(clusterCoreIndex, typeConverter);
  patterns.insert<CallMicrokernelOpLowering>(symbolTable, typeConverter);

  LLVMConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto) { return true; });
  target.addIllegalDialect<QuidditchSnitchDialect>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
