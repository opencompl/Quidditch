#include "ConvertSnitchToLLVM.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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

  auto builder = OpBuilder::atBlockEnd(moduleOp.getBody());
  IntegerType i32 = builder.getI32Type();
  auto computeCoreIndex = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_cluster_core_idx",
      LLVM::LLVMFunctionType::get(i32, ArrayRef<Type>{}));
  computeCoreIndex->setAttr("hal.import.bitcode", builder.getUnitAttr());

  patterns.insert<L1MemoryViewOpLowering, BarrierOpLowering,
                  MicrokernelFenceOpLowering>(typeConverter);
  patterns.insert<ComputeCoreIndexOpLowering>(computeCoreIndex, typeConverter);
  patterns.insert<CallMicrokernelOpLowering>(SymbolTable(moduleOp),
                                             typeConverter);
}
