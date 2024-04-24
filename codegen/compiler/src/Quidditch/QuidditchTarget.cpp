
#include "iree/compiler/PluginAPI/Client.h"

#include "compiler/plugins/target/LLVMCPU/LLVMIRPasses.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Linker/Linker.h"

#include "compiler/plugins/target/LLVMCPU/LibraryBuilder.h"

#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "compiler/plugins/target/LLVMCPU/StaticLibraryGenerator.h"

#include "compiler/plugins/target/LLVMCPU/Builtins/Device.h"
#include "compiler/src/iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

class QuidditchTargetDevice final : public IREE::HAL::TargetDevice {
public:
  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("quidditch")
        ->getDefaultExecutableTargets(context, "quidditch",
                                      b.getDictionaryAttr({}),
                                      executableTargetAttrs);

    std::optional<IREE::HAL::LLVMTarget> target = IREE::HAL::LLVMTarget::create(
        "riscv32-unknown-unknown-elf", "generic-rv32", "+m,+f,+d,+zfh",
        /*requestLinkEmbedded=*/false);
    if (!target) {
      emitError(b.getUnknownLoc()) << "Failed to find RISC-V target for "
                                      "snitch. Is the backend disabled?";
      return nullptr;
    }
    // target->populateDefaultsFromTargetMachine();

    target->linkStatic = true;
    target->llvmTargetOptions.FloatABIType = llvm::FloatABI::Hard;
    target->llvmTargetOptions.MCOptions.ABIName = "ilp32d";

    SmallVector<NamedAttribute> configItems;
    target->storeToConfigAttrs(context, configItems);
    executableTargetAttrs.push_back(b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("llvm-cpu"), b.getStringAttr("static"),
        b.getDictionaryAttr(configItems)));

    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr("quidditch_device"), b.getDictionaryAttr({}),
        executableTargetAttrs);
  }
};

class QuidditchTargetBackend final : public IREE::HAL::TargetBackend {

public:
  [[nodiscard]] std::string getLegacyDefaultDeviceID() const override {
    return "quidditch_device";
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(
        IREE::HAL::ExecutableTargetAttr::get(context, "quidditch", "snitch"));
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    OpPassManager &modulePassManager = passManager.nest<ModuleOp>();

    // Bufferize to memref as that is what xDSL currently needs.
    addCPUDefaultPassPipeline(modulePassManager.nest<func::FuncOp>());

    // TODO: Remove the following pass and plumb support for
    // #hal.descriptor_type memory space through the stack.
    FunctionLikeNest(modulePassManager)
        .addPass(createEraseHALDescriptorTypeFromMemRefPass);

    FunctionLikeNest(modulePassManager)
        // LinalgExt -> SCF
        .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
        // Linalg -> SCF
        .addPass(createMemrefCopyToLinalgPass)
        .addPass(createConvertLinalgToLoopsPass)
        .addPass(createConvertBf16ArithToF32Pass)
        .addPass(createConvertBf16ToUInt16BuffersPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    // Handled tensor-type constants.
    modulePassManager.addPass(arith::createConstantBufferizePass());

    FunctionLikeNest(modulePassManager)
        .addPass(createFoldTensorExtractOpPass)
        // Handle complex operation conversion.
        .addPass(createConvertComplexToStandardPass)
        // math dialect elementry functions -> polynomial form.
        .addPass(createPolynomialApproximationPass)
        .addPass(createHoistStaticallyBoundAllocationsPass);

    FunctionLikeNest(modulePassManager)
        // Resolve get_buffer_descriptor ops. All structural buffer
        // manipulations must conclude before this point.
        .addPass(createIREEExpandStridedMetadataPass)
        .addPass(createCleanupBufferAllocViewPass);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    [] {}();
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // TODO: This is a 'builtin.module' containing a 'func.func' containing the
    //       kernel which consists of memref, linalg, scf and some 'hal' ops for
    //       I/O. It should be lowered to an object file that is later linked
    //       into the final executable.
    [[maybe_unused]] ModuleOp module = variantOp.getInnerModule();

    // TODO: Be inspired by or use the static library export in the LLVMCPU
    //       target: It just exports the library name for the purpose of the
    //       runtime and generates an entry point for the runtime.
    //
    //       See runtime/src/iree/hal/local/loaders/static_library_loader.c
    //       and compiler/plugins/target/LLVMCPU/LLVMCPUTarget.cpp:360
    std::string libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // Embed the library name in the executable binary op. This informs the
    // loader which static library to load for the target binary.
    std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                           libraryName.end());
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(), "static",
        libraryNameVector);

    return success();
  }
};

class QuidditchSession final
    : public PluginSession<QuidditchSession, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("quidditch_device",
                []() { return std::make_shared<QuidditchTargetDevice>(); });
  }

  void
  populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) override {
    targets.add("quidditch",
                []() { return std::make_shared<QuidditchTargetBackend>(); });
  }
};
} // namespace

extern "C" bool iree_register_compiler_plugin_hal_target_quidditch(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<QuidditchSession>("hal_target_quidditch");
  return true;
}
