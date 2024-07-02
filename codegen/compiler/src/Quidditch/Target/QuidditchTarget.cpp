
#include "iree/compiler/PluginAPI/Client.h"

#include "compiler/plugins/target/LLVMCPU/LLVMIRPasses.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "Quidditch/Conversion/Passes.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h"

#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "compiler/plugins/target/LLVMCPU/StaticLibraryGenerator.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"

#include "LibraryBuilder.h"
#include "Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace quidditch::Snitch;

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
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr("quidditch_device"), b.getDictionaryAttr({}),
        executableTargetAttrs);
  }
};

struct QuidditchTargetOptions {
  std::string staticLibraryOutputPath;
  std::string xDSLOptPath;
  std::string toolChainRoot;
  bool assertCompiled = false;
  // TODO: This should actually be 112640 but DMA stack overflows. Ooopsie!
  unsigned l1MemoryBytes = 100000;

  void bindOptions(OptionsBinder &binder) {
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVAsmPrinter();
    LLVMInitializeRISCVAsmParser();

    static llvm::cl::OptionCategory category("Quidditch HAL Target");

    binder.opt<std::string>(
        "iree-quidditch-static-library-output-path", staticLibraryOutputPath,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Path to output static object (EX: '/path/to/static-library.o'). "
            "This will produce the static library at the specified path along "
            "with a similarly named '.h' file for static linking."));
    binder.opt<std::string>("iree-quidditch-xdsl-opt-path", xDSLOptPath,
                            llvm::cl::cat(category),
                            llvm::cl::desc("Path to the 'xdsl-opt' executable "
                                           "to use for kernel compilation."));
    binder.opt<std::string>(
        "iree-quidditch-toolchain-root", toolChainRoot, llvm::cl::cat(category),
        llvm::cl::desc("Path to the root directory of the Quidditch toolchain "
                       "(containing the toolchain file)"));
    binder.opt<bool>(
        "iree-quidditch-assert-compiled", assertCompiled,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, errors if any kernel could not be compiled with xDSL."
            "Otherwise, removes the kernel from the output and emits a warning "
            "instead."));
    binder.opt<unsigned>(
        "iree-quidditch-l1-memory-bytes", l1MemoryBytes,
        llvm::cl::cat(category),
        llvm::cl::desc("Size of the useable L1 memory in bytes"));
  }
};

class QuidditchTargetBackend final : public IREE::HAL::TargetBackend {
public:
  explicit QuidditchTargetBackend(QuidditchTargetOptions options)
      : targetOptions(std::move(options)) {}

  [[nodiscard]] std::string getLegacyDefaultDeviceID() const override {
    return "quidditch_device";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    registry.insert<arm_neon::ArmNeonDialect, arm_sme::ArmSMEDialect,
                    quidditch::Snitch::QuidditchSnitchDialect>();
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {

    NamedAttrList list;
    // Target attribute info used by the LLVM lowering.
    // TODO: Ideally shouldn't be hardcoded.
    list.append("data_layout",
                StringAttr::get(context, "e-m:e-p:32:32-i64:64-n32-S128"));
    list.append("target_triple",
                StringAttr::get(context, "riscv32-unknown-elf"));
    list.append("compute_cores",
                IntegerAttr::get(IntegerType::get(context, 32), 8));
    executableTargetAttrs.push_back(IREE::HAL::ExecutableTargetAttr::get(
        context, StringAttr::get(context, "quidditch"),
        StringAttr::get(context, "static"), list.getDictionary(context)));
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) override {
    OpPassManager &modulePassManager = passManager.nest<ModuleOp>();
    {
      FunctionLikeNest funcPassManager(modulePassManager);
      addCommonTargetExecutablePreprocessingPasses(
          funcPassManager,
          /*useDecomposeSoftmaxFusion=*/false);
    }
    modulePassManager.addPass(createMaterializeUserConfigsPass());
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(quidditch::createConfigureForSnitchPass);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    OpPassManager &modulePassManager = passManager.nest<ModuleOp>();

    FunctionLikeNest(modulePassManager)
        .addPass([] { return createTileAndDistributeToWorkgroupsPass(); })
        .addPass([] { return createConvertToDestinationPassingStylePass(); })
        .addPass(createFoldAffineMinInDistributedLoopsPass)
        .addPass(createRemoveSingleIterationLoopPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(createFuseTensorPadWithConsumerPass)
        .addPass(createConcretizePadResultShapePass)
        .addPass([] {
          return quidditch::createTensorTilePass(
              {quidditch::TilingLevel::Reduction});
        })
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(quidditch::Snitch::createPromoteOperandsToL1Pass)
        // TODO: Fuse scf.forall after.
        .addPass([] {
          return quidditch::createTensorTilePass(
              {quidditch::TilingLevel::Thread});
        });

    BufferizationOptions::AllocationFn allocationFn =
        [](OpBuilder &builder, Location loc, MemRefType memRefType,
           ValueRange dynamicSizes, unsigned alignment) -> Value {
      return builder.create<memref::AllocaOp>(
          loc, memRefType, dynamicSizes, builder.getI64IntegerAttr(alignment));
    };
    BufferizationOptions::MemCpyFn memcpyFn = [](OpBuilder &builder,
                                                 Location loc, Value from,
                                                 Value to) {
      Value token =
          builder.create<quidditch::Snitch::StartDMATransferOp>(loc, from, to);
      builder.create<quidditch::Snitch::WaitForDMATransfersOp>(loc, token);
      return success();
    };

    FunctionLikeNest(modulePassManager)
        .addPass(createEliminateEmptyTensorsPass)
        .addPass(bufferization::createEmptyTensorToAllocTensorPass)
        .addPass(quidditch::Snitch::createPromoteAllocsToL1Pass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass([&] {
          return createIREEComprehensiveBufferizePass(allocationFn, memcpyFn);
        });
    addIREEPostBufferizationPasses(modulePassManager.nest<func::FuncOp>());

    FunctionLikeNest(modulePassManager)
        .addPass(createForallToForLoopPass)
        // TODO: Remove the following pass and plumb support for
        // #hal.descriptor_type memory space through the stack.
        .addPass(createEraseHALDescriptorTypeFromMemRefPass)
        .addPass([&] {
          return quidditch::Snitch::createLowerL1AllocationsPass(
              {targetOptions.l1MemoryBytes, targetOptions.assertCompiled});
        })
        .addPass(quidditch::createReluToMaxPass)
        .addPass(createCanonicalizerPass)
        .addPass(createLinalgGeneralizeNamedOpsPass)
        .addPass(createRemoveSingleIterationLoopPass)
        .addPass(quidditch::Snitch::createFormMicrokernelsPass);

    modulePassManager.addPass(quidditch::Snitch::createSpecializeDMACodePass());
    FunctionLikeNest(modulePassManager)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);
    modulePassManager.addPass(quidditch::createConvertToRISCVPass(
        {targetOptions.xDSLOptPath, targetOptions.assertCompiled}));

    FunctionLikeNest(modulePassManager)
        .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
        .addPass(createMemrefCopyToLinalgPass)
        .addPass(createConvertLinalgToLoopsPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    addConstantBufferizePasses(modulePassManager);

    FunctionLikeNest(modulePassManager)
        .addPass(createFoldTensorExtractOpPass)
        // Handle complex operation conversion.
        .addPass(createConvertComplexToStandardPass)
        // math dialect elementary functions -> polynomial form.
        .addPass(createPolynomialApproximationPass)
        .addPass(createHoistStaticallyBoundAllocationsPass)
        .addPass(createIREEExpandStridedMetadataPass)
        .addPass(createCleanupBufferAllocViewPass);

    FunctionLikeNest(modulePassManager)
        .addPass(arith::createArithExpandOpsPass)
        .addPass(memref::createExpandOpsPass)
        .addPass(memref::createFoldMemRefAliasOpsPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    modulePassManager.addPass(quidditch::createConvertSnitchToLLVMPass());
    modulePassManager.addPass(createConvertToLLVMPass({}));
    modulePassManager.addPass(createReconcileUnrealizedCastsPass());
    // We rely on MLIR symbol visibility being correct after this point and
    // need to mirror the LLVM linkage that was assigned during conversion.
    modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
    modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(
        createAddFastMathFlagsPass());
    passManager.addPass(quidditch::createDisableQuidditchVariantPass());
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(quidditch::createLinkExecutablesPass());
    // Cleanup IR duplication.
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        mlir::createCanonicalizerPass());

    // Assign final executable constant and import ordinals.
    auto &variantPassManager = passManager.nest<IREE::HAL::ExecutableOp>()
                                   .nest<IREE::HAL::ExecutableVariantOp>();
    variantPassManager.addPass(createLLVMCPUAssignConstantOrdinalsPass());
    variantPassManager.addPass(createLLVMCPUAssignImportOrdinalsPass());
  }

  FailureOr<SmallVector<IREE::HAL::Artifact>>
  assembleXDSLOutput(ModuleOp module) {

    auto *dialect =
        module.getContext()->getLoadedDialect<QuidditchSnitchDialect>();

    SmallVector<IREE::HAL::Artifact> objectFiles;
    for (auto func :
         llvm::make_early_inc_range(module.getOps<LLVM::LLVMFuncOp>())) {
      auto assembly = dialect->getRiscvAssemblyAttrHelper().getAttr(func);
      if (!assembly)
        continue;

      SmallString<64> stdinFile;
      int stdinFd;
      if (llvm::sys::fs::createTemporaryFile("xdsl-in", "S", stdinFd,
                                             stdinFile)) {
        return failure();
      }
      llvm::FileRemover stdinFileRemove(stdinFile);
      {
        llvm::raw_fd_ostream ss(stdinFd, /*shouldClose=*/true);
        ss << assembly.getValue();
      }

      auto &objectFile = objectFiles.emplace_back(
          IREE::HAL::Artifact::createTemporary("xdsl-out", "o"));
      int ret = llvm::sys::ExecuteAndWait(
          targetOptions.toolChainRoot + "/bin/pulp-as",
          {targetOptions.toolChainRoot + "/bin/pulp-as", "--filetype=obj",
           "--target-abi=ilp32d", stdinFile.str(), "-o", objectFile.path,
           "--mcpu=snitch", "-g"});
      if (ret != 0)
        return failure();
    }
    return objectFiles;
  }

  static std::unique_ptr<llvm::Module>
  toLLVMModule(llvm::LLVMContext &context, ModuleOp module,
               const llvm::TargetMachine &machine,
               IREE::HAL::ExecutableVariantOp variantOp) {
    module->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        StringAttr::get(module.getContext(), machine.getTargetTriple().str()));

    std::string libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule =
        mlir::translateModuleToLLVMIR(module, context, libraryName);
    if (!llvmModule) {
      module.emitError() << "failed to translate the MLIR LLVM "
                            "dialect to the native llvm::Module";
      return nullptr;
    }

    auto *dialect =
        module.getContext()->getLoadedDialect<QuidditchSnitchDialect>();

    SymbolTable symbolTable(module);
    Quidditch::LibraryBuilder libraryBuilder(
        llvmModule.get(), Quidditch::LibraryBuilder::Mode::NONE,
        Quidditch::LibraryBuilder::Version::LATEST);
    auto align16 = llvm::Attribute::getWithAlignment(context, llvm::Align(16));
    for (auto exportOp :
         variantOp.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
      // Find the matching function in the LLVM module.
      auto *llvmFunc = llvmModule->getFunction((exportOp.getName()).str());
      if (!llvmFunc)
        continue;

      llvm::Function *dmaPointer = nullptr;
      if (Operation *mlirFunc = symbolTable.lookup(exportOp.getName())) {
        if (FlatSymbolRefAttr dmaFunc =
                dialect->getDmaSpecializationAttrHelper().getAttr(mlirFunc)) {
          dmaPointer = llvmModule->getFunction(dmaFunc.getValue());
          if (!dmaPointer) {
            module.emitError()
                << "failed to find DMA code for " << exportOp.getName();
            return nullptr;
          }
          dmaPointer->setLinkage(
              llvm::GlobalValue::LinkageTypes::InternalLinkage);
          // Name suffix recognized by tooling for xDSL generated kernels.
          llvmFunc->setName(llvmFunc->getName() + "$iree_to_xdsl");
        }
      }

      llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      llvmFunc->setDSOLocal(true);

      // Tag the function parameters in case they got removed during conversion.
      // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
      for (unsigned i = 0; i <= 2; ++i) {
        llvmFunc->addParamAttr(i, llvm::Attribute::NonNull);
        llvmFunc->addParamAttr(i, llvm::Attribute::NoAlias);
        llvmFunc->addParamAttr(i, align16);
      }

      // Optionally entry points may specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      int64_t localMemorySize = exportOp.getWorkgroupLocalMemory()
                                    .value_or(APInt(64, 0))
                                    .getSExtValue();

      Quidditch::LibraryBuilder::SourceLocation sourceLocation;
      SmallVector<Quidditch::LibraryBuilder::SourceLocation> stageLocations;
      libraryBuilder.addExport(
          exportOp.getName(), std::move(sourceLocation),
          std::move(stageLocations), /*tag=*/"",
          Quidditch::LibraryBuilder::DispatchAttrs{localMemorySize}, llvmFunc,
          dmaPointer);
    }
    auto *queryLibraryFunc =
        libraryBuilder.build("quidditch_" + libraryName + "_library_query");

    // The query function must be exported for dynamic libraries.
    queryLibraryFunc->setDSOLocal(false);
    queryLibraryFunc->setVisibility(
        llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    queryLibraryFunc->setLinkage(
        llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    queryLibraryFunc->setDLLStorageClass(
        llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

    // Specialize the module to our target machine.
    llvmModule->setDataLayout(machine.createDataLayout());
    llvmModule->setTargetTriple(machine.getTargetTriple().str());
    return llvmModule;
  }

  static void optimizeLLVMModule(llvm::Module &module,
                                 llvm::TargetMachine &machine) {

    llvm::LoopAnalysisManager loopAnalysisManager;
    llvm::FunctionAnalysisManager functionAnalysisManager;
    llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
    llvm::ModuleAnalysisManager moduleAnalysisManager;

    llvm::PassBuilder passBuilder(&machine);
    llvm::AAManager aa = passBuilder.buildDefaultAAPipeline();
    functionAnalysisManager.registerPass([&] { return std::move(aa); });

    passBuilder.registerModuleAnalyses(moduleAnalysisManager);
    passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
    passBuilder.registerFunctionAnalyses(functionAnalysisManager);
    passBuilder.registerLoopAnalyses(loopAnalysisManager);
    passBuilder.crossRegisterProxies(
        loopAnalysisManager, functionAnalysisManager, cGSCCAnalysisManager,
        moduleAnalysisManager);

    llvm::ModulePassManager modulePassManager;
    modulePassManager =
        passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    modulePassManager.run(module, moduleAnalysisManager);
  }

  static FailureOr<IREE::HAL::Artifact>
  compileLLVMModule(llvm::Module &module, llvm::TargetMachine &machine) {
    auto objectFile = IREE::HAL::Artifact::createTemporary("iree-out", "o");

    llvm::raw_fd_ostream &os = objectFile.outputFile->os();
    llvm::legacy::PassManager passManager;
    passManager.add(
        new llvm::TargetLibraryInfoWrapperPass(machine.getTargetTriple()));
    if (machine.addPassesToEmitFile(passManager, os,
                                    /*DwoOut=*/nullptr,
                                    llvm::CodeGenFileType::ObjectFile))
      return failure();

    passManager.run(module);
    os.flush();
    os.close();
    return objectFile;
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp module = variantOp.getInnerModule();

    FailureOr<SmallVector<IREE::HAL::Artifact>> objectFilesOrFailure =
        assembleXDSLOutput(module);
    if (failed(objectFilesOrFailure))
      return failure();

    SmallVector<IREE::HAL::Artifact> objectFiles =
        std::move(*objectFilesOrFailure);

    std::string errorMessage;
    auto llvmTarget = llvm::TargetRegistry::lookupTarget(
        "riscv32-unknown-unknown-elf", errorMessage);
    if (!llvmTarget)
      return variantOp.emitError(errorMessage);

    std::unique_ptr<llvm::TargetMachine> machine(
        llvmTarget->createTargetMachine(
            "riscv32-unknown-unknown-elf", "generic-rv32" /* cpu e.g k8 */,
            "+m,+f,+d,+zfh", {}, llvm::Reloc::Model::PIC_, {},
            llvm::CodeGenOptLevel::Aggressive,
            /*JIT=*/false));

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> llvmModule =
        toLLVMModule(context, module, *machine, variantOp);
    if (!llvmModule)
      return failure();

    optimizeLLVMModule(*llvmModule, *machine);

    FailureOr<IREE::HAL::Artifact> objectFileOrFailure =
        compileLLVMModule(*llvmModule, *machine);
    if (failed(objectFileOrFailure))
      return failure();

    objectFiles.push_back(std::move(*objectFileOrFailure));

    SmallVector<StringRef> arguments = {"ld.lld", "-r"};
    llvm::append_range(
        arguments,
        llvm::map_range(objectFiles,
                        [](IREE::HAL::Artifact &artifact) -> StringRef {
                          return artifact.path;
                        }));

    std::string libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();
    auto linkedObject = IREE::HAL::Artifact::createTemporary(libraryName, "o");
    arguments.push_back("-o");
    arguments.push_back(linkedObject.path);
    int ret = llvm::sys::ExecuteAndWait(
        targetOptions.toolChainRoot + "/bin/ld.lld", arguments);
    if (ret != 0)
      return failure();

    if (!IREE::HAL::outputStaticLibrary(
            "quidditch_" + libraryName,
            "quidditch_" + libraryName + "_library_query",
            targetOptions.staticLibraryOutputPath, linkedObject.path))
      return variantOp.emitError() << "static library generation failed";

    // Embed the library name in the executable binary op. This informs the
    // loader which static library to load for the target binary.
    std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                           libraryName.end());
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(), "snitch",
        libraryNameVector);

    return success();
  }

private:
  QuidditchTargetOptions targetOptions;
};

class QuidditchSession final
    : public PluginSession<QuidditchSession, QuidditchTargetOptions,
                           PluginActivationPolicy::DefaultActivated> {

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("quidditch_device",
                []() { return std::make_shared<QuidditchTargetDevice>(); });
  }

  void
  populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) override {
    targets.add("quidditch", [&]() {
      return std::make_shared<QuidditchTargetBackend>(options);
    });
  }
};

} // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(::QuidditchTargetOptions);

extern "C" bool iree_register_compiler_plugin_hal_target_quidditch(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<QuidditchSession>("hal_target_quidditch");
  return true;
}
