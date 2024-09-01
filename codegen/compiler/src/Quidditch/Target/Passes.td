#ifndef QUIDDITCH_TARGET_PASSES
#define QUIDDITCH_TARGET_PASSES

include "mlir/Pass/PassBase.td"

def LinkExecutablesPass : Pass<"quidditch-link-executables", "mlir::ModuleOp"> {
  let description = [{
    Combines all `hal.executable.variant`s of the same target into a single
    `hal.executable.variant` nested within one `hal.executable`.

    Additionally performs replacement of any kernels that xDSL failed to
    compile with LLVM implementations.
  }];
}

def DisableQuidditchVariantPass : Pass<"quidditch-disable-variant",
  "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let description = [{
    Disables the `hal.executable.variant` for Quidditch if `xDSL` failed
    compilation. This is done by adding a `hal.executable.condition` returning
    false.
  }];
}

def ReluToMaxPass : Pass<"quidditch-relu-to-max">;
def PadToTilingConfigPass : Pass<"quidditch-pad-to-tiling-config">;

def ConfigureForSnitchPass
  : InterfacePass<"quidditch-configure-for-snitch",
      "mlir::FunctionOpInterface">;

def TensorTilePass : InterfacePass<"quidditch-tensor-tile",
  "mlir::FunctionOpInterface"> {
  let options = [
    Option<"tilingLevel", "tiling-level", "quidditch::TilingLevel",
           /*default=*/"quidditch::TilingLevel::L1",
           "Tiling level to tile. Supported levels are 'reduction' and 'thread'",
           [{llvm::cl::values(
              clEnumValN(quidditch::TilingLevel::L1, "l1",
                         "Tile and fuse all annotated ops to fit into L1 memory"),
              clEnumValN(quidditch::TilingLevel::Thread, "thread",
                         "Tile and fuse all annotated ops to threads")
           )}]>,
  ];
}

def RemoveTrivialLoopsPass
  : InterfacePass<"quidditch-remove-trivial-loops",
      "mlir::FunctionOpInterface"> {

}

def ConvertToLLVMPass : Pass<"quidditch-convert-to-llvm", "mlir::ModuleOp"> {
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::affine::AffineDialect",
    "quidditch::SnitchDMA::SnitchDMADialect",
  ];
}

#endif
