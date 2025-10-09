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

def ConfigureTiles : InterfacePass<"quidditch-configure-tiles", "mlir::FunctionOpInterface"> {
  let summary = "Annotate linalg operations with tile sizes";
  let description = [{
    Within each iree dispatch, annotate the root linalg operation with a tiling scheme (tile sizes + loop interchange).
    Caveat: only tiles linalg operations of type matmul_transpose_b (for now)
    Set the importTiles option to the path to the json file containing the tiling scheme for each dispatch.
  }];
  let options = [
    Option<"importTiles", "import-tiles", "std::string", /*default=*/"",
           "Name of a JSON file specifying loop bounds and order for each root linalg operation.">,
    Option<"tablePointer", "NeverPassAValueHere", "std::uintptr_t", /*default=*/"0",
           "Avoids opening the input file multiple times. Never pass a value to this option via the command line.">,
  ];
}

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
