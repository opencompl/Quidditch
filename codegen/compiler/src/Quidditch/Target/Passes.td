#ifndef QUIDDITCH_TARGET_PASSES
#define QUIDDITCH_TARGET_PASSES

include "mlir/Pass/PassBase.td"

def FormMicrokernelsPass
  : InterfacePass<"quidditch-form-microkernels", "mlir::FunctionOpInterface"> {
  let dependentDialects = [
    "quidditch::Snitch::QuidditchSnitchDialect",
  ];
}

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

#endif
