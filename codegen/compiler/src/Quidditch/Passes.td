#ifndef QUIDDITCH_PASSES
#define QUIDDITCH_PASSES

include "mlir/Pass/PassBase.td"

defvar assertCompiledOption =
  Option<"assertCompiled", "assert-compiled", "bool", "false",
         "If true, errors if any kernel could not be compiled with xDSL."
         "Otherwise, removes the kernel from the output and emits a warning "
         "instead.">;

def OutlineLinalgOpsToxDSLPass
  : Pass<"quidditch-outline-linalg-ops-to-xdsl", "mlir::ModuleOp"> {
  let description = [{
    Outlines a series of linalg operation in a function into their own
    functions with a suitable calling convention for xDSL.
    The generated functions can be identified via `xdsl_generated` attribute.

    The original function that the outlining was performed on will have a
    `xdsl_kernels` which is an array of symbol references to the outlined
    functions.
  }];

  let options = [assertCompiledOption];
}

def ConvertToRISCVPass : Pass<"quidditch-convert-to-riscv", "mlir::func::FuncOp"> {
  let options = [
    Option<"xDSLOptPath", "xdsl-opt-path", "std::string", [{""}],
      "Path to the 'xdsl-opt' executable to use for kernel compilation.">,
    assertCompiledOption,
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
