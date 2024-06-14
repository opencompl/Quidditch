#ifndef QUIDDITCH_PASSES
#define QUIDDITCH_PASSES

include "mlir/Pass/PassBase.td"

defvar assertCompiledOption =
  Option<"assertCompiled", "assert-compiled", "bool", "false",
         "If true, errors if any kernel could not be compiled with xDSL."
         "Otherwise, removes the kernel from the output and emits a warning "
         "instead.">;

def HoistHALOpsToFuncPass
  : Pass<"quidditch-hoist-hal-ops-to-func", "mlir::ModuleOp"> {
  let description = [{
    Pass performing HAL operation outlining.
    For every function in the module this creates another function called "<old-name>$iree_to_xDSL" and moves all HAL
    operations to that function before calling the original function.

    This allows to pass the original function to xDSL for kernel compilation while compiling the new generated function
    can be lowered by IREE.
    This pass can be removed if xDSL were to learn how to lower IREE HAL operations to RISC-V.

    The original function is additionally tagged with a "xdsl_generated" unit attribute.
  }];

  let options = [assertCompiledOption];
}

def FilterForxDSLPass
  : Pass<"quidditch-filter-for-xdsl"> {
  let description = [{
    Pass rewriting operations to workaround xDSL operations.

    Right now these are:
    * Remove `memref.assume_alignment`.
  }];
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

#endif
