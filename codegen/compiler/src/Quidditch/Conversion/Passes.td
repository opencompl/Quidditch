#ifndef QUIDDITCH_CONVERSION_PASSES
#define QUIDDITCH_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertToRISCVPass : Pass<"quidditch-convert-to-riscv", "mlir::ModuleOp"> {
  let options = [
    Option<"xDSLOptPath", "xdsl-opt-path", "std::string", [{""}],
      "Path to the 'xdsl-opt' executable to use for kernel compilation.">,
    Option<"assertCompiled", "assert-compiled", "bool", "false",
           "If true, errors if any kernel could not be compiled with xDSL."
           "Otherwise, removes the kernel from the output and emits a warning "
           "instead.">,
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
}

def ConvertSnitchToLLVMPass
    : Pass<"quidditch-convert-snitch-to-llvm", "mlir::ModuleOp"> {
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
  ];
}

#endif
