#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHDIALECT
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHDIALECT

include "mlir/IR/DialectBase.td"

def QuidditchSnitch_Dialect : Dialect {
  let name = "quidditch_snitch";
  let cppNamespace = "::quidditch::Snitch";

  let discardableAttrs = (ins
    "mlir::StringAttr":$riscv_assembly,
    "mlir::UnitAttr":$xdsl_compilation_failed
  );

  let useDefaultAttributePrinterParser = 1;
}

#endif
