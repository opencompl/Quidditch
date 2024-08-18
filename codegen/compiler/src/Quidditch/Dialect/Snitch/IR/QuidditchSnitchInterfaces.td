#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHINTERFACES
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHINTERFACES

include "mlir/IR/Interfaces.td"

def QuidditchSnitch_CoreSpecializationOpInterface
  : OpInterface<"CoreSpecializationOpInterface"> {
  let cppNamespace = "::quidditch::Snitch";

  let description = [{
    Interface used as a base class for ops meant to only run on a specific core.
    When specializing a function for a specific core, ops implementing this
    interface but not supported on a specific core will be removed using
    `replaceWithNoop`.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method called to replace this operation with a noop in an unsupported
        specialization. `rewriter`s insertion point is set right before the
        operation.

        The op must have been erased when this method returns.
      }],
      /*retTy=*/"void",
      /*methodName=*/"replaceWithNoop",
      /*args=*/(ins "mlir::RewriterBase&":$rewriter)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if this operation requires synchronization between all cores.
      }],
      "bool", "needsSynchronization", (ins), [{}], [{
        return false;
      }]
    >
  ];
}

def QuidditchSnitch_DMACoreSpecializationOpInterface
  : OpInterface<"DMACoreSpecializationOpInterface", [QuidditchSnitch_CoreSpecializationOpInterface]> {
  let cppNamespace = "::quidditch::Snitch";
}

def QuidditchSnitch_ComputeCoreSpecializationOpInterface
  : OpInterface<"ComputeCoreSpecializationOpInterface", [QuidditchSnitch_CoreSpecializationOpInterface]> {
  let cppNamespace = "::quidditch::Snitch";
}

#endif
