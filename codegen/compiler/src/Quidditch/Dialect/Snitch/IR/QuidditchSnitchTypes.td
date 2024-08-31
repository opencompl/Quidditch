#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHTYPES
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHTYPES

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "mlir/IR/AttrTypeBase.td"

class QuidditchSnitch_Type<string name, list<Trait> traits = []> :
  TypeDef<QuidditchSnitch_Dialect, name, traits>;

#endif
