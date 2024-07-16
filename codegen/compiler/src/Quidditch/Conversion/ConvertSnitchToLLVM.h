
#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace quidditch {
void populateSnitchToLLVMConversionPatterns(mlir::ModuleOp moduleOp,
                                            mlir::LLVMTypeConverter &converter,
                                            mlir::RewritePatternSet &patterns);
}
