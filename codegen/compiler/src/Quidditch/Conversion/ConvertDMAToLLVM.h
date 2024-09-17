
#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace quidditch {
void populateDMAToLLVMConversionPatterns(mlir::ModuleOp moduleOp,
                                         mlir::LLVMTypeConverter &converter,
                                         mlir::RewritePatternSet &patterns);
}
