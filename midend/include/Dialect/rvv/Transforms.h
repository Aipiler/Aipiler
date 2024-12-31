#ifndef RVV_TRANSFORMS_H
#define RVV_TRANSFORMS_H

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Collect a set of patterns to lower RVV ops to ops that map to LLVM
/// intrinsics.
void populateRVVLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns,
                                              int64_t RVVIndexBitwidth);

/// Configure the target to support lowering RVV ops to ops that map to LLVM
/// intrinsics.
void configureRVVLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // RVV_TRANSFORMS_H
