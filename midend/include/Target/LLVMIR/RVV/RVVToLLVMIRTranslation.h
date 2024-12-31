//=======- RVVToLLVMIRTranslation.h - RVV to LLVM IR ------------*- C++ -*-===//
//
// This provides registration calls for RVV dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LLVMIR_DIALECT_RVV_RVVTOLLVMIRTRANSLATION_H
#define TARGET_LLVMIR_DIALECT_RVV_RVVTOLLVMIRTRANSLATION_H

/// Register the RVV dialect and the translation from it to the LLVM IR in
/// the given registry.
void registerRVVDialectTranslation(mlir::DialectRegistry &registry);

/// Register the RVV dialect and the translation from it in the registry
/// associated with the given context.
void registerRVVDialectTranslation(mlir::MLIRContext &context);


#endif // TARGET_LLVMIR_DIALECT_RVV_RVVTOLLVMIRTRANSLATION_H
