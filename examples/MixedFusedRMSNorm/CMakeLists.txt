add_executable(genMixedFusedRMSNorm
    genMixedFusedRMSNorm.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(genMixedFusedRMSNorm
    PUBLIC
    LLVMSupport
    MLIRToLLVMIRTranslationRegistration
    MLIRFromLLVMIRTranslationRegistration
    LLVMX86Info
    LLVMX86TargetMCA
    LLVMX86Disassembler
    LLVMX86Desc
    LLVMX86CodeGen
    LLVMX86AsmParser
    MLIRTargetCpp
    MLIRFuncAllExtensions
    MLIRSPIRVTranslateRegistration
    MLIRLLVMIRToLLVMTranslation
    ${dialect_libs}
    ${conversion_libs}
    MLIRMIX
    LowerModule
    LowerComposite
    LowerPrimaryToTosa
    loadPytorchModel
)