add_executable(genTelechat
    genTelechat.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(genTelechat
    PUBLIC
    LLVMSupport
    MLIRToLLVMIRTranslationRegistration
    MLIRFromLLVMIRTranslationRegistration
    LLVMX86Info
    # LLVMX86TargetMCA
    LLVMX86Disassembler
    LLVMX86Desc
    LLVMX86CodeGen
    LLVMX86AsmParser
    MLIRTargetCpp
    MLIRFuncAllExtensions
    MLIRSPIRVTranslateRegistration
    MLIRLLVMIRToLLVMTranslation
    lldELF
    lldCommon
    ${dialect_libs}
    ${conversion_libs}
    MLIRMIX
    LowerModule
    LowerComposite
    LowerPrimaryToTosa
    loadPytorchModel
    StaticMLIRRunnerUtils
    StaticMLIRCRunnerUtils
    mixlogger
)