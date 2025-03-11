add_executable(genTelechat_dynamic_load_params
    genTelechat.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(genTelechat_dynamic_load_params
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
    compileUtils
)

add_dependencies(genTelechat_dynamic_load_params loadPytorchModelShared)