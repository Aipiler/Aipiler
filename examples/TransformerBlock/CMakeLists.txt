add_executable(genBlock
    genBlock.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(genBlock
    PUBLIC
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRSupport
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRSCFToControlFlow
    MLIRAffineToStandard
    ${dialect_libs}
    ${conversion_libs}
    MLIRMIX
    LowerModule
    LowerComposite
    LowerPrimaryToTosa
)