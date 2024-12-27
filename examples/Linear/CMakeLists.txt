find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)


add_executable(linear
linear.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(linear
    PUBLIC
    pybind11::embed 
    Python3::Python
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

target_include_directories(linear PRIVATE ${Python3_INCLUDE_DIRS})
