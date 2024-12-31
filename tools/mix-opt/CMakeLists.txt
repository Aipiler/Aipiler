get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  MLIRMIX
  LowerModule
  LowerComposite
  LowerPrimaryToTosa
  OpsOptimization
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRTestTransforms
  MLIRTestTransformDialect
  MLIRTransformUtils
  MLIRTestTransforms
  MLIRTestTransformDialect
  MLIRTransformUtils
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
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRAffineToStandard
  )

add_mlir_tool(mix-opt
  mix-opt.cpp
  
  DEPENDS
  ${LIBS}
  SUPPORT_PLUGINS
)

target_link_libraries(mix-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-opt)

mlir_check_all_link_libraries(mlir-opt)
export_executable_symbols_for_plugins(mlir-opt)