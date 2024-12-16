get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_tool(mix-opt
  mix-opt.cpp
)

target_link_libraries(mix-opt
  PRIVATE
  MLIRMIX
  LowerModule
  LowerComposite
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
