add_mlir_dialect_library(RVVDialect
  rvvDialect.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}/include/Dialect/RISCVV
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  )

add_mlir_dialect_library(RVVTransforms
  LegalizeForLLVMExport.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  RVVConversionsIncGen

  LINK_LIBS PUBLIC
  RVVDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )