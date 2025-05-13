set(LLVM_TARGET_DEFINITIONS einsumDialect.td)
mlir_tablegen(einsumDialect.h.inc -gen-dialect-decls)
mlir_tablegen(einsumDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(einsumDialectIncGen)

set(LLVM_TARGET_DEFINITIONS einsumEnums.td)
mlir_tablegen(einsumEnums.h.inc -gen-enum-decls)
mlir_tablegen(einsumEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(einsumEnumsIncGen)
add_dependencies(mlir-headers einsumEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS einsumOps.td)
mlir_tablegen(einsumOps.h.inc -gen-op-decls)
mlir_tablegen(einsumOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(einsumOpsIncGen)

set(LLVM_TARGET_DEFINITIONS  einsumDialect.td)
mlir_tablegen(einsumOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(einsumOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(einsumOpsAttributesIncGen)
add_dependencies(mlir-headers einsumOpsAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS einsumTypes.td)
mlir_tablegen(einsumTypes.h.inc -gen-typedef-decls -typedefs-dialect=einsum)
mlir_tablegen(einsumTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=einsum)
add_public_tablegen_target(einsumTypesIncGen)


