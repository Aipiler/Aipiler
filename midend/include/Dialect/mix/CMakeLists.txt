set(LLVM_TARGET_DEFINITIONS mixDialect.td)
mlir_tablegen(mixDialect.h.inc -gen-dialect-decls)
mlir_tablegen(mixDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(mixDialectIncGen)

set(LLVM_TARGET_DEFINITIONS mixOps.td)
mlir_tablegen(mixOps.h.inc -gen-op-decls)
mlir_tablegen(mixOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(mixOpsIncGen)

set(LLVM_TARGET_DEFINITIONS mixTypes.td)
mlir_tablegen(mixTypes.h.inc -gen-typedef-decls -typedefs-dialect=mix)
mlir_tablegen(mixTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=mix)
add_public_tablegen_target(mixTypesIncGen)
