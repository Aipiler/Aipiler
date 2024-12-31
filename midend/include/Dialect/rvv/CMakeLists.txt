set(LLVM_TARGET_DEFINITIONS rvvDialect.td)
mlir_tablegen(rvvDialect.h.inc -gen-dialect-decls -dialect=rvv)
mlir_tablegen(rvvDialect.cpp.inc -gen-dialect-defs -dialect=rvv)
add_public_tablegen_target(RVVDialectIncGen)

set(LLVM_TARGET_DEFINITIONS rvvOps.td)
mlir_tablegen(rvvOps.h.inc -gen-op-decls -dialect=rvv)
mlir_tablegen(rvvOps.cpp.inc -gen-op-defs -dialect=rvv)
add_public_tablegen_target(RVVOpsIncGen)

set(LLVM_TARGET_DEFINITIONS rvvTypes.td)
mlir_tablegen(rvvTypes.h.inc -gen-typedef-decls -typedefs-dialect=rvv)
mlir_tablegen(rvvTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=rvv)
add_public_tablegen_target(RVVTypesIncGen)

set(LLVM_TARGET_DEFINITIONS rvvOps.td)
mlir_tablegen(rvvConversions.inc -gen-llvmir-conversions -dialect=rvv)
add_public_tablegen_target(RVVConversionsIncGen)