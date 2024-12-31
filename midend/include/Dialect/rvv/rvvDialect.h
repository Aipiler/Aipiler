#ifndef RVV_RVVDIALECT_H
#define RVV_RVVDIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "rvv/rvvDialect.h.inc"

#define GET_OP_CLASSES
#include "rvv/rvvOps.h.inc"

#endif // RVV_RVVDIALECT_H