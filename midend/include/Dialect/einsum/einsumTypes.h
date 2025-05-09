#ifndef EINSUM_TYPES_H
#define EINSUM_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include <optional>

#define GET_TYPEDEF_CLASSES
#include "einsum/einsumTypes.h.inc"

#endif