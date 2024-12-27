#ifndef MIX_TYPES_H
#define MIX_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include <optional>


#define GET_TYPEDEF_CLASSES
#include "mix/mixTypes.h.inc"

#endif