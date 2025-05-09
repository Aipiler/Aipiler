add_mlir_dialect_library(MLIREINSUM
        einsumDialect.cpp
        einsumOps.cpp
        einsumTypes.cpp

        DEPENDS
        einsumDialectIncGen
        einsumOpsIncGen
        einsumTypesIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)

