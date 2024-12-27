add_mlir_dialect_library(MLIRMIX
        mixDialect.cpp
        mixOps.cpp
        mixTypes.cpp

        DEPENDS
        mixDialectIncGen
        mixOpsIncGen
        mixTypesIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)

