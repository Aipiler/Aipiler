add_mlir_library(LowerPrimaryToTosa
    LowerPrimaryToTosa.cpp
)

target_link_libraries(LowerPrimaryToTosa PRIVATE readjson)