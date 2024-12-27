add_mlir_library(LowerComposite
    LowerComposite.cpp
)

target_link_libraries(LowerComposite PRIVATE readjson)