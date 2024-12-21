llvm_map_components_to_libnames(llvm_libs Support)
add_library(readjson
    STATIC
    readjson.cpp
)

target_link_libraries(readjson PRIVATE ${llvm_libs})