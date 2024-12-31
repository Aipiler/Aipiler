llvm_map_components_to_libnames(llvm_libs Support)
add_library(readjson
    STATIC
    readjson.cpp
)

target_link_libraries(readjson PRIVATE ${llvm_libs})

add_library(mixlogger
    STATIC
    logger.cpp
)

add_library(loadPytorchModel
    STATIC
    loadPytorchModel.cpp
)

target_link_libraries(loadPytorchModel PRIVATE 
    mixlogger
    ${llvm_libs}     
    pybind11::embed 
    Python3::Python
)

target_include_directories(loadPytorchModel 
    PRIVATE 
    ${Python3_INCLUDE_DIRS}
)
