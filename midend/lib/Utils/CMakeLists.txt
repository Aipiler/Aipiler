llvm_map_components_to_libnames(llvm_libs Support)
add_library(readjson
    STATIC
    readjson.cpp
)

target_link_libraries(readjson PRIVATE ${llvm_libs})

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)

add_library(loadPytorchModel
    STATIC
    loadPytorchModel.cpp
)

target_link_libraries(loadPytorchModel PRIVATE 
    ${llvm_libs}     
    pybind11::embed 
    Python3::Python
)

target_include_directories(loadPytorchModel 
    PRIVATE 
    ${Python3_INCLUDE_DIRS}
)
