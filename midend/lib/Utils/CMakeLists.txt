llvm_map_components_to_libnames(llvm_libs Support)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


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

add_library(loadPytorchModelShared
    SHARED
    loadPytorchModel.cpp
)

target_link_libraries(loadPytorchModelShared PRIVATE 
    mixlogger
    ${llvm_libs}   
    ${dialect_libs}
    ${conversion_libs}
    pybind11::embed 
    Python3::Python
)

target_include_directories(loadPytorchModelShared 
    PRIVATE 
    ${Python3_INCLUDE_DIRS}
)

add_library(compileUtils
    STATIC
    compileUtils.cpp
)


target_link_libraries(compileUtils
    PRIVATE
    LLVMSupport
    MLIRToLLVMIRTranslationRegistration
    MLIRFromLLVMIRTranslationRegistration
    LLVMX86Info
    # LLVMX86TargetMCA
    LLVMX86Disassembler
    LLVMX86Desc
    LLVMX86CodeGen
    LLVMX86AsmParser
    MLIRTargetCpp
    MLIRFuncAllExtensions
    MLIRSPIRVTranslateRegistration
    MLIRLLVMIRToLLVMTranslation
    lldELF
    lldCommon
    ${dialect_libs}
    ${conversion_libs}
    mixlogger
)