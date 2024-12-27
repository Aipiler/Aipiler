
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)

# 添加 C++ 源文件
add_executable(loadWeight main.cpp)

# 链接 pybind11 和 Python 库
target_link_libraries(loadWeight PRIVATE 
    pybind11::embed 
    Python3::Python
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRSupport
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRSCFToControlFlow
    MLIRAffineToStandard
    ${dialect_libs}
    ${conversion_libs})
    
target_include_directories(loadWeight PRIVATE ${Python3_INCLUDE_DIRS})

# 设置 Python 模块路径
# set(PYTHON_MODULE_PATH ${Aipiler_PYTHON_PACKAGES_DIR}/Aipiler/utils)
