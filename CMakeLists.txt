##===- CMakeLists.txt - Aipiler cmake root -----------------*- cmake -*-===//
##
## Configure the Aipiler build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(mix-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)
option(BUDDY_MLIR_OUT_OF_TREE_BUILD "Specifies an out of tree build" OFF)

find_package(MLIR REQUIRED CONFIG)
find_package(LLD REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if(NOT DEFINED LLVM_MAIN_SRC_DIR)
  get_filename_component(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llvm/llvm ABSOLUTE)
endif()
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(LLD_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../lld)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLD_CMAKE_DIR}")

find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
        NO_DEFAULT_PATH)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)




include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLD_INCLUDE_DIRS})


#-------------------------------------------------------------------------------
# Aipiler configuration
#-------------------------------------------------------------------------------

set(Aipiler_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(Aipiler_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(Aipiler_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(Aipiler_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(Aipiler_EXAMPLES_DIR ${Aipiler_SOURCE_DIR}/examples)
set(Aipiler_PYTHON_PACKAGES_DIR ${BUDDY_BUILD_DIR}/python)

set(EXECUTABLE_OUTPUT_PATH ${Aipiler_BINARY_DIR})
set(LIBRARY_OUTPUT_DIRECTORY ${Aipiler_LIBRARY_DIR})

# Add Aipiler files to the include path
include_directories(${PROJECT_SOURCE_DIR}/midend/include)
include_directories(${PROJECT_SOURCE_DIR}/midend/include/Dialect)
include_directories(${PROJECT_BINARY_DIR}/midend/include)
include_directories(${PROJECT_BINARY_DIR}/midend/include/Dialect)



#-------------------------------------------------------------------------------
# Initialize Python packages
#-------------------------------------------------------------------------------

message(STATUS "Using Python3Config.cmake in: ${Python3_DIR}")
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
add_subdirectory(thirdparty/pybind11)



# if(DEFINED ENV{CONDA_PREFIX})
#    set(Python3_ROOT_DIR $ENV{CONDA_PREFIX})
# endif()


# 设置 Python 模块路径
set(PYTHON_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python)

# Create directories for the BUDDY-MLIR Python packages
# file(MAKE_DIRECTORY ${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy)
# file(MAKE_DIRECTORY ${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler)
# # Create empty __init__.py files to make these directories Python packages
# file(WRITE ${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/__init__.py "")
# file(WRITE ${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy/compiler/__init__.py "")

# install(DIRECTORY ${BUDDY_MLIR_PYTHON_PACKAGES_DIR}/buddy DESTINATION python_packages)


add_subdirectory(midend)
add_subdirectory(tools)
add_subdirectory(examples)
# add_subdirectory(runtime)