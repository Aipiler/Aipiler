add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Utils)




# Build static library for MLIR C runner utils runtime.
add_mlir_library(StaticMLIRCRunnerUtils
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/CRunnerUtils.cpp
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/SparseTensorRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  mlir_float16_utils
  MLIRSparseTensorEnums
  MLIRSparseTensorRuntime
  )
set_property(TARGET StaticMLIRCRunnerUtils PROPERTY CXX_STANDARD 17)
target_compile_definitions(StaticMLIRCRunnerUtils PRIVATE StaticMLIRCRunnerUtils_EXPORTS)

# Build static library for MLIR runner utils runtime.
add_mlir_library(StaticMLIRRunnerUtils
  ${MLIR_MAIN_SRC_DIR}/lib/ExecutionEngine/RunnerUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
target_compile_definitions(StaticMLIRRunnerUtils PRIVATE StaticMLIRRunnerUtils_EXPORTS)
