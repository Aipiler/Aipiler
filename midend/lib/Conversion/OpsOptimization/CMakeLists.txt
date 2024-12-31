add_mlir_library(OpsOptimization
  BatchMatmulVec.cpp
  Conv2dNhwcFhwcVec.cpp
  PoolingNhwcMaxVec.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  
)
