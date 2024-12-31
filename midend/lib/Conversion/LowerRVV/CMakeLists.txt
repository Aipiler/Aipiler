add_mlir_library(LowerRVVPass
  LowerPass.cpp

  LINK_LIBS PUBLIC
  RVVTransforms
  )
