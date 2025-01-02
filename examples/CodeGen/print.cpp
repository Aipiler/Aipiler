#include <cstdint>
#include <stdio.h>
template <typename T, int N> struct RankedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
extern "C" void _mlir_ciface_printMemrefF16(RankedMemRefType<uint16_t, 3> *in) {
  for (int64_t i = 0; i < 3; i++) {
    printf("size[%ld] = %ld\n", i, in->sizes[i]);
  }
  return;
}

extern "C" void _mlir_ciface_printpoint() {
  printf("point");
  return;
}