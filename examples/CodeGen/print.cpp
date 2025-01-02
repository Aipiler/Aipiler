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
  int64_t shape[3];
  for (int64_t i = 0; i < 3; i++) {
    shape[i] = in->sizes[i];
    printf("size[%ld] = %ld\n", i, in->sizes[i]);
  }
  printf("[");
  for (int64_t i = 0; i < shape[0]; i++) {
    printf("[");
    for (int64_t j = 0; j < shape[1]; j++) {
      printf("[");
      for (int64_t k = 0; k < shape[2]; k++) {
        printf("%d, ", in->data[k + j * shape[2] + i * shape[2] * shape[1]]);
        if (k > 64) {
          printf("...");
          break;
        }
      }
      printf("], \n");
      if (j > 64) {
        printf("...");
        break;
      }
    }
    printf("], \n");
    if (i > 64) {
      printf("...");
      break;
    }
  }
  printf("]\n");
  return;
}

extern "C" void _mlir_ciface_printpoint() {
  printf("point\n");
  return;
}