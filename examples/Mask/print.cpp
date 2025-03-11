#include <cstdint>
#include <stdio.h>

template <typename T, int N> struct RankedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" void _mlir_ciface_printMemrefI1(RankedMemRefType<uint8_t, 3> *in) {
  int64_t shape[3];
  for (int64_t i = 0; i < 3; i++) {
    shape[i] = in->sizes[i];
    printf("size[%ld] = %ld\n", i, in->sizes[i]);
  }

  printf("[\n");
  for (int64_t i = 0; i < shape[0]; i++) {
    printf("  [\n");
    for (int64_t j = 0; j < shape[1]; j++) {
      printf("    [");
      for (int64_t k = 0; k < shape[2]; k++) {
        int index = k / 8;      // Byte index
        int bit_offset = k % 8; // Bit index within the byte
        uint8_t byte = in->data[index + j * (shape[2] / 8) +
                                i * (shape[2] * shape[1] / 8)];
        bool bit_value = (byte >> bit_offset) & 1;

        printf("%s", bit_value ? "true" : "false");

        if (k < shape[2] - 1)
          printf(", ");

        if (k > 64) {
          printf("...");
          break;
        }
      }
      printf("],\n");
      if (j > 64) {
        printf("...");
        break;
      }
    }
    printf("  ],\n");
    if (i > 64) {
      printf("...");
      break;
    }
  }
  printf("]\n");
}
