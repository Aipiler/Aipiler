#include "container.h"
#include <cstdint>

extern "C" {
intptr_t size[2] = {2, 2};

float weight3[2] = {1, 2};

extern void graph0(void *alloca0, void *aligned0, intptr_t offset0,
                   intptr_t size0_0, intptr_t size0_1, intptr_t stride0_0,
                   intptr_t stride0_1);
}

void callGraph0(MemRef<float, 2> &arg0) {
  graph0(arg0.allocated, arg0.aligned, arg0.offset, arg0.sizes[0],
         arg0.sizes[1], arg0.strides[0], arg0.strides[1]);
}

int main() {
  float data1[4] = {2, 3, 4, 5};
  MemRef<float, 2> arg0(data1, size);
  callGraph0(arg0);
  return 0;
}