#include "container.h"
#include <cstdint>

extern "C" {
intptr_t size[2] = {2, 2};

float weight0[2 * 2] = {1, 2, 3, 4};
// MemRef<float, 2> weight0(weight0_date, size);

float weight1[2 * 2] = {1, 2, 3, 4};
// MemRef<float, 2> weight1(weight1_date, size);

float weight2[2 * 2] = {1, 2, 3, 4};
// MemRef<float, 2> weight2(weight2_date, size);

float bias2[2 * 2] = {1, 2, 3, 4};
// MemRef<float, 2> bias2(bias2_date, size);

extern void graph0(void *alloca0, void *aligned0, intptr_t offset0,
                   intptr_t size0_0, intptr_t size0_1, intptr_t stride0_0,
                   intptr_t stride0_1, void *alloca1, void *aligned1,
                   intptr_t offset1, intptr_t size1_0, intptr_t size1_1,
                   intptr_t stride1_0, intptr_t stride1_1);
}

void callGraph0(MemRef<float, 2> &arg0, MemRef<float, 2> &arg1) {
  graph0(arg0.allocated, arg0.aligned, arg0.offset, arg0.sizes[0],
         arg0.sizes[1], arg0.strides[0], arg0.strides[1], arg1.allocated,
         arg1.aligned, arg1.offset, arg1.sizes[0], arg1.sizes[1],
         arg1.strides[0], arg1.strides[1]);
}

int main() {
  float data1[4] = {2, 3, 4, 5};
  float data2[4] = {4, 5, 6, 7};
  MemRef<float, 2> arg0(data1, size);
  MemRef<float, 2> arg1(data2, size);
  callGraph0(arg0, arg1);
  return 0;
}