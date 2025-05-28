#ifndef PRINT_H
#define PRINT_H

#include <cmath>
#include <cstdint>

template <typename T, int N> struct RankedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];

  explicit RankedMemRefType(T *data, int64_t sizes[N]) {
    this->offset = 0;
    this->data = data;
    this->basePtr = data;
    int64_t stride = 1;
    for (int i = N - 1; i >= 0; i--) {
      auto size = sizes[i];
      this->sizes[i] = size;
      this->strides[i] = stride;
      stride *= size;
    }
  }
};

template <typename T> class DynamicMemRefType {
public:
  int64_t rank;
  T *basePtr;
  T *data;
  int64_t offset;
  const int64_t *sizes;
  const int64_t *strides;

  template <int N>
  explicit DynamicMemRefType(const RankedMemRefType<T, N> &memRef)
      : rank(N), basePtr(memRef.basePtr), data(memRef.data),
        offset(memRef.offset), sizes(memRef.sizes), strides(memRef.strides) {}
};
#endif