#ifndef MEMREF_H
#define MEMREF_H

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

using half = int16_t;

namespace mix {
namespace utils {
void load_model_f16(const std::vector<std::string> &model_paths,
                    std::map<std::string, int16_t *> &param_and_loc);
}
} // namespace mix

// 将 float 转换为 int16_t 表示的 f16
int16_t float_to_f16(float value);

float f16_to_float(int16_t f16);

template <typename T, int N> struct RankedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
  RankedMemRefType(T *data, int64_t sizes[N]) {
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
  void print() {
    using namespace std;
    cout << endl << "Data content:" << endl;
    for (int i = 0;
         i < ((strides[0] * sizes[0]) < 10 ? (strides[0] * sizes[0]) : 10);
         i += 1) {
      for (int j = 0; j < N - 1; j++) {
        if (i != 0 && i % this->strides[j] == 0)
          cout << "\n";
      }
      cout << f16_to_float(data[i]) << " ";
    }
    cout << endl;
  }
};

#endif