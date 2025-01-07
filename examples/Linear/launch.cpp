#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace mix {
namespace utils {
template <typename T>
void load_model(const std::vector<std::string> model_paths,
                std::map<std::string, T *> param_and_loc);
}
} // namespace mix

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
    cout << endl << "Data content:" << endl;
    for (int i = 0; i < strides[0] * sizes[0]; i += 1) {
      for (int j = 0; j < N - 1; j++) {
        if (i != 0 && i % this->strides[j] == 0)
          cout << "\n";
      }
      cout << data[i] << " ";
    }
    cout << endl;
  }
};

extern "C" {
void _mlir_ciface_graph0(RankedMemRefType<float, 2> *input,
                         RankedMemRefType<float, 2> *output);
void call_graph0();
float linear_weight[5 * 10];
float linear_bias[5];
}

int main() {
  // initialize weight and bias
  mix::utils::load_model<float>(
      {"./linear_model.bin"},
      {{"linear.weight", linear_weight}, {"linear.bias", linear_bias}});

  // cout << "weight" << endl;
  // for (int i = 0; i < 50; i++) {
  //   cout << "\t" << linear_weight[i] << " ";
  //   if (i != 0 && i % 5 == 0) {
  //     cout << endl;
  //   }
  // }
  // cout << endl << "bias" << endl;
  // for (int i = 0; i < 5; i++) {
  //   cout << linear_bias[i] << " ";
  // }
  // cout << endl;

  float input_data[20] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  int64_t input_size[2] = {2, 10};
  RankedMemRefType<float, 2> input(input_data, input_size);

  float *output_data = new float[10];
  int64_t output_size[] = {2, 5};
  RankedMemRefType<float, 2> output(output_data, output_size);

  _mlir_ciface_graph0(&output, &input);
  output.print();
  return 0;
}