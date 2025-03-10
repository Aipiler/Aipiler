#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define INPUT_SIZE 512

using namespace std;
using half = int16_t;

namespace mix {
namespace utils {
template <typename T>
void load_model(const std::vector<std::string> model_paths,
                std::map<std::string, T *> param_and_loc);
}
} // namespace mix

// 将 float 转换为 int16_t 表示的 f16
int16_t float_to_f16(float value) {
  // 提取 float 的符号位、指数位和尾数位
  uint32_t float_bits = *reinterpret_cast<uint32_t *>(&value);
  int sign = (float_bits >> 31) & 1;
  int exponent = ((float_bits >> 23) & 0xFF) - 127;
  int mantissa = float_bits & 0x7FFFFF;

  // 处理特殊情况：零
  if (exponent == -127 && mantissa == 0) {
    return sign << 15;
  }

  // 处理特殊情况：无穷大或 NaN
  if (exponent == 128) {
    if (mantissa == 0) {
      return (sign << 15) | 0x7C00;
    } else {
      return (sign << 15) | 0x7E00;
    }
  }

  // 处理规范化数
  exponent += 15;
  if (exponent >= 31) {
    // 溢出，返回无穷大
    return (sign << 15) | 0x7C00;
  } else if (exponent <= 0) {
    // 下溢，返回零
    return sign << 15;
  }

  // 舍入尾数
  mantissa >>= 13;
  if ((mantissa & 0x1) && ((mantissa & 0x2) || (mantissa & 0x1FF))) {
    mantissa++;
  }

  // 组合符号位、指数位和尾数位
  return (sign << 15) | ((exponent & 0x1F) << 10) | (mantissa & 0x3FF);
}

float f16_to_float(int16_t f16) {
  // 提取符号位
  int sign = (f16 >> 15) & 1;
  // 提取指数位
  int exponent = (f16 >> 10) & 0x1F;
  // 提取尾数位
  int mantissa = f16 & 0x3FF;

  // 处理特殊情况：零
  if (exponent == 0 && mantissa == 0) {
    return sign ? -0.0f : 0.0f;
  }

  // 处理特殊情况：非规范化数
  if (exponent == 0) {
    return (sign ? -1.0f : 1.0f) * mantissa * std::pow(2, -24);
  }

  // 处理特殊情况：无穷大或 NaN
  if (exponent == 0x1F) {
    if (mantissa == 0) {
      return sign ? -std::numeric_limits<float>::infinity()
                  : std::numeric_limits<float>::infinity();
    } else {
      return std::numeric_limits<float>::quiet_NaN();
    }
  }

  // 处理规范化数
  float normalized_mantissa = 1.0f + (mantissa / 1024.0f);
  float power = std::pow(2, exponent - 15);
  return (sign ? -1.0f : 1.0f) * normalized_mantissa * power;
}

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

extern "C" {
void _mlir_ciface_graph0(RankedMemRefType<half, 2> *input,
                         RankedMemRefType<half, 2> *output);
void call_graph0();

RankedMemRefType<half, 2> *linear_weight;
RankedMemRefType<half, 1> *linear_bias;
}

int main() {
  // initialize weight and bias
  half *linear_weight_data = new half[INPUT_SIZE * INPUT_SIZE];
  int64_t linear_weight_shape[2] = {INPUT_SIZE, INPUT_SIZE};

  half *linear_bias_data = new half[INPUT_SIZE];
  int64_t linear_bias_shape[1] = {INPUT_SIZE};

  linear_weight =
      new RankedMemRefType<half, 2>(linear_weight_data, linear_weight_shape);
  linear_bias =
      new RankedMemRefType<half, 1>(linear_bias_data, linear_bias_shape);

  mix::utils::load_model<half>({"./linear_model.bin"},
                               {{"linear.weight", linear_weight_data},
                                {"linear.bias", linear_bias_data}});

  cout << "weight" << endl;
  linear_weight->print();
  cout << endl << "bias" << endl;
  linear_bias->print();
  cout << endl;

  half *input_data = new half[INPUT_SIZE];
  for (int i = 0; i < INPUT_SIZE; i++) {
    input_data[i] = float_to_f16(1);
  }

  int64_t input_size[2] = {1, INPUT_SIZE};
  RankedMemRefType<half, 2> input(input_data, input_size);

  half *output_data = new half[INPUT_SIZE];
  int64_t output_size[] = {1, INPUT_SIZE};
  RankedMemRefType<half, 2> output(output_data, output_size);

  _mlir_ciface_graph0(&output, &input);
  input.print();
  std::cout << "***********" << std::endl;
  output.print();

  delete[] linear_weight_data;
  delete[] linear_bias_data;
  delete[] output_data;
  return 0;
}