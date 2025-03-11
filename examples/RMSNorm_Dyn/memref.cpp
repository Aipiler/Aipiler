#include "memref.h"
#include <limits>

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