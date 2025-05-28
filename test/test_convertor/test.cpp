#include "memref.h"
#include <cstdint>
#include <cstdio>
#include <iostream>

extern "C" void printMemrefI32(DynamicMemRefType<int32_t> *dy) {
  std::cout << "rank=" << dy->rank << std::endl;
  // print size
  int64_t size = 0;
  std::cout << "size = [";
  for (int i = 0; i < dy->rank; i++) {
    std::cout << dy->sizes[i];
    if (i != dy->rank - 1) {
      std::cout << ", ";
    }
    size += dy->sizes[i];
  }
  std::cout << "]\n";
  // print data
  std::cout << "data = {";
  for (int64_t i = 0; i < size; i++) {
    std::cout << dy->data[i];
    if (i != size - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "}\n";
  std::cout << "data storage = ";
  printf("%p\n", dy->data);
  return;
}

extern "C" void printMemrefF32(DynamicMemRefType<float> *dy) {
  std::cout << "rank=" << dy->rank << std::endl;
  // print size
  int64_t size = 0;
  std::cout << "size = [";
  for (int i = 0; i < dy->rank; i++) {
    std::cout << dy->sizes[i];
    if (i != dy->rank - 1) {
      std::cout << ", ";
    }
    size += dy->sizes[i];
  }
  std::cout << "]\n";
  // print data
  std::cout << "data = {";
  for (int64_t i = 0; i < size; i++) {
    std::cout << dy->data[i];
    if (i != size - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "}\n";
  std::cout << "data storage = ";
  printf("%p\n\n", dy->data);
  return;
}

extern "C" void matmul(DynamicMemRefType<float> *A, DynamicMemRefType<float> *B,
                       DynamicMemRefType<float> *C) {
  // 参数校验
  if (!A || !B || !C || A->rank != 2 || B->rank != 2 || C->rank != 2 ||
      A->sizes[1] != B->sizes[0] || // K维度必须一致
      A->sizes[0] != C->sizes[0] || // M维度必须一致
      B->sizes[1] != C->sizes[1]) { // N维度必须一致
    return;
  }

  // 获取矩阵维度
  const int64_t M = A->sizes[0];
  const int64_t K = A->sizes[1];
  const int64_t N = B->sizes[1];

// 内存访问宏（处理任意stride）
#define A_ELEM(i, k) *(A->data + (i) * A->strides[0] + (k) * A->strides[1])
#define B_ELEM(k, j) *(B->data + (k) * B->strides[0] + (j) * B->strides[1])
#define C_ELEM(i, j) *(C->data + (i) * C->strides[0] + (j) * C->strides[1])

  // 矩阵乘法核心
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float a = A_ELEM(i, k);
        float b = B_ELEM(k, j);
        sum += a * b;
      }
      C_ELEM(i, j) = sum;
    }
  }

#undef A_ELEM
#undef B_ELEM
#undef C_ELEM
}

// int main() {
//   int32_t a_data[] = {1, 2, 3, 4};
//   int64_t a_sizes[] = {2, 2};
//   RankedMemRefType<int32_t, 2> raw_A(a_data, a_sizes);
//   DynamicMemRefType<int32_t> A(std::move(raw_A));

//   int32_t b_data[] = {2, 3, 4, 5};
//   int64_t b_sizes[] = {2, 2};
//   RankedMemRefType<int32_t, 2> raw_B(b_data, b_sizes);
//   DynamicMemRefType<int32_t> B(std::move(raw_B));

//   int32_t c_data[4];
//   int64_t c_sizes[] = {2, 2};
//   RankedMemRefType<int32_t, 2> raw_C(c_data, c_sizes);
//   DynamicMemRefType<int32_t> C(std::move(raw_C));

//   matmul(&A, &B, &C);
//   printMemrefI32(&A);
//   printMemrefI32(&B);
//   printMemrefI32(&C);
//   return 0;
// }