#include "print.h"
#include <cstdio>
#include <iostream>

extern "C" void printMemref(DynamicMemRefType<int32_t> *dy) {
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