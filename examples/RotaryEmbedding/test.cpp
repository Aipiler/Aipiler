#include <iostream>

extern "C" void print() {
  std::cout << "call print" << std::endl;
  return;
}