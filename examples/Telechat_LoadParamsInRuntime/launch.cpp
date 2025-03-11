#include <cmath>
#include <cstdint>
#include <iostream>

#include "globals.h"

using namespace std;

extern "C" {
void _mlir_ciface_Telechat(RankedMemRefType<half, 2> *output,
                           RankedMemRefType<int, 1> *input);
}

int main() {
  // initialize all weight and bias
  init_all_globals();

  // init inputs
  int *input_data = new int[5];

  for (int i = 0; i < 5; i++) {
    input_data[i] = 1;
  }

  int64_t input_size[] = {5};
  RankedMemRefType<int, 1> input(input_data, input_size);

  half *output_data = new half[5 * 120000];
  int64_t output_size[] = {5, 120000};
  RankedMemRefType<half, 2> output(output_data, output_size);

  _mlir_ciface_Telechat(&output, &input);
  input.print();
  std::cout << "***********" << std::endl;
  output.print();

  delete_all_globals();
  return 0;
}