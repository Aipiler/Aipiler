#include <cmath>
#include <cstdint>
#include <iostream>

#include "globals.h"

using namespace std;

extern "C" {
void _mlir_ciface_graph0(RankedMemRefType<half, 2> *output,
                         RankedMemRefType<int, 2> *input);
}

int main() {
  // initialize all weight and bias
  init_all_globals();

  // init inputs
  int *input_data = new int[5 * 5120];
  int64_t input_size[] = {5, 5120};
  RankedMemRefType<int, 2> input(input_data, input_size);
  for (int i = 0; i < 5; i++) {
    input_data[i] = 1;
  }

  half *output_data = new half[5 * 5120];
  int64_t output_size[] = {5, 5120};
  RankedMemRefType<half, 2> output(output_data, output_size);

  _mlir_ciface_graph0(&output, &input);
  // input.print();
  rms_weight->print();
  std::cout << "***********" << std::endl;
  output.print();

  delete_all_globals();
  return 0;
}