#include <cmath>
#include <cstdint>
#include <iostream>

#include "globals.h"

using namespace std;

const int max_seq_len = 5;
const int hidden_size = 10;

extern "C" {
void _mlir_ciface_graph0(RankedMemRefType<half, 2> *output,
                         RankedMemRefType<int, 2> *input);
}

int main() {
  // initialize all weight and bias
  init_all_globals();

  // init inputs
  int *input_data = new int[max_seq_len * hidden_size];
  int64_t input_size[] = {max_seq_len, hidden_size};
  RankedMemRefType<int, 2> input(input_data, input_size);
  for (int i = 0; i < max_seq_len * hidden_size; i++) {
    input_data[i] = 1;
  }

  half *output_data = new half[max_seq_len * hidden_size];
  int64_t output_size[] = {max_seq_len, hidden_size};
  RankedMemRefType<half, 2> output(output_data, output_size);

  _mlir_ciface_graph0(&output, &input);
  // input.print();
  rms_weight->print();
  std::cout << "***********" << std::endl;
  output.print();

  delete_all_globals();
  return 0;
}