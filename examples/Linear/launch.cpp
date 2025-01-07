#include <iostream>

extern "C" {
void call_graph0();
float linear_bias[5];
float linear_weight[5 * 10];
}

int main() {
  for (int i = 0; i < 5; i++) {
    linear_bias[i] = i;
    for (int j = 0; j < 10; j++) {
      linear_weight[i * 10 + j] = j;
    }
  }

  call_graph0();
  return 0;
}