module {
  ml_program.subgraph private @self_attention(%hidden_states: tensor<?x?x?xf32>, %residual: tensor<?x?x?xf32>, %attention_mask: tensor<?x?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = "mix.self_attn"(%hidden_states, %residual, %attention_mask) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?xf32>
    ml_program.output %0 : tensor<?x?x?xf32>
  }
}