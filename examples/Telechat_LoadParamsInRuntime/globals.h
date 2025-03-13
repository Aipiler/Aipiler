#ifndef GLOBALS_H
#define GLOBALS_H
#include "memref.h"

// Layer transformer.word_embeddings.weight: shape torch.Size([120000, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_word_embeddings_weight;
}
void init_transformer_word_embeddings_weight() {
  half *transformer_word_embeddings_weight_data = new half[120000 * 5120];
  int64_t transformer_word_embeddings_weight_shape[2] = {120000, 5120};
  transformer_word_embeddings_weight =
      new RankedMemRefType<half, 2>(transformer_word_embeddings_weight_data,
                                    transformer_word_embeddings_weight_shape);
}

void delete_transformer_word_embeddings_weight() {
  delete transformer_word_embeddings_weight;
}

// Layer transformer.h.0.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_0_input_layernorm_weight;
}
void init_transformer_h_0_input_layernorm_weight() {
  half *transformer_h_0_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_0_input_layernorm_weight_shape[1] = {5120};
  transformer_h_0_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_0_input_layernorm_weight_data,
      transformer_h_0_input_layernorm_weight_shape);
}

void delete_transformer_h_0_input_layernorm_weight() {
  delete transformer_h_0_input_layernorm_weight;
}

// Layer transformer.h.0.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_0_self_attention_query_weight;
}
void init_transformer_h_0_self_attention_query_weight() {
  half *transformer_h_0_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_0_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_0_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_0_self_attention_query_weight_data,
      transformer_h_0_self_attention_query_weight_shape);
}

void delete_transformer_h_0_self_attention_query_weight() {
  delete transformer_h_0_self_attention_query_weight;
}

// Layer transformer.h.0.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_0_self_attention_key_value_weight;
}
void init_transformer_h_0_self_attention_key_value_weight() {
  half *transformer_h_0_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_0_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_0_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_0_self_attention_key_value_weight_data,
          transformer_h_0_self_attention_key_value_weight_shape);
}

void delete_transformer_h_0_self_attention_key_value_weight() {
  delete transformer_h_0_self_attention_key_value_weight;
}

// Layer transformer.h.0.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_0_self_attention_dense_weight;
}
void init_transformer_h_0_self_attention_dense_weight() {
  half *transformer_h_0_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_0_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_0_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_0_self_attention_dense_weight_data,
      transformer_h_0_self_attention_dense_weight_shape);
}

void delete_transformer_h_0_self_attention_dense_weight() {
  delete transformer_h_0_self_attention_dense_weight;
}

// Layer transformer.h.0.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_0_self_attention_dense_bias;
}
void init_transformer_h_0_self_attention_dense_bias() {
  half *transformer_h_0_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_0_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_0_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_0_self_attention_dense_bias_data,
      transformer_h_0_self_attention_dense_bias_shape);
}

void delete_transformer_h_0_self_attention_dense_bias() {
  delete transformer_h_0_self_attention_dense_bias;
}

// Layer transformer.h.0.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_0_post_attention_layernorm_weight;
}
void init_transformer_h_0_post_attention_layernorm_weight() {
  half *transformer_h_0_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_0_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_0_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_0_post_attention_layernorm_weight_data,
          transformer_h_0_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_0_post_attention_layernorm_weight() {
  delete transformer_h_0_post_attention_layernorm_weight;
}

// Layer transformer.h.0.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_0_mlp_gate_proj_weight;
}
void init_transformer_h_0_mlp_gate_proj_weight() {
  half *transformer_h_0_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_0_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_0_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_0_mlp_gate_proj_weight_data,
                                    transformer_h_0_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_0_mlp_gate_proj_weight() {
  delete transformer_h_0_mlp_gate_proj_weight;
}

// Layer transformer.h.0.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_0_mlp_up_proj_weight;
}
void init_transformer_h_0_mlp_up_proj_weight() {
  half *transformer_h_0_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_0_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_0_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_0_mlp_up_proj_weight_data,
                                    transformer_h_0_mlp_up_proj_weight_shape);
}

void delete_transformer_h_0_mlp_up_proj_weight() {
  delete transformer_h_0_mlp_up_proj_weight;
}

// Layer transformer.h.0.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_0_mlp_down_proj_weight;
}
void init_transformer_h_0_mlp_down_proj_weight() {
  half *transformer_h_0_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_0_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_0_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_0_mlp_down_proj_weight_data,
                                    transformer_h_0_mlp_down_proj_weight_shape);
}

void delete_transformer_h_0_mlp_down_proj_weight() {
  delete transformer_h_0_mlp_down_proj_weight;
}

// Layer transformer.h.0.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_0_mlp_down_proj_bias;
}
void init_transformer_h_0_mlp_down_proj_bias() {
  half *transformer_h_0_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_0_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_0_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_0_mlp_down_proj_bias_data,
                                    transformer_h_0_mlp_down_proj_bias_shape);
}

void delete_transformer_h_0_mlp_down_proj_bias() {
  delete transformer_h_0_mlp_down_proj_bias;
}

// Layer transformer.h.1.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_1_input_layernorm_weight;
}
void init_transformer_h_1_input_layernorm_weight() {
  half *transformer_h_1_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_1_input_layernorm_weight_shape[1] = {5120};
  transformer_h_1_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_1_input_layernorm_weight_data,
      transformer_h_1_input_layernorm_weight_shape);
}

void delete_transformer_h_1_input_layernorm_weight() {
  delete transformer_h_1_input_layernorm_weight;
}

// Layer transformer.h.1.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_1_self_attention_query_weight;
}
void init_transformer_h_1_self_attention_query_weight() {
  half *transformer_h_1_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_1_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_1_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_1_self_attention_query_weight_data,
      transformer_h_1_self_attention_query_weight_shape);
}

void delete_transformer_h_1_self_attention_query_weight() {
  delete transformer_h_1_self_attention_query_weight;
}

// Layer transformer.h.1.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_1_self_attention_key_value_weight;
}
void init_transformer_h_1_self_attention_key_value_weight() {
  half *transformer_h_1_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_1_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_1_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_1_self_attention_key_value_weight_data,
          transformer_h_1_self_attention_key_value_weight_shape);
}

void delete_transformer_h_1_self_attention_key_value_weight() {
  delete transformer_h_1_self_attention_key_value_weight;
}

// Layer transformer.h.1.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_1_self_attention_dense_weight;
}
void init_transformer_h_1_self_attention_dense_weight() {
  half *transformer_h_1_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_1_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_1_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_1_self_attention_dense_weight_data,
      transformer_h_1_self_attention_dense_weight_shape);
}

void delete_transformer_h_1_self_attention_dense_weight() {
  delete transformer_h_1_self_attention_dense_weight;
}

// Layer transformer.h.1.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_1_self_attention_dense_bias;
}
void init_transformer_h_1_self_attention_dense_bias() {
  half *transformer_h_1_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_1_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_1_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_1_self_attention_dense_bias_data,
      transformer_h_1_self_attention_dense_bias_shape);
}

void delete_transformer_h_1_self_attention_dense_bias() {
  delete transformer_h_1_self_attention_dense_bias;
}

// Layer transformer.h.1.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_1_post_attention_layernorm_weight;
}
void init_transformer_h_1_post_attention_layernorm_weight() {
  half *transformer_h_1_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_1_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_1_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_1_post_attention_layernorm_weight_data,
          transformer_h_1_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_1_post_attention_layernorm_weight() {
  delete transformer_h_1_post_attention_layernorm_weight;
}

// Layer transformer.h.1.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_1_mlp_gate_proj_weight;
}
void init_transformer_h_1_mlp_gate_proj_weight() {
  half *transformer_h_1_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_1_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_1_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_1_mlp_gate_proj_weight_data,
                                    transformer_h_1_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_1_mlp_gate_proj_weight() {
  delete transformer_h_1_mlp_gate_proj_weight;
}

// Layer transformer.h.1.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_1_mlp_up_proj_weight;
}
void init_transformer_h_1_mlp_up_proj_weight() {
  half *transformer_h_1_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_1_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_1_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_1_mlp_up_proj_weight_data,
                                    transformer_h_1_mlp_up_proj_weight_shape);
}

void delete_transformer_h_1_mlp_up_proj_weight() {
  delete transformer_h_1_mlp_up_proj_weight;
}

// Layer transformer.h.1.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_1_mlp_down_proj_weight;
}
void init_transformer_h_1_mlp_down_proj_weight() {
  half *transformer_h_1_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_1_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_1_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_1_mlp_down_proj_weight_data,
                                    transformer_h_1_mlp_down_proj_weight_shape);
}

void delete_transformer_h_1_mlp_down_proj_weight() {
  delete transformer_h_1_mlp_down_proj_weight;
}

// Layer transformer.h.1.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_1_mlp_down_proj_bias;
}
void init_transformer_h_1_mlp_down_proj_bias() {
  half *transformer_h_1_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_1_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_1_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_1_mlp_down_proj_bias_data,
                                    transformer_h_1_mlp_down_proj_bias_shape);
}

void delete_transformer_h_1_mlp_down_proj_bias() {
  delete transformer_h_1_mlp_down_proj_bias;
}

// Layer transformer.h.2.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_2_input_layernorm_weight;
}
void init_transformer_h_2_input_layernorm_weight() {
  half *transformer_h_2_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_2_input_layernorm_weight_shape[1] = {5120};
  transformer_h_2_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_2_input_layernorm_weight_data,
      transformer_h_2_input_layernorm_weight_shape);
}

void delete_transformer_h_2_input_layernorm_weight() {
  delete transformer_h_2_input_layernorm_weight;
}

// Layer transformer.h.2.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_2_self_attention_query_weight;
}
void init_transformer_h_2_self_attention_query_weight() {
  half *transformer_h_2_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_2_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_2_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_2_self_attention_query_weight_data,
      transformer_h_2_self_attention_query_weight_shape);
}

void delete_transformer_h_2_self_attention_query_weight() {
  delete transformer_h_2_self_attention_query_weight;
}

// Layer transformer.h.2.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_2_self_attention_key_value_weight;
}
void init_transformer_h_2_self_attention_key_value_weight() {
  half *transformer_h_2_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_2_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_2_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_2_self_attention_key_value_weight_data,
          transformer_h_2_self_attention_key_value_weight_shape);
}

void delete_transformer_h_2_self_attention_key_value_weight() {
  delete transformer_h_2_self_attention_key_value_weight;
}

// Layer transformer.h.2.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_2_self_attention_dense_weight;
}
void init_transformer_h_2_self_attention_dense_weight() {
  half *transformer_h_2_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_2_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_2_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_2_self_attention_dense_weight_data,
      transformer_h_2_self_attention_dense_weight_shape);
}

void delete_transformer_h_2_self_attention_dense_weight() {
  delete transformer_h_2_self_attention_dense_weight;
}

// Layer transformer.h.2.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_2_self_attention_dense_bias;
}
void init_transformer_h_2_self_attention_dense_bias() {
  half *transformer_h_2_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_2_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_2_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_2_self_attention_dense_bias_data,
      transformer_h_2_self_attention_dense_bias_shape);
}

void delete_transformer_h_2_self_attention_dense_bias() {
  delete transformer_h_2_self_attention_dense_bias;
}

// Layer transformer.h.2.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_2_post_attention_layernorm_weight;
}
void init_transformer_h_2_post_attention_layernorm_weight() {
  half *transformer_h_2_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_2_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_2_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_2_post_attention_layernorm_weight_data,
          transformer_h_2_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_2_post_attention_layernorm_weight() {
  delete transformer_h_2_post_attention_layernorm_weight;
}

// Layer transformer.h.2.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_2_mlp_gate_proj_weight;
}
void init_transformer_h_2_mlp_gate_proj_weight() {
  half *transformer_h_2_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_2_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_2_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_2_mlp_gate_proj_weight_data,
                                    transformer_h_2_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_2_mlp_gate_proj_weight() {
  delete transformer_h_2_mlp_gate_proj_weight;
}

// Layer transformer.h.2.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_2_mlp_up_proj_weight;
}
void init_transformer_h_2_mlp_up_proj_weight() {
  half *transformer_h_2_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_2_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_2_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_2_mlp_up_proj_weight_data,
                                    transformer_h_2_mlp_up_proj_weight_shape);
}

void delete_transformer_h_2_mlp_up_proj_weight() {
  delete transformer_h_2_mlp_up_proj_weight;
}

// Layer transformer.h.2.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_2_mlp_down_proj_weight;
}
void init_transformer_h_2_mlp_down_proj_weight() {
  half *transformer_h_2_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_2_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_2_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_2_mlp_down_proj_weight_data,
                                    transformer_h_2_mlp_down_proj_weight_shape);
}

void delete_transformer_h_2_mlp_down_proj_weight() {
  delete transformer_h_2_mlp_down_proj_weight;
}

// Layer transformer.h.2.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_2_mlp_down_proj_bias;
}
void init_transformer_h_2_mlp_down_proj_bias() {
  half *transformer_h_2_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_2_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_2_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_2_mlp_down_proj_bias_data,
                                    transformer_h_2_mlp_down_proj_bias_shape);
}

void delete_transformer_h_2_mlp_down_proj_bias() {
  delete transformer_h_2_mlp_down_proj_bias;
}

// Layer transformer.h.3.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_3_input_layernorm_weight;
}
void init_transformer_h_3_input_layernorm_weight() {
  half *transformer_h_3_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_3_input_layernorm_weight_shape[1] = {5120};
  transformer_h_3_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_3_input_layernorm_weight_data,
      transformer_h_3_input_layernorm_weight_shape);
}

void delete_transformer_h_3_input_layernorm_weight() {
  delete transformer_h_3_input_layernorm_weight;
}

// Layer transformer.h.3.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_3_self_attention_query_weight;
}
void init_transformer_h_3_self_attention_query_weight() {
  half *transformer_h_3_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_3_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_3_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_3_self_attention_query_weight_data,
      transformer_h_3_self_attention_query_weight_shape);
}

void delete_transformer_h_3_self_attention_query_weight() {
  delete transformer_h_3_self_attention_query_weight;
}

// Layer transformer.h.3.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_3_self_attention_key_value_weight;
}
void init_transformer_h_3_self_attention_key_value_weight() {
  half *transformer_h_3_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_3_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_3_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_3_self_attention_key_value_weight_data,
          transformer_h_3_self_attention_key_value_weight_shape);
}

void delete_transformer_h_3_self_attention_key_value_weight() {
  delete transformer_h_3_self_attention_key_value_weight;
}

// Layer transformer.h.3.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_3_self_attention_dense_weight;
}
void init_transformer_h_3_self_attention_dense_weight() {
  half *transformer_h_3_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_3_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_3_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_3_self_attention_dense_weight_data,
      transformer_h_3_self_attention_dense_weight_shape);
}

void delete_transformer_h_3_self_attention_dense_weight() {
  delete transformer_h_3_self_attention_dense_weight;
}

// Layer transformer.h.3.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_3_self_attention_dense_bias;
}
void init_transformer_h_3_self_attention_dense_bias() {
  half *transformer_h_3_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_3_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_3_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_3_self_attention_dense_bias_data,
      transformer_h_3_self_attention_dense_bias_shape);
}

void delete_transformer_h_3_self_attention_dense_bias() {
  delete transformer_h_3_self_attention_dense_bias;
}

// Layer transformer.h.3.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_3_post_attention_layernorm_weight;
}
void init_transformer_h_3_post_attention_layernorm_weight() {
  half *transformer_h_3_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_3_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_3_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_3_post_attention_layernorm_weight_data,
          transformer_h_3_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_3_post_attention_layernorm_weight() {
  delete transformer_h_3_post_attention_layernorm_weight;
}

// Layer transformer.h.3.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_3_mlp_gate_proj_weight;
}
void init_transformer_h_3_mlp_gate_proj_weight() {
  half *transformer_h_3_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_3_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_3_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_3_mlp_gate_proj_weight_data,
                                    transformer_h_3_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_3_mlp_gate_proj_weight() {
  delete transformer_h_3_mlp_gate_proj_weight;
}

// Layer transformer.h.3.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_3_mlp_up_proj_weight;
}
void init_transformer_h_3_mlp_up_proj_weight() {
  half *transformer_h_3_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_3_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_3_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_3_mlp_up_proj_weight_data,
                                    transformer_h_3_mlp_up_proj_weight_shape);
}

void delete_transformer_h_3_mlp_up_proj_weight() {
  delete transformer_h_3_mlp_up_proj_weight;
}

// Layer transformer.h.3.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_3_mlp_down_proj_weight;
}
void init_transformer_h_3_mlp_down_proj_weight() {
  half *transformer_h_3_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_3_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_3_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_3_mlp_down_proj_weight_data,
                                    transformer_h_3_mlp_down_proj_weight_shape);
}

void delete_transformer_h_3_mlp_down_proj_weight() {
  delete transformer_h_3_mlp_down_proj_weight;
}

// Layer transformer.h.3.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_3_mlp_down_proj_bias;
}
void init_transformer_h_3_mlp_down_proj_bias() {
  half *transformer_h_3_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_3_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_3_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_3_mlp_down_proj_bias_data,
                                    transformer_h_3_mlp_down_proj_bias_shape);
}

void delete_transformer_h_3_mlp_down_proj_bias() {
  delete transformer_h_3_mlp_down_proj_bias;
}

// Layer transformer.h.4.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_4_input_layernorm_weight;
}
void init_transformer_h_4_input_layernorm_weight() {
  half *transformer_h_4_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_4_input_layernorm_weight_shape[1] = {5120};
  transformer_h_4_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_4_input_layernorm_weight_data,
      transformer_h_4_input_layernorm_weight_shape);
}

void delete_transformer_h_4_input_layernorm_weight() {
  delete transformer_h_4_input_layernorm_weight;
}

// Layer transformer.h.4.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_4_self_attention_query_weight;
}
void init_transformer_h_4_self_attention_query_weight() {
  half *transformer_h_4_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_4_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_4_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_4_self_attention_query_weight_data,
      transformer_h_4_self_attention_query_weight_shape);
}

void delete_transformer_h_4_self_attention_query_weight() {
  delete transformer_h_4_self_attention_query_weight;
}

// Layer transformer.h.4.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_4_self_attention_key_value_weight;
}
void init_transformer_h_4_self_attention_key_value_weight() {
  half *transformer_h_4_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_4_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_4_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_4_self_attention_key_value_weight_data,
          transformer_h_4_self_attention_key_value_weight_shape);
}

void delete_transformer_h_4_self_attention_key_value_weight() {
  delete transformer_h_4_self_attention_key_value_weight;
}

// Layer transformer.h.4.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_4_self_attention_dense_weight;
}
void init_transformer_h_4_self_attention_dense_weight() {
  half *transformer_h_4_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_4_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_4_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_4_self_attention_dense_weight_data,
      transformer_h_4_self_attention_dense_weight_shape);
}

void delete_transformer_h_4_self_attention_dense_weight() {
  delete transformer_h_4_self_attention_dense_weight;
}

// Layer transformer.h.4.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_4_self_attention_dense_bias;
}
void init_transformer_h_4_self_attention_dense_bias() {
  half *transformer_h_4_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_4_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_4_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_4_self_attention_dense_bias_data,
      transformer_h_4_self_attention_dense_bias_shape);
}

void delete_transformer_h_4_self_attention_dense_bias() {
  delete transformer_h_4_self_attention_dense_bias;
}

// Layer transformer.h.4.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_4_post_attention_layernorm_weight;
}
void init_transformer_h_4_post_attention_layernorm_weight() {
  half *transformer_h_4_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_4_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_4_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_4_post_attention_layernorm_weight_data,
          transformer_h_4_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_4_post_attention_layernorm_weight() {
  delete transformer_h_4_post_attention_layernorm_weight;
}

// Layer transformer.h.4.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_4_mlp_gate_proj_weight;
}
void init_transformer_h_4_mlp_gate_proj_weight() {
  half *transformer_h_4_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_4_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_4_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_4_mlp_gate_proj_weight_data,
                                    transformer_h_4_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_4_mlp_gate_proj_weight() {
  delete transformer_h_4_mlp_gate_proj_weight;
}

// Layer transformer.h.4.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_4_mlp_up_proj_weight;
}
void init_transformer_h_4_mlp_up_proj_weight() {
  half *transformer_h_4_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_4_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_4_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_4_mlp_up_proj_weight_data,
                                    transformer_h_4_mlp_up_proj_weight_shape);
}

void delete_transformer_h_4_mlp_up_proj_weight() {
  delete transformer_h_4_mlp_up_proj_weight;
}

// Layer transformer.h.4.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_4_mlp_down_proj_weight;
}
void init_transformer_h_4_mlp_down_proj_weight() {
  half *transformer_h_4_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_4_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_4_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_4_mlp_down_proj_weight_data,
                                    transformer_h_4_mlp_down_proj_weight_shape);
}

void delete_transformer_h_4_mlp_down_proj_weight() {
  delete transformer_h_4_mlp_down_proj_weight;
}

// Layer transformer.h.4.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_4_mlp_down_proj_bias;
}
void init_transformer_h_4_mlp_down_proj_bias() {
  half *transformer_h_4_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_4_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_4_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_4_mlp_down_proj_bias_data,
                                    transformer_h_4_mlp_down_proj_bias_shape);
}

void delete_transformer_h_4_mlp_down_proj_bias() {
  delete transformer_h_4_mlp_down_proj_bias;
}

// Layer transformer.h.5.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_5_input_layernorm_weight;
}
void init_transformer_h_5_input_layernorm_weight() {
  half *transformer_h_5_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_5_input_layernorm_weight_shape[1] = {5120};
  transformer_h_5_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_5_input_layernorm_weight_data,
      transformer_h_5_input_layernorm_weight_shape);
}

void delete_transformer_h_5_input_layernorm_weight() {
  delete transformer_h_5_input_layernorm_weight;
}

// Layer transformer.h.5.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_5_self_attention_query_weight;
}
void init_transformer_h_5_self_attention_query_weight() {
  half *transformer_h_5_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_5_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_5_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_5_self_attention_query_weight_data,
      transformer_h_5_self_attention_query_weight_shape);
}

void delete_transformer_h_5_self_attention_query_weight() {
  delete transformer_h_5_self_attention_query_weight;
}

// Layer transformer.h.5.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_5_self_attention_key_value_weight;
}
void init_transformer_h_5_self_attention_key_value_weight() {
  half *transformer_h_5_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_5_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_5_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_5_self_attention_key_value_weight_data,
          transformer_h_5_self_attention_key_value_weight_shape);
}

void delete_transformer_h_5_self_attention_key_value_weight() {
  delete transformer_h_5_self_attention_key_value_weight;
}

// Layer transformer.h.5.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_5_self_attention_dense_weight;
}
void init_transformer_h_5_self_attention_dense_weight() {
  half *transformer_h_5_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_5_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_5_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_5_self_attention_dense_weight_data,
      transformer_h_5_self_attention_dense_weight_shape);
}

void delete_transformer_h_5_self_attention_dense_weight() {
  delete transformer_h_5_self_attention_dense_weight;
}

// Layer transformer.h.5.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_5_self_attention_dense_bias;
}
void init_transformer_h_5_self_attention_dense_bias() {
  half *transformer_h_5_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_5_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_5_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_5_self_attention_dense_bias_data,
      transformer_h_5_self_attention_dense_bias_shape);
}

void delete_transformer_h_5_self_attention_dense_bias() {
  delete transformer_h_5_self_attention_dense_bias;
}

// Layer transformer.h.5.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_5_post_attention_layernorm_weight;
}
void init_transformer_h_5_post_attention_layernorm_weight() {
  half *transformer_h_5_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_5_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_5_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_5_post_attention_layernorm_weight_data,
          transformer_h_5_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_5_post_attention_layernorm_weight() {
  delete transformer_h_5_post_attention_layernorm_weight;
}

// Layer transformer.h.5.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_5_mlp_gate_proj_weight;
}
void init_transformer_h_5_mlp_gate_proj_weight() {
  half *transformer_h_5_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_5_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_5_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_5_mlp_gate_proj_weight_data,
                                    transformer_h_5_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_5_mlp_gate_proj_weight() {
  delete transformer_h_5_mlp_gate_proj_weight;
}

// Layer transformer.h.5.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_5_mlp_up_proj_weight;
}
void init_transformer_h_5_mlp_up_proj_weight() {
  half *transformer_h_5_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_5_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_5_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_5_mlp_up_proj_weight_data,
                                    transformer_h_5_mlp_up_proj_weight_shape);
}

void delete_transformer_h_5_mlp_up_proj_weight() {
  delete transformer_h_5_mlp_up_proj_weight;
}

// Layer transformer.h.5.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_5_mlp_down_proj_weight;
}
void init_transformer_h_5_mlp_down_proj_weight() {
  half *transformer_h_5_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_5_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_5_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_5_mlp_down_proj_weight_data,
                                    transformer_h_5_mlp_down_proj_weight_shape);
}

void delete_transformer_h_5_mlp_down_proj_weight() {
  delete transformer_h_5_mlp_down_proj_weight;
}

// Layer transformer.h.5.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_5_mlp_down_proj_bias;
}
void init_transformer_h_5_mlp_down_proj_bias() {
  half *transformer_h_5_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_5_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_5_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_5_mlp_down_proj_bias_data,
                                    transformer_h_5_mlp_down_proj_bias_shape);
}

void delete_transformer_h_5_mlp_down_proj_bias() {
  delete transformer_h_5_mlp_down_proj_bias;
}

// Layer transformer.h.6.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_6_input_layernorm_weight;
}
void init_transformer_h_6_input_layernorm_weight() {
  half *transformer_h_6_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_6_input_layernorm_weight_shape[1] = {5120};
  transformer_h_6_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_6_input_layernorm_weight_data,
      transformer_h_6_input_layernorm_weight_shape);
}

void delete_transformer_h_6_input_layernorm_weight() {
  delete transformer_h_6_input_layernorm_weight;
}

// Layer transformer.h.6.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_6_self_attention_query_weight;
}
void init_transformer_h_6_self_attention_query_weight() {
  half *transformer_h_6_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_6_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_6_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_6_self_attention_query_weight_data,
      transformer_h_6_self_attention_query_weight_shape);
}

void delete_transformer_h_6_self_attention_query_weight() {
  delete transformer_h_6_self_attention_query_weight;
}

// Layer transformer.h.6.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_6_self_attention_key_value_weight;
}
void init_transformer_h_6_self_attention_key_value_weight() {
  half *transformer_h_6_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_6_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_6_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_6_self_attention_key_value_weight_data,
          transformer_h_6_self_attention_key_value_weight_shape);
}

void delete_transformer_h_6_self_attention_key_value_weight() {
  delete transformer_h_6_self_attention_key_value_weight;
}

// Layer transformer.h.6.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_6_self_attention_dense_weight;
}
void init_transformer_h_6_self_attention_dense_weight() {
  half *transformer_h_6_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_6_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_6_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_6_self_attention_dense_weight_data,
      transformer_h_6_self_attention_dense_weight_shape);
}

void delete_transformer_h_6_self_attention_dense_weight() {
  delete transformer_h_6_self_attention_dense_weight;
}

// Layer transformer.h.6.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_6_self_attention_dense_bias;
}
void init_transformer_h_6_self_attention_dense_bias() {
  half *transformer_h_6_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_6_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_6_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_6_self_attention_dense_bias_data,
      transformer_h_6_self_attention_dense_bias_shape);
}

void delete_transformer_h_6_self_attention_dense_bias() {
  delete transformer_h_6_self_attention_dense_bias;
}

// Layer transformer.h.6.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_6_post_attention_layernorm_weight;
}
void init_transformer_h_6_post_attention_layernorm_weight() {
  half *transformer_h_6_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_6_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_6_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_6_post_attention_layernorm_weight_data,
          transformer_h_6_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_6_post_attention_layernorm_weight() {
  delete transformer_h_6_post_attention_layernorm_weight;
}

// Layer transformer.h.6.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_6_mlp_gate_proj_weight;
}
void init_transformer_h_6_mlp_gate_proj_weight() {
  half *transformer_h_6_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_6_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_6_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_6_mlp_gate_proj_weight_data,
                                    transformer_h_6_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_6_mlp_gate_proj_weight() {
  delete transformer_h_6_mlp_gate_proj_weight;
}

// Layer transformer.h.6.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_6_mlp_up_proj_weight;
}
void init_transformer_h_6_mlp_up_proj_weight() {
  half *transformer_h_6_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_6_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_6_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_6_mlp_up_proj_weight_data,
                                    transformer_h_6_mlp_up_proj_weight_shape);
}

void delete_transformer_h_6_mlp_up_proj_weight() {
  delete transformer_h_6_mlp_up_proj_weight;
}

// Layer transformer.h.6.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_6_mlp_down_proj_weight;
}
void init_transformer_h_6_mlp_down_proj_weight() {
  half *transformer_h_6_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_6_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_6_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_6_mlp_down_proj_weight_data,
                                    transformer_h_6_mlp_down_proj_weight_shape);
}

void delete_transformer_h_6_mlp_down_proj_weight() {
  delete transformer_h_6_mlp_down_proj_weight;
}

// Layer transformer.h.6.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_6_mlp_down_proj_bias;
}
void init_transformer_h_6_mlp_down_proj_bias() {
  half *transformer_h_6_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_6_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_6_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_6_mlp_down_proj_bias_data,
                                    transformer_h_6_mlp_down_proj_bias_shape);
}

void delete_transformer_h_6_mlp_down_proj_bias() {
  delete transformer_h_6_mlp_down_proj_bias;
}

// Layer transformer.h.7.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_7_input_layernorm_weight;
}
void init_transformer_h_7_input_layernorm_weight() {
  half *transformer_h_7_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_7_input_layernorm_weight_shape[1] = {5120};
  transformer_h_7_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_7_input_layernorm_weight_data,
      transformer_h_7_input_layernorm_weight_shape);
}

void delete_transformer_h_7_input_layernorm_weight() {
  delete transformer_h_7_input_layernorm_weight;
}

// Layer transformer.h.7.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_7_self_attention_query_weight;
}
void init_transformer_h_7_self_attention_query_weight() {
  half *transformer_h_7_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_7_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_7_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_7_self_attention_query_weight_data,
      transformer_h_7_self_attention_query_weight_shape);
}

void delete_transformer_h_7_self_attention_query_weight() {
  delete transformer_h_7_self_attention_query_weight;
}

// Layer transformer.h.7.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_7_self_attention_key_value_weight;
}
void init_transformer_h_7_self_attention_key_value_weight() {
  half *transformer_h_7_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_7_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_7_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_7_self_attention_key_value_weight_data,
          transformer_h_7_self_attention_key_value_weight_shape);
}

void delete_transformer_h_7_self_attention_key_value_weight() {
  delete transformer_h_7_self_attention_key_value_weight;
}

// Layer transformer.h.7.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_7_self_attention_dense_weight;
}
void init_transformer_h_7_self_attention_dense_weight() {
  half *transformer_h_7_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_7_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_7_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_7_self_attention_dense_weight_data,
      transformer_h_7_self_attention_dense_weight_shape);
}

void delete_transformer_h_7_self_attention_dense_weight() {
  delete transformer_h_7_self_attention_dense_weight;
}

// Layer transformer.h.7.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_7_self_attention_dense_bias;
}
void init_transformer_h_7_self_attention_dense_bias() {
  half *transformer_h_7_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_7_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_7_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_7_self_attention_dense_bias_data,
      transformer_h_7_self_attention_dense_bias_shape);
}

void delete_transformer_h_7_self_attention_dense_bias() {
  delete transformer_h_7_self_attention_dense_bias;
}

// Layer transformer.h.7.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_7_post_attention_layernorm_weight;
}
void init_transformer_h_7_post_attention_layernorm_weight() {
  half *transformer_h_7_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_7_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_7_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_7_post_attention_layernorm_weight_data,
          transformer_h_7_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_7_post_attention_layernorm_weight() {
  delete transformer_h_7_post_attention_layernorm_weight;
}

// Layer transformer.h.7.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_7_mlp_gate_proj_weight;
}
void init_transformer_h_7_mlp_gate_proj_weight() {
  half *transformer_h_7_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_7_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_7_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_7_mlp_gate_proj_weight_data,
                                    transformer_h_7_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_7_mlp_gate_proj_weight() {
  delete transformer_h_7_mlp_gate_proj_weight;
}

// Layer transformer.h.7.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_7_mlp_up_proj_weight;
}
void init_transformer_h_7_mlp_up_proj_weight() {
  half *transformer_h_7_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_7_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_7_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_7_mlp_up_proj_weight_data,
                                    transformer_h_7_mlp_up_proj_weight_shape);
}

void delete_transformer_h_7_mlp_up_proj_weight() {
  delete transformer_h_7_mlp_up_proj_weight;
}

// Layer transformer.h.7.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_7_mlp_down_proj_weight;
}
void init_transformer_h_7_mlp_down_proj_weight() {
  half *transformer_h_7_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_7_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_7_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_7_mlp_down_proj_weight_data,
                                    transformer_h_7_mlp_down_proj_weight_shape);
}

void delete_transformer_h_7_mlp_down_proj_weight() {
  delete transformer_h_7_mlp_down_proj_weight;
}

// Layer transformer.h.7.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_7_mlp_down_proj_bias;
}
void init_transformer_h_7_mlp_down_proj_bias() {
  half *transformer_h_7_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_7_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_7_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_7_mlp_down_proj_bias_data,
                                    transformer_h_7_mlp_down_proj_bias_shape);
}

void delete_transformer_h_7_mlp_down_proj_bias() {
  delete transformer_h_7_mlp_down_proj_bias;
}

// Layer transformer.h.8.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_8_input_layernorm_weight;
}
void init_transformer_h_8_input_layernorm_weight() {
  half *transformer_h_8_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_8_input_layernorm_weight_shape[1] = {5120};
  transformer_h_8_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_8_input_layernorm_weight_data,
      transformer_h_8_input_layernorm_weight_shape);
}

void delete_transformer_h_8_input_layernorm_weight() {
  delete transformer_h_8_input_layernorm_weight;
}

// Layer transformer.h.8.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_8_self_attention_query_weight;
}
void init_transformer_h_8_self_attention_query_weight() {
  half *transformer_h_8_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_8_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_8_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_8_self_attention_query_weight_data,
      transformer_h_8_self_attention_query_weight_shape);
}

void delete_transformer_h_8_self_attention_query_weight() {
  delete transformer_h_8_self_attention_query_weight;
}

// Layer transformer.h.8.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_8_self_attention_key_value_weight;
}
void init_transformer_h_8_self_attention_key_value_weight() {
  half *transformer_h_8_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_8_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_8_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_8_self_attention_key_value_weight_data,
          transformer_h_8_self_attention_key_value_weight_shape);
}

void delete_transformer_h_8_self_attention_key_value_weight() {
  delete transformer_h_8_self_attention_key_value_weight;
}

// Layer transformer.h.8.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_8_self_attention_dense_weight;
}
void init_transformer_h_8_self_attention_dense_weight() {
  half *transformer_h_8_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_8_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_8_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_8_self_attention_dense_weight_data,
      transformer_h_8_self_attention_dense_weight_shape);
}

void delete_transformer_h_8_self_attention_dense_weight() {
  delete transformer_h_8_self_attention_dense_weight;
}

// Layer transformer.h.8.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_8_self_attention_dense_bias;
}
void init_transformer_h_8_self_attention_dense_bias() {
  half *transformer_h_8_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_8_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_8_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_8_self_attention_dense_bias_data,
      transformer_h_8_self_attention_dense_bias_shape);
}

void delete_transformer_h_8_self_attention_dense_bias() {
  delete transformer_h_8_self_attention_dense_bias;
}

// Layer transformer.h.8.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_8_post_attention_layernorm_weight;
}
void init_transformer_h_8_post_attention_layernorm_weight() {
  half *transformer_h_8_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_8_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_8_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_8_post_attention_layernorm_weight_data,
          transformer_h_8_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_8_post_attention_layernorm_weight() {
  delete transformer_h_8_post_attention_layernorm_weight;
}

// Layer transformer.h.8.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_8_mlp_gate_proj_weight;
}
void init_transformer_h_8_mlp_gate_proj_weight() {
  half *transformer_h_8_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_8_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_8_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_8_mlp_gate_proj_weight_data,
                                    transformer_h_8_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_8_mlp_gate_proj_weight() {
  delete transformer_h_8_mlp_gate_proj_weight;
}

// Layer transformer.h.8.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_8_mlp_up_proj_weight;
}
void init_transformer_h_8_mlp_up_proj_weight() {
  half *transformer_h_8_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_8_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_8_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_8_mlp_up_proj_weight_data,
                                    transformer_h_8_mlp_up_proj_weight_shape);
}

void delete_transformer_h_8_mlp_up_proj_weight() {
  delete transformer_h_8_mlp_up_proj_weight;
}

// Layer transformer.h.8.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_8_mlp_down_proj_weight;
}
void init_transformer_h_8_mlp_down_proj_weight() {
  half *transformer_h_8_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_8_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_8_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_8_mlp_down_proj_weight_data,
                                    transformer_h_8_mlp_down_proj_weight_shape);
}

void delete_transformer_h_8_mlp_down_proj_weight() {
  delete transformer_h_8_mlp_down_proj_weight;
}

// Layer transformer.h.8.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_8_mlp_down_proj_bias;
}
void init_transformer_h_8_mlp_down_proj_bias() {
  half *transformer_h_8_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_8_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_8_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_8_mlp_down_proj_bias_data,
                                    transformer_h_8_mlp_down_proj_bias_shape);
}

void delete_transformer_h_8_mlp_down_proj_bias() {
  delete transformer_h_8_mlp_down_proj_bias;
}

// Layer transformer.h.9.input_layernorm.weight: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_9_input_layernorm_weight;
}
void init_transformer_h_9_input_layernorm_weight() {
  half *transformer_h_9_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_9_input_layernorm_weight_shape[1] = {5120};
  transformer_h_9_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_9_input_layernorm_weight_data,
      transformer_h_9_input_layernorm_weight_shape);
}

void delete_transformer_h_9_input_layernorm_weight() {
  delete transformer_h_9_input_layernorm_weight;
}

// Layer transformer.h.9.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_9_self_attention_query_weight;
}
void init_transformer_h_9_self_attention_query_weight() {
  half *transformer_h_9_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_9_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_9_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_9_self_attention_query_weight_data,
      transformer_h_9_self_attention_query_weight_shape);
}

void delete_transformer_h_9_self_attention_query_weight() {
  delete transformer_h_9_self_attention_query_weight;
}

// Layer transformer.h.9.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_9_self_attention_key_value_weight;
}
void init_transformer_h_9_self_attention_key_value_weight() {
  half *transformer_h_9_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_9_self_attention_key_value_weight_shape[2] = {10240,
                                                                      5120};
  transformer_h_9_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_9_self_attention_key_value_weight_data,
          transformer_h_9_self_attention_key_value_weight_shape);
}

void delete_transformer_h_9_self_attention_key_value_weight() {
  delete transformer_h_9_self_attention_key_value_weight;
}

// Layer transformer.h.9.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_9_self_attention_dense_weight;
}
void init_transformer_h_9_self_attention_dense_weight() {
  half *transformer_h_9_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_9_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_9_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_9_self_attention_dense_weight_data,
      transformer_h_9_self_attention_dense_weight_shape);
}

void delete_transformer_h_9_self_attention_dense_weight() {
  delete transformer_h_9_self_attention_dense_weight;
}

// Layer transformer.h.9.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_9_self_attention_dense_bias;
}
void init_transformer_h_9_self_attention_dense_bias() {
  half *transformer_h_9_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_9_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_9_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_9_self_attention_dense_bias_data,
      transformer_h_9_self_attention_dense_bias_shape);
}

void delete_transformer_h_9_self_attention_dense_bias() {
  delete transformer_h_9_self_attention_dense_bias;
}

// Layer transformer.h.9.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_9_post_attention_layernorm_weight;
}
void init_transformer_h_9_post_attention_layernorm_weight() {
  half *transformer_h_9_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_9_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_9_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_9_post_attention_layernorm_weight_data,
          transformer_h_9_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_9_post_attention_layernorm_weight() {
  delete transformer_h_9_post_attention_layernorm_weight;
}

// Layer transformer.h.9.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_9_mlp_gate_proj_weight;
}
void init_transformer_h_9_mlp_gate_proj_weight() {
  half *transformer_h_9_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_9_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_9_mlp_gate_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_9_mlp_gate_proj_weight_data,
                                    transformer_h_9_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_9_mlp_gate_proj_weight() {
  delete transformer_h_9_mlp_gate_proj_weight;
}

// Layer transformer.h.9.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_9_mlp_up_proj_weight;
}
void init_transformer_h_9_mlp_up_proj_weight() {
  half *transformer_h_9_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_9_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_9_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_9_mlp_up_proj_weight_data,
                                    transformer_h_9_mlp_up_proj_weight_shape);
}

void delete_transformer_h_9_mlp_up_proj_weight() {
  delete transformer_h_9_mlp_up_proj_weight;
}

// Layer transformer.h.9.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_9_mlp_down_proj_weight;
}
void init_transformer_h_9_mlp_down_proj_weight() {
  half *transformer_h_9_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_9_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_9_mlp_down_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_9_mlp_down_proj_weight_data,
                                    transformer_h_9_mlp_down_proj_weight_shape);
}

void delete_transformer_h_9_mlp_down_proj_weight() {
  delete transformer_h_9_mlp_down_proj_weight;
}

// Layer transformer.h.9.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_9_mlp_down_proj_bias;
}
void init_transformer_h_9_mlp_down_proj_bias() {
  half *transformer_h_9_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_9_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_9_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_9_mlp_down_proj_bias_data,
                                    transformer_h_9_mlp_down_proj_bias_shape);
}

void delete_transformer_h_9_mlp_down_proj_bias() {
  delete transformer_h_9_mlp_down_proj_bias;
}

// Layer transformer.h.10.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_10_input_layernorm_weight;
}
void init_transformer_h_10_input_layernorm_weight() {
  half *transformer_h_10_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_10_input_layernorm_weight_shape[1] = {5120};
  transformer_h_10_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_10_input_layernorm_weight_data,
      transformer_h_10_input_layernorm_weight_shape);
}

void delete_transformer_h_10_input_layernorm_weight() {
  delete transformer_h_10_input_layernorm_weight;
}

// Layer transformer.h.10.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_10_self_attention_query_weight;
}
void init_transformer_h_10_self_attention_query_weight() {
  half *transformer_h_10_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_10_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_10_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_10_self_attention_query_weight_data,
      transformer_h_10_self_attention_query_weight_shape);
}

void delete_transformer_h_10_self_attention_query_weight() {
  delete transformer_h_10_self_attention_query_weight;
}

// Layer transformer.h.10.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_10_self_attention_key_value_weight;
}
void init_transformer_h_10_self_attention_key_value_weight() {
  half *transformer_h_10_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_10_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_10_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_10_self_attention_key_value_weight_data,
          transformer_h_10_self_attention_key_value_weight_shape);
}

void delete_transformer_h_10_self_attention_key_value_weight() {
  delete transformer_h_10_self_attention_key_value_weight;
}

// Layer transformer.h.10.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_10_self_attention_dense_weight;
}
void init_transformer_h_10_self_attention_dense_weight() {
  half *transformer_h_10_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_10_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_10_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_10_self_attention_dense_weight_data,
      transformer_h_10_self_attention_dense_weight_shape);
}

void delete_transformer_h_10_self_attention_dense_weight() {
  delete transformer_h_10_self_attention_dense_weight;
}

// Layer transformer.h.10.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_10_self_attention_dense_bias;
}
void init_transformer_h_10_self_attention_dense_bias() {
  half *transformer_h_10_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_10_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_10_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_10_self_attention_dense_bias_data,
      transformer_h_10_self_attention_dense_bias_shape);
}

void delete_transformer_h_10_self_attention_dense_bias() {
  delete transformer_h_10_self_attention_dense_bias;
}

// Layer transformer.h.10.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_10_post_attention_layernorm_weight;
}
void init_transformer_h_10_post_attention_layernorm_weight() {
  half *transformer_h_10_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_10_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_10_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_10_post_attention_layernorm_weight_data,
          transformer_h_10_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_10_post_attention_layernorm_weight() {
  delete transformer_h_10_post_attention_layernorm_weight;
}

// Layer transformer.h.10.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_10_mlp_gate_proj_weight;
}
void init_transformer_h_10_mlp_gate_proj_weight() {
  half *transformer_h_10_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_10_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_10_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_10_mlp_gate_proj_weight_data,
      transformer_h_10_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_10_mlp_gate_proj_weight() {
  delete transformer_h_10_mlp_gate_proj_weight;
}

// Layer transformer.h.10.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_10_mlp_up_proj_weight;
}
void init_transformer_h_10_mlp_up_proj_weight() {
  half *transformer_h_10_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_10_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_10_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_10_mlp_up_proj_weight_data,
                                    transformer_h_10_mlp_up_proj_weight_shape);
}

void delete_transformer_h_10_mlp_up_proj_weight() {
  delete transformer_h_10_mlp_up_proj_weight;
}

// Layer transformer.h.10.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_10_mlp_down_proj_weight;
}
void init_transformer_h_10_mlp_down_proj_weight() {
  half *transformer_h_10_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_10_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_10_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_10_mlp_down_proj_weight_data,
      transformer_h_10_mlp_down_proj_weight_shape);
}

void delete_transformer_h_10_mlp_down_proj_weight() {
  delete transformer_h_10_mlp_down_proj_weight;
}

// Layer transformer.h.10.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_10_mlp_down_proj_bias;
}
void init_transformer_h_10_mlp_down_proj_bias() {
  half *transformer_h_10_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_10_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_10_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_10_mlp_down_proj_bias_data,
                                    transformer_h_10_mlp_down_proj_bias_shape);
}

void delete_transformer_h_10_mlp_down_proj_bias() {
  delete transformer_h_10_mlp_down_proj_bias;
}

// Layer transformer.h.11.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_11_input_layernorm_weight;
}
void init_transformer_h_11_input_layernorm_weight() {
  half *transformer_h_11_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_11_input_layernorm_weight_shape[1] = {5120};
  transformer_h_11_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_11_input_layernorm_weight_data,
      transformer_h_11_input_layernorm_weight_shape);
}

void delete_transformer_h_11_input_layernorm_weight() {
  delete transformer_h_11_input_layernorm_weight;
}

// Layer transformer.h.11.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_11_self_attention_query_weight;
}
void init_transformer_h_11_self_attention_query_weight() {
  half *transformer_h_11_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_11_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_11_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_11_self_attention_query_weight_data,
      transformer_h_11_self_attention_query_weight_shape);
}

void delete_transformer_h_11_self_attention_query_weight() {
  delete transformer_h_11_self_attention_query_weight;
}

// Layer transformer.h.11.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_11_self_attention_key_value_weight;
}
void init_transformer_h_11_self_attention_key_value_weight() {
  half *transformer_h_11_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_11_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_11_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_11_self_attention_key_value_weight_data,
          transformer_h_11_self_attention_key_value_weight_shape);
}

void delete_transformer_h_11_self_attention_key_value_weight() {
  delete transformer_h_11_self_attention_key_value_weight;
}

// Layer transformer.h.11.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_11_self_attention_dense_weight;
}
void init_transformer_h_11_self_attention_dense_weight() {
  half *transformer_h_11_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_11_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_11_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_11_self_attention_dense_weight_data,
      transformer_h_11_self_attention_dense_weight_shape);
}

void delete_transformer_h_11_self_attention_dense_weight() {
  delete transformer_h_11_self_attention_dense_weight;
}

// Layer transformer.h.11.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_11_self_attention_dense_bias;
}
void init_transformer_h_11_self_attention_dense_bias() {
  half *transformer_h_11_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_11_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_11_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_11_self_attention_dense_bias_data,
      transformer_h_11_self_attention_dense_bias_shape);
}

void delete_transformer_h_11_self_attention_dense_bias() {
  delete transformer_h_11_self_attention_dense_bias;
}

// Layer transformer.h.11.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_11_post_attention_layernorm_weight;
}
void init_transformer_h_11_post_attention_layernorm_weight() {
  half *transformer_h_11_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_11_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_11_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_11_post_attention_layernorm_weight_data,
          transformer_h_11_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_11_post_attention_layernorm_weight() {
  delete transformer_h_11_post_attention_layernorm_weight;
}

// Layer transformer.h.11.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_11_mlp_gate_proj_weight;
}
void init_transformer_h_11_mlp_gate_proj_weight() {
  half *transformer_h_11_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_11_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_11_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_11_mlp_gate_proj_weight_data,
      transformer_h_11_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_11_mlp_gate_proj_weight() {
  delete transformer_h_11_mlp_gate_proj_weight;
}

// Layer transformer.h.11.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_11_mlp_up_proj_weight;
}
void init_transformer_h_11_mlp_up_proj_weight() {
  half *transformer_h_11_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_11_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_11_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_11_mlp_up_proj_weight_data,
                                    transformer_h_11_mlp_up_proj_weight_shape);
}

void delete_transformer_h_11_mlp_up_proj_weight() {
  delete transformer_h_11_mlp_up_proj_weight;
}

// Layer transformer.h.11.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_11_mlp_down_proj_weight;
}
void init_transformer_h_11_mlp_down_proj_weight() {
  half *transformer_h_11_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_11_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_11_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_11_mlp_down_proj_weight_data,
      transformer_h_11_mlp_down_proj_weight_shape);
}

void delete_transformer_h_11_mlp_down_proj_weight() {
  delete transformer_h_11_mlp_down_proj_weight;
}

// Layer transformer.h.11.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_11_mlp_down_proj_bias;
}
void init_transformer_h_11_mlp_down_proj_bias() {
  half *transformer_h_11_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_11_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_11_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_11_mlp_down_proj_bias_data,
                                    transformer_h_11_mlp_down_proj_bias_shape);
}

void delete_transformer_h_11_mlp_down_proj_bias() {
  delete transformer_h_11_mlp_down_proj_bias;
}

// Layer transformer.h.12.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_12_input_layernorm_weight;
}
void init_transformer_h_12_input_layernorm_weight() {
  half *transformer_h_12_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_12_input_layernorm_weight_shape[1] = {5120};
  transformer_h_12_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_12_input_layernorm_weight_data,
      transformer_h_12_input_layernorm_weight_shape);
}

void delete_transformer_h_12_input_layernorm_weight() {
  delete transformer_h_12_input_layernorm_weight;
}

// Layer transformer.h.12.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_12_self_attention_query_weight;
}
void init_transformer_h_12_self_attention_query_weight() {
  half *transformer_h_12_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_12_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_12_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_12_self_attention_query_weight_data,
      transformer_h_12_self_attention_query_weight_shape);
}

void delete_transformer_h_12_self_attention_query_weight() {
  delete transformer_h_12_self_attention_query_weight;
}

// Layer transformer.h.12.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_12_self_attention_key_value_weight;
}
void init_transformer_h_12_self_attention_key_value_weight() {
  half *transformer_h_12_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_12_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_12_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_12_self_attention_key_value_weight_data,
          transformer_h_12_self_attention_key_value_weight_shape);
}

void delete_transformer_h_12_self_attention_key_value_weight() {
  delete transformer_h_12_self_attention_key_value_weight;
}

// Layer transformer.h.12.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_12_self_attention_dense_weight;
}
void init_transformer_h_12_self_attention_dense_weight() {
  half *transformer_h_12_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_12_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_12_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_12_self_attention_dense_weight_data,
      transformer_h_12_self_attention_dense_weight_shape);
}

void delete_transformer_h_12_self_attention_dense_weight() {
  delete transformer_h_12_self_attention_dense_weight;
}

// Layer transformer.h.12.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_12_self_attention_dense_bias;
}
void init_transformer_h_12_self_attention_dense_bias() {
  half *transformer_h_12_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_12_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_12_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_12_self_attention_dense_bias_data,
      transformer_h_12_self_attention_dense_bias_shape);
}

void delete_transformer_h_12_self_attention_dense_bias() {
  delete transformer_h_12_self_attention_dense_bias;
}

// Layer transformer.h.12.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_12_post_attention_layernorm_weight;
}
void init_transformer_h_12_post_attention_layernorm_weight() {
  half *transformer_h_12_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_12_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_12_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_12_post_attention_layernorm_weight_data,
          transformer_h_12_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_12_post_attention_layernorm_weight() {
  delete transformer_h_12_post_attention_layernorm_weight;
}

// Layer transformer.h.12.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_12_mlp_gate_proj_weight;
}
void init_transformer_h_12_mlp_gate_proj_weight() {
  half *transformer_h_12_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_12_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_12_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_12_mlp_gate_proj_weight_data,
      transformer_h_12_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_12_mlp_gate_proj_weight() {
  delete transformer_h_12_mlp_gate_proj_weight;
}

// Layer transformer.h.12.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_12_mlp_up_proj_weight;
}
void init_transformer_h_12_mlp_up_proj_weight() {
  half *transformer_h_12_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_12_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_12_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_12_mlp_up_proj_weight_data,
                                    transformer_h_12_mlp_up_proj_weight_shape);
}

void delete_transformer_h_12_mlp_up_proj_weight() {
  delete transformer_h_12_mlp_up_proj_weight;
}

// Layer transformer.h.12.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_12_mlp_down_proj_weight;
}
void init_transformer_h_12_mlp_down_proj_weight() {
  half *transformer_h_12_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_12_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_12_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_12_mlp_down_proj_weight_data,
      transformer_h_12_mlp_down_proj_weight_shape);
}

void delete_transformer_h_12_mlp_down_proj_weight() {
  delete transformer_h_12_mlp_down_proj_weight;
}

// Layer transformer.h.12.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_12_mlp_down_proj_bias;
}
void init_transformer_h_12_mlp_down_proj_bias() {
  half *transformer_h_12_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_12_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_12_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_12_mlp_down_proj_bias_data,
                                    transformer_h_12_mlp_down_proj_bias_shape);
}

void delete_transformer_h_12_mlp_down_proj_bias() {
  delete transformer_h_12_mlp_down_proj_bias;
}

// Layer transformer.h.13.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_13_input_layernorm_weight;
}
void init_transformer_h_13_input_layernorm_weight() {
  half *transformer_h_13_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_13_input_layernorm_weight_shape[1] = {5120};
  transformer_h_13_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_13_input_layernorm_weight_data,
      transformer_h_13_input_layernorm_weight_shape);
}

void delete_transformer_h_13_input_layernorm_weight() {
  delete transformer_h_13_input_layernorm_weight;
}

// Layer transformer.h.13.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_13_self_attention_query_weight;
}
void init_transformer_h_13_self_attention_query_weight() {
  half *transformer_h_13_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_13_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_13_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_13_self_attention_query_weight_data,
      transformer_h_13_self_attention_query_weight_shape);
}

void delete_transformer_h_13_self_attention_query_weight() {
  delete transformer_h_13_self_attention_query_weight;
}

// Layer transformer.h.13.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_13_self_attention_key_value_weight;
}
void init_transformer_h_13_self_attention_key_value_weight() {
  half *transformer_h_13_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_13_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_13_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_13_self_attention_key_value_weight_data,
          transformer_h_13_self_attention_key_value_weight_shape);
}

void delete_transformer_h_13_self_attention_key_value_weight() {
  delete transformer_h_13_self_attention_key_value_weight;
}

// Layer transformer.h.13.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_13_self_attention_dense_weight;
}
void init_transformer_h_13_self_attention_dense_weight() {
  half *transformer_h_13_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_13_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_13_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_13_self_attention_dense_weight_data,
      transformer_h_13_self_attention_dense_weight_shape);
}

void delete_transformer_h_13_self_attention_dense_weight() {
  delete transformer_h_13_self_attention_dense_weight;
}

// Layer transformer.h.13.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_13_self_attention_dense_bias;
}
void init_transformer_h_13_self_attention_dense_bias() {
  half *transformer_h_13_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_13_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_13_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_13_self_attention_dense_bias_data,
      transformer_h_13_self_attention_dense_bias_shape);
}

void delete_transformer_h_13_self_attention_dense_bias() {
  delete transformer_h_13_self_attention_dense_bias;
}

// Layer transformer.h.13.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_13_post_attention_layernorm_weight;
}
void init_transformer_h_13_post_attention_layernorm_weight() {
  half *transformer_h_13_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_13_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_13_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_13_post_attention_layernorm_weight_data,
          transformer_h_13_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_13_post_attention_layernorm_weight() {
  delete transformer_h_13_post_attention_layernorm_weight;
}

// Layer transformer.h.13.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_13_mlp_gate_proj_weight;
}
void init_transformer_h_13_mlp_gate_proj_weight() {
  half *transformer_h_13_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_13_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_13_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_13_mlp_gate_proj_weight_data,
      transformer_h_13_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_13_mlp_gate_proj_weight() {
  delete transformer_h_13_mlp_gate_proj_weight;
}

// Layer transformer.h.13.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_13_mlp_up_proj_weight;
}
void init_transformer_h_13_mlp_up_proj_weight() {
  half *transformer_h_13_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_13_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_13_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_13_mlp_up_proj_weight_data,
                                    transformer_h_13_mlp_up_proj_weight_shape);
}

void delete_transformer_h_13_mlp_up_proj_weight() {
  delete transformer_h_13_mlp_up_proj_weight;
}

// Layer transformer.h.13.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_13_mlp_down_proj_weight;
}
void init_transformer_h_13_mlp_down_proj_weight() {
  half *transformer_h_13_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_13_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_13_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_13_mlp_down_proj_weight_data,
      transformer_h_13_mlp_down_proj_weight_shape);
}

void delete_transformer_h_13_mlp_down_proj_weight() {
  delete transformer_h_13_mlp_down_proj_weight;
}

// Layer transformer.h.13.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_13_mlp_down_proj_bias;
}
void init_transformer_h_13_mlp_down_proj_bias() {
  half *transformer_h_13_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_13_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_13_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_13_mlp_down_proj_bias_data,
                                    transformer_h_13_mlp_down_proj_bias_shape);
}

void delete_transformer_h_13_mlp_down_proj_bias() {
  delete transformer_h_13_mlp_down_proj_bias;
}

// Layer transformer.h.14.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_14_input_layernorm_weight;
}
void init_transformer_h_14_input_layernorm_weight() {
  half *transformer_h_14_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_14_input_layernorm_weight_shape[1] = {5120};
  transformer_h_14_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_14_input_layernorm_weight_data,
      transformer_h_14_input_layernorm_weight_shape);
}

void delete_transformer_h_14_input_layernorm_weight() {
  delete transformer_h_14_input_layernorm_weight;
}

// Layer transformer.h.14.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_14_self_attention_query_weight;
}
void init_transformer_h_14_self_attention_query_weight() {
  half *transformer_h_14_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_14_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_14_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_14_self_attention_query_weight_data,
      transformer_h_14_self_attention_query_weight_shape);
}

void delete_transformer_h_14_self_attention_query_weight() {
  delete transformer_h_14_self_attention_query_weight;
}

// Layer transformer.h.14.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_14_self_attention_key_value_weight;
}
void init_transformer_h_14_self_attention_key_value_weight() {
  half *transformer_h_14_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_14_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_14_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_14_self_attention_key_value_weight_data,
          transformer_h_14_self_attention_key_value_weight_shape);
}

void delete_transformer_h_14_self_attention_key_value_weight() {
  delete transformer_h_14_self_attention_key_value_weight;
}

// Layer transformer.h.14.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_14_self_attention_dense_weight;
}
void init_transformer_h_14_self_attention_dense_weight() {
  half *transformer_h_14_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_14_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_14_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_14_self_attention_dense_weight_data,
      transformer_h_14_self_attention_dense_weight_shape);
}

void delete_transformer_h_14_self_attention_dense_weight() {
  delete transformer_h_14_self_attention_dense_weight;
}

// Layer transformer.h.14.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_14_self_attention_dense_bias;
}
void init_transformer_h_14_self_attention_dense_bias() {
  half *transformer_h_14_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_14_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_14_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_14_self_attention_dense_bias_data,
      transformer_h_14_self_attention_dense_bias_shape);
}

void delete_transformer_h_14_self_attention_dense_bias() {
  delete transformer_h_14_self_attention_dense_bias;
}

// Layer transformer.h.14.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_14_post_attention_layernorm_weight;
}
void init_transformer_h_14_post_attention_layernorm_weight() {
  half *transformer_h_14_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_14_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_14_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_14_post_attention_layernorm_weight_data,
          transformer_h_14_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_14_post_attention_layernorm_weight() {
  delete transformer_h_14_post_attention_layernorm_weight;
}

// Layer transformer.h.14.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_14_mlp_gate_proj_weight;
}
void init_transformer_h_14_mlp_gate_proj_weight() {
  half *transformer_h_14_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_14_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_14_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_14_mlp_gate_proj_weight_data,
      transformer_h_14_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_14_mlp_gate_proj_weight() {
  delete transformer_h_14_mlp_gate_proj_weight;
}

// Layer transformer.h.14.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_14_mlp_up_proj_weight;
}
void init_transformer_h_14_mlp_up_proj_weight() {
  half *transformer_h_14_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_14_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_14_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_14_mlp_up_proj_weight_data,
                                    transformer_h_14_mlp_up_proj_weight_shape);
}

void delete_transformer_h_14_mlp_up_proj_weight() {
  delete transformer_h_14_mlp_up_proj_weight;
}

// Layer transformer.h.14.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_14_mlp_down_proj_weight;
}
void init_transformer_h_14_mlp_down_proj_weight() {
  half *transformer_h_14_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_14_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_14_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_14_mlp_down_proj_weight_data,
      transformer_h_14_mlp_down_proj_weight_shape);
}

void delete_transformer_h_14_mlp_down_proj_weight() {
  delete transformer_h_14_mlp_down_proj_weight;
}

// Layer transformer.h.14.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_14_mlp_down_proj_bias;
}
void init_transformer_h_14_mlp_down_proj_bias() {
  half *transformer_h_14_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_14_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_14_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_14_mlp_down_proj_bias_data,
                                    transformer_h_14_mlp_down_proj_bias_shape);
}

void delete_transformer_h_14_mlp_down_proj_bias() {
  delete transformer_h_14_mlp_down_proj_bias;
}

// Layer transformer.h.15.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_15_input_layernorm_weight;
}
void init_transformer_h_15_input_layernorm_weight() {
  half *transformer_h_15_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_15_input_layernorm_weight_shape[1] = {5120};
  transformer_h_15_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_15_input_layernorm_weight_data,
      transformer_h_15_input_layernorm_weight_shape);
}

void delete_transformer_h_15_input_layernorm_weight() {
  delete transformer_h_15_input_layernorm_weight;
}

// Layer transformer.h.15.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_15_self_attention_query_weight;
}
void init_transformer_h_15_self_attention_query_weight() {
  half *transformer_h_15_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_15_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_15_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_15_self_attention_query_weight_data,
      transformer_h_15_self_attention_query_weight_shape);
}

void delete_transformer_h_15_self_attention_query_weight() {
  delete transformer_h_15_self_attention_query_weight;
}

// Layer transformer.h.15.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_15_self_attention_key_value_weight;
}
void init_transformer_h_15_self_attention_key_value_weight() {
  half *transformer_h_15_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_15_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_15_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_15_self_attention_key_value_weight_data,
          transformer_h_15_self_attention_key_value_weight_shape);
}

void delete_transformer_h_15_self_attention_key_value_weight() {
  delete transformer_h_15_self_attention_key_value_weight;
}

// Layer transformer.h.15.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_15_self_attention_dense_weight;
}
void init_transformer_h_15_self_attention_dense_weight() {
  half *transformer_h_15_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_15_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_15_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_15_self_attention_dense_weight_data,
      transformer_h_15_self_attention_dense_weight_shape);
}

void delete_transformer_h_15_self_attention_dense_weight() {
  delete transformer_h_15_self_attention_dense_weight;
}

// Layer transformer.h.15.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_15_self_attention_dense_bias;
}
void init_transformer_h_15_self_attention_dense_bias() {
  half *transformer_h_15_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_15_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_15_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_15_self_attention_dense_bias_data,
      transformer_h_15_self_attention_dense_bias_shape);
}

void delete_transformer_h_15_self_attention_dense_bias() {
  delete transformer_h_15_self_attention_dense_bias;
}

// Layer transformer.h.15.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_15_post_attention_layernorm_weight;
}
void init_transformer_h_15_post_attention_layernorm_weight() {
  half *transformer_h_15_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_15_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_15_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_15_post_attention_layernorm_weight_data,
          transformer_h_15_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_15_post_attention_layernorm_weight() {
  delete transformer_h_15_post_attention_layernorm_weight;
}

// Layer transformer.h.15.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_15_mlp_gate_proj_weight;
}
void init_transformer_h_15_mlp_gate_proj_weight() {
  half *transformer_h_15_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_15_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_15_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_15_mlp_gate_proj_weight_data,
      transformer_h_15_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_15_mlp_gate_proj_weight() {
  delete transformer_h_15_mlp_gate_proj_weight;
}

// Layer transformer.h.15.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_15_mlp_up_proj_weight;
}
void init_transformer_h_15_mlp_up_proj_weight() {
  half *transformer_h_15_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_15_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_15_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_15_mlp_up_proj_weight_data,
                                    transformer_h_15_mlp_up_proj_weight_shape);
}

void delete_transformer_h_15_mlp_up_proj_weight() {
  delete transformer_h_15_mlp_up_proj_weight;
}

// Layer transformer.h.15.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_15_mlp_down_proj_weight;
}
void init_transformer_h_15_mlp_down_proj_weight() {
  half *transformer_h_15_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_15_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_15_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_15_mlp_down_proj_weight_data,
      transformer_h_15_mlp_down_proj_weight_shape);
}

void delete_transformer_h_15_mlp_down_proj_weight() {
  delete transformer_h_15_mlp_down_proj_weight;
}

// Layer transformer.h.15.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_15_mlp_down_proj_bias;
}
void init_transformer_h_15_mlp_down_proj_bias() {
  half *transformer_h_15_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_15_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_15_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_15_mlp_down_proj_bias_data,
                                    transformer_h_15_mlp_down_proj_bias_shape);
}

void delete_transformer_h_15_mlp_down_proj_bias() {
  delete transformer_h_15_mlp_down_proj_bias;
}

// Layer transformer.h.16.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_16_input_layernorm_weight;
}
void init_transformer_h_16_input_layernorm_weight() {
  half *transformer_h_16_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_16_input_layernorm_weight_shape[1] = {5120};
  transformer_h_16_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_16_input_layernorm_weight_data,
      transformer_h_16_input_layernorm_weight_shape);
}

void delete_transformer_h_16_input_layernorm_weight() {
  delete transformer_h_16_input_layernorm_weight;
}

// Layer transformer.h.16.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_16_self_attention_query_weight;
}
void init_transformer_h_16_self_attention_query_weight() {
  half *transformer_h_16_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_16_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_16_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_16_self_attention_query_weight_data,
      transformer_h_16_self_attention_query_weight_shape);
}

void delete_transformer_h_16_self_attention_query_weight() {
  delete transformer_h_16_self_attention_query_weight;
}

// Layer transformer.h.16.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_16_self_attention_key_value_weight;
}
void init_transformer_h_16_self_attention_key_value_weight() {
  half *transformer_h_16_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_16_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_16_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_16_self_attention_key_value_weight_data,
          transformer_h_16_self_attention_key_value_weight_shape);
}

void delete_transformer_h_16_self_attention_key_value_weight() {
  delete transformer_h_16_self_attention_key_value_weight;
}

// Layer transformer.h.16.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_16_self_attention_dense_weight;
}
void init_transformer_h_16_self_attention_dense_weight() {
  half *transformer_h_16_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_16_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_16_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_16_self_attention_dense_weight_data,
      transformer_h_16_self_attention_dense_weight_shape);
}

void delete_transformer_h_16_self_attention_dense_weight() {
  delete transformer_h_16_self_attention_dense_weight;
}

// Layer transformer.h.16.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_16_self_attention_dense_bias;
}
void init_transformer_h_16_self_attention_dense_bias() {
  half *transformer_h_16_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_16_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_16_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_16_self_attention_dense_bias_data,
      transformer_h_16_self_attention_dense_bias_shape);
}

void delete_transformer_h_16_self_attention_dense_bias() {
  delete transformer_h_16_self_attention_dense_bias;
}

// Layer transformer.h.16.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_16_post_attention_layernorm_weight;
}
void init_transformer_h_16_post_attention_layernorm_weight() {
  half *transformer_h_16_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_16_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_16_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_16_post_attention_layernorm_weight_data,
          transformer_h_16_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_16_post_attention_layernorm_weight() {
  delete transformer_h_16_post_attention_layernorm_weight;
}

// Layer transformer.h.16.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_16_mlp_gate_proj_weight;
}
void init_transformer_h_16_mlp_gate_proj_weight() {
  half *transformer_h_16_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_16_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_16_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_16_mlp_gate_proj_weight_data,
      transformer_h_16_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_16_mlp_gate_proj_weight() {
  delete transformer_h_16_mlp_gate_proj_weight;
}

// Layer transformer.h.16.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_16_mlp_up_proj_weight;
}
void init_transformer_h_16_mlp_up_proj_weight() {
  half *transformer_h_16_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_16_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_16_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_16_mlp_up_proj_weight_data,
                                    transformer_h_16_mlp_up_proj_weight_shape);
}

void delete_transformer_h_16_mlp_up_proj_weight() {
  delete transformer_h_16_mlp_up_proj_weight;
}

// Layer transformer.h.16.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_16_mlp_down_proj_weight;
}
void init_transformer_h_16_mlp_down_proj_weight() {
  half *transformer_h_16_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_16_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_16_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_16_mlp_down_proj_weight_data,
      transformer_h_16_mlp_down_proj_weight_shape);
}

void delete_transformer_h_16_mlp_down_proj_weight() {
  delete transformer_h_16_mlp_down_proj_weight;
}

// Layer transformer.h.16.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_16_mlp_down_proj_bias;
}
void init_transformer_h_16_mlp_down_proj_bias() {
  half *transformer_h_16_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_16_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_16_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_16_mlp_down_proj_bias_data,
                                    transformer_h_16_mlp_down_proj_bias_shape);
}

void delete_transformer_h_16_mlp_down_proj_bias() {
  delete transformer_h_16_mlp_down_proj_bias;
}

// Layer transformer.h.17.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_17_input_layernorm_weight;
}
void init_transformer_h_17_input_layernorm_weight() {
  half *transformer_h_17_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_17_input_layernorm_weight_shape[1] = {5120};
  transformer_h_17_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_17_input_layernorm_weight_data,
      transformer_h_17_input_layernorm_weight_shape);
}

void delete_transformer_h_17_input_layernorm_weight() {
  delete transformer_h_17_input_layernorm_weight;
}

// Layer transformer.h.17.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_17_self_attention_query_weight;
}
void init_transformer_h_17_self_attention_query_weight() {
  half *transformer_h_17_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_17_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_17_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_17_self_attention_query_weight_data,
      transformer_h_17_self_attention_query_weight_shape);
}

void delete_transformer_h_17_self_attention_query_weight() {
  delete transformer_h_17_self_attention_query_weight;
}

// Layer transformer.h.17.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_17_self_attention_key_value_weight;
}
void init_transformer_h_17_self_attention_key_value_weight() {
  half *transformer_h_17_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_17_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_17_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_17_self_attention_key_value_weight_data,
          transformer_h_17_self_attention_key_value_weight_shape);
}

void delete_transformer_h_17_self_attention_key_value_weight() {
  delete transformer_h_17_self_attention_key_value_weight;
}

// Layer transformer.h.17.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_17_self_attention_dense_weight;
}
void init_transformer_h_17_self_attention_dense_weight() {
  half *transformer_h_17_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_17_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_17_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_17_self_attention_dense_weight_data,
      transformer_h_17_self_attention_dense_weight_shape);
}

void delete_transformer_h_17_self_attention_dense_weight() {
  delete transformer_h_17_self_attention_dense_weight;
}

// Layer transformer.h.17.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_17_self_attention_dense_bias;
}
void init_transformer_h_17_self_attention_dense_bias() {
  half *transformer_h_17_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_17_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_17_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_17_self_attention_dense_bias_data,
      transformer_h_17_self_attention_dense_bias_shape);
}

void delete_transformer_h_17_self_attention_dense_bias() {
  delete transformer_h_17_self_attention_dense_bias;
}

// Layer transformer.h.17.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_17_post_attention_layernorm_weight;
}
void init_transformer_h_17_post_attention_layernorm_weight() {
  half *transformer_h_17_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_17_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_17_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_17_post_attention_layernorm_weight_data,
          transformer_h_17_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_17_post_attention_layernorm_weight() {
  delete transformer_h_17_post_attention_layernorm_weight;
}

// Layer transformer.h.17.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_17_mlp_gate_proj_weight;
}
void init_transformer_h_17_mlp_gate_proj_weight() {
  half *transformer_h_17_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_17_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_17_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_17_mlp_gate_proj_weight_data,
      transformer_h_17_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_17_mlp_gate_proj_weight() {
  delete transformer_h_17_mlp_gate_proj_weight;
}

// Layer transformer.h.17.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_17_mlp_up_proj_weight;
}
void init_transformer_h_17_mlp_up_proj_weight() {
  half *transformer_h_17_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_17_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_17_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_17_mlp_up_proj_weight_data,
                                    transformer_h_17_mlp_up_proj_weight_shape);
}

void delete_transformer_h_17_mlp_up_proj_weight() {
  delete transformer_h_17_mlp_up_proj_weight;
}

// Layer transformer.h.17.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_17_mlp_down_proj_weight;
}
void init_transformer_h_17_mlp_down_proj_weight() {
  half *transformer_h_17_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_17_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_17_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_17_mlp_down_proj_weight_data,
      transformer_h_17_mlp_down_proj_weight_shape);
}

void delete_transformer_h_17_mlp_down_proj_weight() {
  delete transformer_h_17_mlp_down_proj_weight;
}

// Layer transformer.h.17.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_17_mlp_down_proj_bias;
}
void init_transformer_h_17_mlp_down_proj_bias() {
  half *transformer_h_17_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_17_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_17_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_17_mlp_down_proj_bias_data,
                                    transformer_h_17_mlp_down_proj_bias_shape);
}

void delete_transformer_h_17_mlp_down_proj_bias() {
  delete transformer_h_17_mlp_down_proj_bias;
}

// Layer transformer.h.18.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_18_input_layernorm_weight;
}
void init_transformer_h_18_input_layernorm_weight() {
  half *transformer_h_18_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_18_input_layernorm_weight_shape[1] = {5120};
  transformer_h_18_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_18_input_layernorm_weight_data,
      transformer_h_18_input_layernorm_weight_shape);
}

void delete_transformer_h_18_input_layernorm_weight() {
  delete transformer_h_18_input_layernorm_weight;
}

// Layer transformer.h.18.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_18_self_attention_query_weight;
}
void init_transformer_h_18_self_attention_query_weight() {
  half *transformer_h_18_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_18_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_18_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_18_self_attention_query_weight_data,
      transformer_h_18_self_attention_query_weight_shape);
}

void delete_transformer_h_18_self_attention_query_weight() {
  delete transformer_h_18_self_attention_query_weight;
}

// Layer transformer.h.18.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_18_self_attention_key_value_weight;
}
void init_transformer_h_18_self_attention_key_value_weight() {
  half *transformer_h_18_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_18_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_18_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_18_self_attention_key_value_weight_data,
          transformer_h_18_self_attention_key_value_weight_shape);
}

void delete_transformer_h_18_self_attention_key_value_weight() {
  delete transformer_h_18_self_attention_key_value_weight;
}

// Layer transformer.h.18.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_18_self_attention_dense_weight;
}
void init_transformer_h_18_self_attention_dense_weight() {
  half *transformer_h_18_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_18_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_18_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_18_self_attention_dense_weight_data,
      transformer_h_18_self_attention_dense_weight_shape);
}

void delete_transformer_h_18_self_attention_dense_weight() {
  delete transformer_h_18_self_attention_dense_weight;
}

// Layer transformer.h.18.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_18_self_attention_dense_bias;
}
void init_transformer_h_18_self_attention_dense_bias() {
  half *transformer_h_18_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_18_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_18_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_18_self_attention_dense_bias_data,
      transformer_h_18_self_attention_dense_bias_shape);
}

void delete_transformer_h_18_self_attention_dense_bias() {
  delete transformer_h_18_self_attention_dense_bias;
}

// Layer transformer.h.18.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_18_post_attention_layernorm_weight;
}
void init_transformer_h_18_post_attention_layernorm_weight() {
  half *transformer_h_18_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_18_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_18_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_18_post_attention_layernorm_weight_data,
          transformer_h_18_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_18_post_attention_layernorm_weight() {
  delete transformer_h_18_post_attention_layernorm_weight;
}

// Layer transformer.h.18.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_18_mlp_gate_proj_weight;
}
void init_transformer_h_18_mlp_gate_proj_weight() {
  half *transformer_h_18_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_18_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_18_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_18_mlp_gate_proj_weight_data,
      transformer_h_18_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_18_mlp_gate_proj_weight() {
  delete transformer_h_18_mlp_gate_proj_weight;
}

// Layer transformer.h.18.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_18_mlp_up_proj_weight;
}
void init_transformer_h_18_mlp_up_proj_weight() {
  half *transformer_h_18_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_18_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_18_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_18_mlp_up_proj_weight_data,
                                    transformer_h_18_mlp_up_proj_weight_shape);
}

void delete_transformer_h_18_mlp_up_proj_weight() {
  delete transformer_h_18_mlp_up_proj_weight;
}

// Layer transformer.h.18.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_18_mlp_down_proj_weight;
}
void init_transformer_h_18_mlp_down_proj_weight() {
  half *transformer_h_18_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_18_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_18_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_18_mlp_down_proj_weight_data,
      transformer_h_18_mlp_down_proj_weight_shape);
}

void delete_transformer_h_18_mlp_down_proj_weight() {
  delete transformer_h_18_mlp_down_proj_weight;
}

// Layer transformer.h.18.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_18_mlp_down_proj_bias;
}
void init_transformer_h_18_mlp_down_proj_bias() {
  half *transformer_h_18_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_18_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_18_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_18_mlp_down_proj_bias_data,
                                    transformer_h_18_mlp_down_proj_bias_shape);
}

void delete_transformer_h_18_mlp_down_proj_bias() {
  delete transformer_h_18_mlp_down_proj_bias;
}

// Layer transformer.h.19.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_19_input_layernorm_weight;
}
void init_transformer_h_19_input_layernorm_weight() {
  half *transformer_h_19_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_19_input_layernorm_weight_shape[1] = {5120};
  transformer_h_19_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_19_input_layernorm_weight_data,
      transformer_h_19_input_layernorm_weight_shape);
}

void delete_transformer_h_19_input_layernorm_weight() {
  delete transformer_h_19_input_layernorm_weight;
}

// Layer transformer.h.19.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_19_self_attention_query_weight;
}
void init_transformer_h_19_self_attention_query_weight() {
  half *transformer_h_19_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_19_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_19_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_19_self_attention_query_weight_data,
      transformer_h_19_self_attention_query_weight_shape);
}

void delete_transformer_h_19_self_attention_query_weight() {
  delete transformer_h_19_self_attention_query_weight;
}

// Layer transformer.h.19.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_19_self_attention_key_value_weight;
}
void init_transformer_h_19_self_attention_key_value_weight() {
  half *transformer_h_19_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_19_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_19_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_19_self_attention_key_value_weight_data,
          transformer_h_19_self_attention_key_value_weight_shape);
}

void delete_transformer_h_19_self_attention_key_value_weight() {
  delete transformer_h_19_self_attention_key_value_weight;
}

// Layer transformer.h.19.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_19_self_attention_dense_weight;
}
void init_transformer_h_19_self_attention_dense_weight() {
  half *transformer_h_19_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_19_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_19_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_19_self_attention_dense_weight_data,
      transformer_h_19_self_attention_dense_weight_shape);
}

void delete_transformer_h_19_self_attention_dense_weight() {
  delete transformer_h_19_self_attention_dense_weight;
}

// Layer transformer.h.19.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_19_self_attention_dense_bias;
}
void init_transformer_h_19_self_attention_dense_bias() {
  half *transformer_h_19_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_19_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_19_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_19_self_attention_dense_bias_data,
      transformer_h_19_self_attention_dense_bias_shape);
}

void delete_transformer_h_19_self_attention_dense_bias() {
  delete transformer_h_19_self_attention_dense_bias;
}

// Layer transformer.h.19.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_19_post_attention_layernorm_weight;
}
void init_transformer_h_19_post_attention_layernorm_weight() {
  half *transformer_h_19_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_19_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_19_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_19_post_attention_layernorm_weight_data,
          transformer_h_19_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_19_post_attention_layernorm_weight() {
  delete transformer_h_19_post_attention_layernorm_weight;
}

// Layer transformer.h.19.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_19_mlp_gate_proj_weight;
}
void init_transformer_h_19_mlp_gate_proj_weight() {
  half *transformer_h_19_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_19_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_19_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_19_mlp_gate_proj_weight_data,
      transformer_h_19_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_19_mlp_gate_proj_weight() {
  delete transformer_h_19_mlp_gate_proj_weight;
}

// Layer transformer.h.19.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_19_mlp_up_proj_weight;
}
void init_transformer_h_19_mlp_up_proj_weight() {
  half *transformer_h_19_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_19_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_19_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_19_mlp_up_proj_weight_data,
                                    transformer_h_19_mlp_up_proj_weight_shape);
}

void delete_transformer_h_19_mlp_up_proj_weight() {
  delete transformer_h_19_mlp_up_proj_weight;
}

// Layer transformer.h.19.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_19_mlp_down_proj_weight;
}
void init_transformer_h_19_mlp_down_proj_weight() {
  half *transformer_h_19_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_19_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_19_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_19_mlp_down_proj_weight_data,
      transformer_h_19_mlp_down_proj_weight_shape);
}

void delete_transformer_h_19_mlp_down_proj_weight() {
  delete transformer_h_19_mlp_down_proj_weight;
}

// Layer transformer.h.19.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_19_mlp_down_proj_bias;
}
void init_transformer_h_19_mlp_down_proj_bias() {
  half *transformer_h_19_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_19_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_19_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_19_mlp_down_proj_bias_data,
                                    transformer_h_19_mlp_down_proj_bias_shape);
}

void delete_transformer_h_19_mlp_down_proj_bias() {
  delete transformer_h_19_mlp_down_proj_bias;
}

// Layer transformer.h.20.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_20_input_layernorm_weight;
}
void init_transformer_h_20_input_layernorm_weight() {
  half *transformer_h_20_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_20_input_layernorm_weight_shape[1] = {5120};
  transformer_h_20_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_20_input_layernorm_weight_data,
      transformer_h_20_input_layernorm_weight_shape);
}

void delete_transformer_h_20_input_layernorm_weight() {
  delete transformer_h_20_input_layernorm_weight;
}

// Layer transformer.h.20.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_20_self_attention_query_weight;
}
void init_transformer_h_20_self_attention_query_weight() {
  half *transformer_h_20_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_20_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_20_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_20_self_attention_query_weight_data,
      transformer_h_20_self_attention_query_weight_shape);
}

void delete_transformer_h_20_self_attention_query_weight() {
  delete transformer_h_20_self_attention_query_weight;
}

// Layer transformer.h.20.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_20_self_attention_key_value_weight;
}
void init_transformer_h_20_self_attention_key_value_weight() {
  half *transformer_h_20_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_20_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_20_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_20_self_attention_key_value_weight_data,
          transformer_h_20_self_attention_key_value_weight_shape);
}

void delete_transformer_h_20_self_attention_key_value_weight() {
  delete transformer_h_20_self_attention_key_value_weight;
}

// Layer transformer.h.20.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_20_self_attention_dense_weight;
}
void init_transformer_h_20_self_attention_dense_weight() {
  half *transformer_h_20_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_20_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_20_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_20_self_attention_dense_weight_data,
      transformer_h_20_self_attention_dense_weight_shape);
}

void delete_transformer_h_20_self_attention_dense_weight() {
  delete transformer_h_20_self_attention_dense_weight;
}

// Layer transformer.h.20.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_20_self_attention_dense_bias;
}
void init_transformer_h_20_self_attention_dense_bias() {
  half *transformer_h_20_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_20_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_20_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_20_self_attention_dense_bias_data,
      transformer_h_20_self_attention_dense_bias_shape);
}

void delete_transformer_h_20_self_attention_dense_bias() {
  delete transformer_h_20_self_attention_dense_bias;
}

// Layer transformer.h.20.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_20_post_attention_layernorm_weight;
}
void init_transformer_h_20_post_attention_layernorm_weight() {
  half *transformer_h_20_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_20_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_20_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_20_post_attention_layernorm_weight_data,
          transformer_h_20_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_20_post_attention_layernorm_weight() {
  delete transformer_h_20_post_attention_layernorm_weight;
}

// Layer transformer.h.20.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_20_mlp_gate_proj_weight;
}
void init_transformer_h_20_mlp_gate_proj_weight() {
  half *transformer_h_20_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_20_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_20_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_20_mlp_gate_proj_weight_data,
      transformer_h_20_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_20_mlp_gate_proj_weight() {
  delete transformer_h_20_mlp_gate_proj_weight;
}

// Layer transformer.h.20.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_20_mlp_up_proj_weight;
}
void init_transformer_h_20_mlp_up_proj_weight() {
  half *transformer_h_20_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_20_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_20_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_20_mlp_up_proj_weight_data,
                                    transformer_h_20_mlp_up_proj_weight_shape);
}

void delete_transformer_h_20_mlp_up_proj_weight() {
  delete transformer_h_20_mlp_up_proj_weight;
}

// Layer transformer.h.20.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_20_mlp_down_proj_weight;
}
void init_transformer_h_20_mlp_down_proj_weight() {
  half *transformer_h_20_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_20_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_20_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_20_mlp_down_proj_weight_data,
      transformer_h_20_mlp_down_proj_weight_shape);
}

void delete_transformer_h_20_mlp_down_proj_weight() {
  delete transformer_h_20_mlp_down_proj_weight;
}

// Layer transformer.h.20.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_20_mlp_down_proj_bias;
}
void init_transformer_h_20_mlp_down_proj_bias() {
  half *transformer_h_20_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_20_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_20_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_20_mlp_down_proj_bias_data,
                                    transformer_h_20_mlp_down_proj_bias_shape);
}

void delete_transformer_h_20_mlp_down_proj_bias() {
  delete transformer_h_20_mlp_down_proj_bias;
}

// Layer transformer.h.21.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_21_input_layernorm_weight;
}
void init_transformer_h_21_input_layernorm_weight() {
  half *transformer_h_21_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_21_input_layernorm_weight_shape[1] = {5120};
  transformer_h_21_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_21_input_layernorm_weight_data,
      transformer_h_21_input_layernorm_weight_shape);
}

void delete_transformer_h_21_input_layernorm_weight() {
  delete transformer_h_21_input_layernorm_weight;
}

// Layer transformer.h.21.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_21_self_attention_query_weight;
}
void init_transformer_h_21_self_attention_query_weight() {
  half *transformer_h_21_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_21_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_21_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_21_self_attention_query_weight_data,
      transformer_h_21_self_attention_query_weight_shape);
}

void delete_transformer_h_21_self_attention_query_weight() {
  delete transformer_h_21_self_attention_query_weight;
}

// Layer transformer.h.21.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_21_self_attention_key_value_weight;
}
void init_transformer_h_21_self_attention_key_value_weight() {
  half *transformer_h_21_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_21_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_21_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_21_self_attention_key_value_weight_data,
          transformer_h_21_self_attention_key_value_weight_shape);
}

void delete_transformer_h_21_self_attention_key_value_weight() {
  delete transformer_h_21_self_attention_key_value_weight;
}

// Layer transformer.h.21.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_21_self_attention_dense_weight;
}
void init_transformer_h_21_self_attention_dense_weight() {
  half *transformer_h_21_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_21_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_21_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_21_self_attention_dense_weight_data,
      transformer_h_21_self_attention_dense_weight_shape);
}

void delete_transformer_h_21_self_attention_dense_weight() {
  delete transformer_h_21_self_attention_dense_weight;
}

// Layer transformer.h.21.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_21_self_attention_dense_bias;
}
void init_transformer_h_21_self_attention_dense_bias() {
  half *transformer_h_21_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_21_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_21_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_21_self_attention_dense_bias_data,
      transformer_h_21_self_attention_dense_bias_shape);
}

void delete_transformer_h_21_self_attention_dense_bias() {
  delete transformer_h_21_self_attention_dense_bias;
}

// Layer transformer.h.21.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_21_post_attention_layernorm_weight;
}
void init_transformer_h_21_post_attention_layernorm_weight() {
  half *transformer_h_21_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_21_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_21_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_21_post_attention_layernorm_weight_data,
          transformer_h_21_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_21_post_attention_layernorm_weight() {
  delete transformer_h_21_post_attention_layernorm_weight;
}

// Layer transformer.h.21.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_21_mlp_gate_proj_weight;
}
void init_transformer_h_21_mlp_gate_proj_weight() {
  half *transformer_h_21_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_21_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_21_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_21_mlp_gate_proj_weight_data,
      transformer_h_21_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_21_mlp_gate_proj_weight() {
  delete transformer_h_21_mlp_gate_proj_weight;
}

// Layer transformer.h.21.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_21_mlp_up_proj_weight;
}
void init_transformer_h_21_mlp_up_proj_weight() {
  half *transformer_h_21_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_21_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_21_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_21_mlp_up_proj_weight_data,
                                    transformer_h_21_mlp_up_proj_weight_shape);
}

void delete_transformer_h_21_mlp_up_proj_weight() {
  delete transformer_h_21_mlp_up_proj_weight;
}

// Layer transformer.h.21.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_21_mlp_down_proj_weight;
}
void init_transformer_h_21_mlp_down_proj_weight() {
  half *transformer_h_21_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_21_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_21_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_21_mlp_down_proj_weight_data,
      transformer_h_21_mlp_down_proj_weight_shape);
}

void delete_transformer_h_21_mlp_down_proj_weight() {
  delete transformer_h_21_mlp_down_proj_weight;
}

// Layer transformer.h.21.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_21_mlp_down_proj_bias;
}
void init_transformer_h_21_mlp_down_proj_bias() {
  half *transformer_h_21_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_21_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_21_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_21_mlp_down_proj_bias_data,
                                    transformer_h_21_mlp_down_proj_bias_shape);
}

void delete_transformer_h_21_mlp_down_proj_bias() {
  delete transformer_h_21_mlp_down_proj_bias;
}

// Layer transformer.h.22.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_22_input_layernorm_weight;
}
void init_transformer_h_22_input_layernorm_weight() {
  half *transformer_h_22_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_22_input_layernorm_weight_shape[1] = {5120};
  transformer_h_22_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_22_input_layernorm_weight_data,
      transformer_h_22_input_layernorm_weight_shape);
}

void delete_transformer_h_22_input_layernorm_weight() {
  delete transformer_h_22_input_layernorm_weight;
}

// Layer transformer.h.22.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_22_self_attention_query_weight;
}
void init_transformer_h_22_self_attention_query_weight() {
  half *transformer_h_22_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_22_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_22_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_22_self_attention_query_weight_data,
      transformer_h_22_self_attention_query_weight_shape);
}

void delete_transformer_h_22_self_attention_query_weight() {
  delete transformer_h_22_self_attention_query_weight;
}

// Layer transformer.h.22.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_22_self_attention_key_value_weight;
}
void init_transformer_h_22_self_attention_key_value_weight() {
  half *transformer_h_22_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_22_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_22_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_22_self_attention_key_value_weight_data,
          transformer_h_22_self_attention_key_value_weight_shape);
}

void delete_transformer_h_22_self_attention_key_value_weight() {
  delete transformer_h_22_self_attention_key_value_weight;
}

// Layer transformer.h.22.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_22_self_attention_dense_weight;
}
void init_transformer_h_22_self_attention_dense_weight() {
  half *transformer_h_22_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_22_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_22_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_22_self_attention_dense_weight_data,
      transformer_h_22_self_attention_dense_weight_shape);
}

void delete_transformer_h_22_self_attention_dense_weight() {
  delete transformer_h_22_self_attention_dense_weight;
}

// Layer transformer.h.22.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_22_self_attention_dense_bias;
}
void init_transformer_h_22_self_attention_dense_bias() {
  half *transformer_h_22_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_22_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_22_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_22_self_attention_dense_bias_data,
      transformer_h_22_self_attention_dense_bias_shape);
}

void delete_transformer_h_22_self_attention_dense_bias() {
  delete transformer_h_22_self_attention_dense_bias;
}

// Layer transformer.h.22.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_22_post_attention_layernorm_weight;
}
void init_transformer_h_22_post_attention_layernorm_weight() {
  half *transformer_h_22_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_22_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_22_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_22_post_attention_layernorm_weight_data,
          transformer_h_22_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_22_post_attention_layernorm_weight() {
  delete transformer_h_22_post_attention_layernorm_weight;
}

// Layer transformer.h.22.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_22_mlp_gate_proj_weight;
}
void init_transformer_h_22_mlp_gate_proj_weight() {
  half *transformer_h_22_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_22_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_22_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_22_mlp_gate_proj_weight_data,
      transformer_h_22_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_22_mlp_gate_proj_weight() {
  delete transformer_h_22_mlp_gate_proj_weight;
}

// Layer transformer.h.22.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_22_mlp_up_proj_weight;
}
void init_transformer_h_22_mlp_up_proj_weight() {
  half *transformer_h_22_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_22_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_22_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_22_mlp_up_proj_weight_data,
                                    transformer_h_22_mlp_up_proj_weight_shape);
}

void delete_transformer_h_22_mlp_up_proj_weight() {
  delete transformer_h_22_mlp_up_proj_weight;
}

// Layer transformer.h.22.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_22_mlp_down_proj_weight;
}
void init_transformer_h_22_mlp_down_proj_weight() {
  half *transformer_h_22_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_22_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_22_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_22_mlp_down_proj_weight_data,
      transformer_h_22_mlp_down_proj_weight_shape);
}

void delete_transformer_h_22_mlp_down_proj_weight() {
  delete transformer_h_22_mlp_down_proj_weight;
}

// Layer transformer.h.22.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_22_mlp_down_proj_bias;
}
void init_transformer_h_22_mlp_down_proj_bias() {
  half *transformer_h_22_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_22_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_22_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_22_mlp_down_proj_bias_data,
                                    transformer_h_22_mlp_down_proj_bias_shape);
}

void delete_transformer_h_22_mlp_down_proj_bias() {
  delete transformer_h_22_mlp_down_proj_bias;
}

// Layer transformer.h.23.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_23_input_layernorm_weight;
}
void init_transformer_h_23_input_layernorm_weight() {
  half *transformer_h_23_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_23_input_layernorm_weight_shape[1] = {5120};
  transformer_h_23_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_23_input_layernorm_weight_data,
      transformer_h_23_input_layernorm_weight_shape);
}

void delete_transformer_h_23_input_layernorm_weight() {
  delete transformer_h_23_input_layernorm_weight;
}

// Layer transformer.h.23.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_23_self_attention_query_weight;
}
void init_transformer_h_23_self_attention_query_weight() {
  half *transformer_h_23_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_23_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_23_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_23_self_attention_query_weight_data,
      transformer_h_23_self_attention_query_weight_shape);
}

void delete_transformer_h_23_self_attention_query_weight() {
  delete transformer_h_23_self_attention_query_weight;
}

// Layer transformer.h.23.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_23_self_attention_key_value_weight;
}
void init_transformer_h_23_self_attention_key_value_weight() {
  half *transformer_h_23_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_23_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_23_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_23_self_attention_key_value_weight_data,
          transformer_h_23_self_attention_key_value_weight_shape);
}

void delete_transformer_h_23_self_attention_key_value_weight() {
  delete transformer_h_23_self_attention_key_value_weight;
}

// Layer transformer.h.23.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_23_self_attention_dense_weight;
}
void init_transformer_h_23_self_attention_dense_weight() {
  half *transformer_h_23_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_23_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_23_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_23_self_attention_dense_weight_data,
      transformer_h_23_self_attention_dense_weight_shape);
}

void delete_transformer_h_23_self_attention_dense_weight() {
  delete transformer_h_23_self_attention_dense_weight;
}

// Layer transformer.h.23.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_23_self_attention_dense_bias;
}
void init_transformer_h_23_self_attention_dense_bias() {
  half *transformer_h_23_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_23_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_23_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_23_self_attention_dense_bias_data,
      transformer_h_23_self_attention_dense_bias_shape);
}

void delete_transformer_h_23_self_attention_dense_bias() {
  delete transformer_h_23_self_attention_dense_bias;
}

// Layer transformer.h.23.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_23_post_attention_layernorm_weight;
}
void init_transformer_h_23_post_attention_layernorm_weight() {
  half *transformer_h_23_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_23_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_23_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_23_post_attention_layernorm_weight_data,
          transformer_h_23_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_23_post_attention_layernorm_weight() {
  delete transformer_h_23_post_attention_layernorm_weight;
}

// Layer transformer.h.23.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_23_mlp_gate_proj_weight;
}
void init_transformer_h_23_mlp_gate_proj_weight() {
  half *transformer_h_23_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_23_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_23_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_23_mlp_gate_proj_weight_data,
      transformer_h_23_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_23_mlp_gate_proj_weight() {
  delete transformer_h_23_mlp_gate_proj_weight;
}

// Layer transformer.h.23.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_23_mlp_up_proj_weight;
}
void init_transformer_h_23_mlp_up_proj_weight() {
  half *transformer_h_23_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_23_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_23_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_23_mlp_up_proj_weight_data,
                                    transformer_h_23_mlp_up_proj_weight_shape);
}

void delete_transformer_h_23_mlp_up_proj_weight() {
  delete transformer_h_23_mlp_up_proj_weight;
}

// Layer transformer.h.23.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_23_mlp_down_proj_weight;
}
void init_transformer_h_23_mlp_down_proj_weight() {
  half *transformer_h_23_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_23_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_23_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_23_mlp_down_proj_weight_data,
      transformer_h_23_mlp_down_proj_weight_shape);
}

void delete_transformer_h_23_mlp_down_proj_weight() {
  delete transformer_h_23_mlp_down_proj_weight;
}

// Layer transformer.h.23.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_23_mlp_down_proj_bias;
}
void init_transformer_h_23_mlp_down_proj_bias() {
  half *transformer_h_23_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_23_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_23_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_23_mlp_down_proj_bias_data,
                                    transformer_h_23_mlp_down_proj_bias_shape);
}

void delete_transformer_h_23_mlp_down_proj_bias() {
  delete transformer_h_23_mlp_down_proj_bias;
}

// Layer transformer.h.24.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_24_input_layernorm_weight;
}
void init_transformer_h_24_input_layernorm_weight() {
  half *transformer_h_24_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_24_input_layernorm_weight_shape[1] = {5120};
  transformer_h_24_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_24_input_layernorm_weight_data,
      transformer_h_24_input_layernorm_weight_shape);
}

void delete_transformer_h_24_input_layernorm_weight() {
  delete transformer_h_24_input_layernorm_weight;
}

// Layer transformer.h.24.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_24_self_attention_query_weight;
}
void init_transformer_h_24_self_attention_query_weight() {
  half *transformer_h_24_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_24_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_24_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_24_self_attention_query_weight_data,
      transformer_h_24_self_attention_query_weight_shape);
}

void delete_transformer_h_24_self_attention_query_weight() {
  delete transformer_h_24_self_attention_query_weight;
}

// Layer transformer.h.24.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_24_self_attention_key_value_weight;
}
void init_transformer_h_24_self_attention_key_value_weight() {
  half *transformer_h_24_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_24_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_24_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_24_self_attention_key_value_weight_data,
          transformer_h_24_self_attention_key_value_weight_shape);
}

void delete_transformer_h_24_self_attention_key_value_weight() {
  delete transformer_h_24_self_attention_key_value_weight;
}

// Layer transformer.h.24.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_24_self_attention_dense_weight;
}
void init_transformer_h_24_self_attention_dense_weight() {
  half *transformer_h_24_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_24_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_24_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_24_self_attention_dense_weight_data,
      transformer_h_24_self_attention_dense_weight_shape);
}

void delete_transformer_h_24_self_attention_dense_weight() {
  delete transformer_h_24_self_attention_dense_weight;
}

// Layer transformer.h.24.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_24_self_attention_dense_bias;
}
void init_transformer_h_24_self_attention_dense_bias() {
  half *transformer_h_24_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_24_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_24_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_24_self_attention_dense_bias_data,
      transformer_h_24_self_attention_dense_bias_shape);
}

void delete_transformer_h_24_self_attention_dense_bias() {
  delete transformer_h_24_self_attention_dense_bias;
}

// Layer transformer.h.24.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_24_post_attention_layernorm_weight;
}
void init_transformer_h_24_post_attention_layernorm_weight() {
  half *transformer_h_24_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_24_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_24_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_24_post_attention_layernorm_weight_data,
          transformer_h_24_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_24_post_attention_layernorm_weight() {
  delete transformer_h_24_post_attention_layernorm_weight;
}

// Layer transformer.h.24.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_24_mlp_gate_proj_weight;
}
void init_transformer_h_24_mlp_gate_proj_weight() {
  half *transformer_h_24_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_24_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_24_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_24_mlp_gate_proj_weight_data,
      transformer_h_24_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_24_mlp_gate_proj_weight() {
  delete transformer_h_24_mlp_gate_proj_weight;
}

// Layer transformer.h.24.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_24_mlp_up_proj_weight;
}
void init_transformer_h_24_mlp_up_proj_weight() {
  half *transformer_h_24_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_24_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_24_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_24_mlp_up_proj_weight_data,
                                    transformer_h_24_mlp_up_proj_weight_shape);
}

void delete_transformer_h_24_mlp_up_proj_weight() {
  delete transformer_h_24_mlp_up_proj_weight;
}

// Layer transformer.h.24.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_24_mlp_down_proj_weight;
}
void init_transformer_h_24_mlp_down_proj_weight() {
  half *transformer_h_24_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_24_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_24_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_24_mlp_down_proj_weight_data,
      transformer_h_24_mlp_down_proj_weight_shape);
}

void delete_transformer_h_24_mlp_down_proj_weight() {
  delete transformer_h_24_mlp_down_proj_weight;
}

// Layer transformer.h.24.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_24_mlp_down_proj_bias;
}
void init_transformer_h_24_mlp_down_proj_bias() {
  half *transformer_h_24_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_24_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_24_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_24_mlp_down_proj_bias_data,
                                    transformer_h_24_mlp_down_proj_bias_shape);
}

void delete_transformer_h_24_mlp_down_proj_bias() {
  delete transformer_h_24_mlp_down_proj_bias;
}

// Layer transformer.h.25.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_25_input_layernorm_weight;
}
void init_transformer_h_25_input_layernorm_weight() {
  half *transformer_h_25_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_25_input_layernorm_weight_shape[1] = {5120};
  transformer_h_25_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_25_input_layernorm_weight_data,
      transformer_h_25_input_layernorm_weight_shape);
}

void delete_transformer_h_25_input_layernorm_weight() {
  delete transformer_h_25_input_layernorm_weight;
}

// Layer transformer.h.25.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_25_self_attention_query_weight;
}
void init_transformer_h_25_self_attention_query_weight() {
  half *transformer_h_25_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_25_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_25_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_25_self_attention_query_weight_data,
      transformer_h_25_self_attention_query_weight_shape);
}

void delete_transformer_h_25_self_attention_query_weight() {
  delete transformer_h_25_self_attention_query_weight;
}

// Layer transformer.h.25.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_25_self_attention_key_value_weight;
}
void init_transformer_h_25_self_attention_key_value_weight() {
  half *transformer_h_25_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_25_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_25_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_25_self_attention_key_value_weight_data,
          transformer_h_25_self_attention_key_value_weight_shape);
}

void delete_transformer_h_25_self_attention_key_value_weight() {
  delete transformer_h_25_self_attention_key_value_weight;
}

// Layer transformer.h.25.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_25_self_attention_dense_weight;
}
void init_transformer_h_25_self_attention_dense_weight() {
  half *transformer_h_25_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_25_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_25_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_25_self_attention_dense_weight_data,
      transformer_h_25_self_attention_dense_weight_shape);
}

void delete_transformer_h_25_self_attention_dense_weight() {
  delete transformer_h_25_self_attention_dense_weight;
}

// Layer transformer.h.25.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_25_self_attention_dense_bias;
}
void init_transformer_h_25_self_attention_dense_bias() {
  half *transformer_h_25_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_25_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_25_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_25_self_attention_dense_bias_data,
      transformer_h_25_self_attention_dense_bias_shape);
}

void delete_transformer_h_25_self_attention_dense_bias() {
  delete transformer_h_25_self_attention_dense_bias;
}

// Layer transformer.h.25.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_25_post_attention_layernorm_weight;
}
void init_transformer_h_25_post_attention_layernorm_weight() {
  half *transformer_h_25_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_25_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_25_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_25_post_attention_layernorm_weight_data,
          transformer_h_25_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_25_post_attention_layernorm_weight() {
  delete transformer_h_25_post_attention_layernorm_weight;
}

// Layer transformer.h.25.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_25_mlp_gate_proj_weight;
}
void init_transformer_h_25_mlp_gate_proj_weight() {
  half *transformer_h_25_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_25_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_25_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_25_mlp_gate_proj_weight_data,
      transformer_h_25_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_25_mlp_gate_proj_weight() {
  delete transformer_h_25_mlp_gate_proj_weight;
}

// Layer transformer.h.25.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_25_mlp_up_proj_weight;
}
void init_transformer_h_25_mlp_up_proj_weight() {
  half *transformer_h_25_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_25_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_25_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_25_mlp_up_proj_weight_data,
                                    transformer_h_25_mlp_up_proj_weight_shape);
}

void delete_transformer_h_25_mlp_up_proj_weight() {
  delete transformer_h_25_mlp_up_proj_weight;
}

// Layer transformer.h.25.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_25_mlp_down_proj_weight;
}
void init_transformer_h_25_mlp_down_proj_weight() {
  half *transformer_h_25_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_25_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_25_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_25_mlp_down_proj_weight_data,
      transformer_h_25_mlp_down_proj_weight_shape);
}

void delete_transformer_h_25_mlp_down_proj_weight() {
  delete transformer_h_25_mlp_down_proj_weight;
}

// Layer transformer.h.25.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_25_mlp_down_proj_bias;
}
void init_transformer_h_25_mlp_down_proj_bias() {
  half *transformer_h_25_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_25_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_25_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_25_mlp_down_proj_bias_data,
                                    transformer_h_25_mlp_down_proj_bias_shape);
}

void delete_transformer_h_25_mlp_down_proj_bias() {
  delete transformer_h_25_mlp_down_proj_bias;
}

// Layer transformer.h.26.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_26_input_layernorm_weight;
}
void init_transformer_h_26_input_layernorm_weight() {
  half *transformer_h_26_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_26_input_layernorm_weight_shape[1] = {5120};
  transformer_h_26_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_26_input_layernorm_weight_data,
      transformer_h_26_input_layernorm_weight_shape);
}

void delete_transformer_h_26_input_layernorm_weight() {
  delete transformer_h_26_input_layernorm_weight;
}

// Layer transformer.h.26.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_26_self_attention_query_weight;
}
void init_transformer_h_26_self_attention_query_weight() {
  half *transformer_h_26_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_26_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_26_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_26_self_attention_query_weight_data,
      transformer_h_26_self_attention_query_weight_shape);
}

void delete_transformer_h_26_self_attention_query_weight() {
  delete transformer_h_26_self_attention_query_weight;
}

// Layer transformer.h.26.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_26_self_attention_key_value_weight;
}
void init_transformer_h_26_self_attention_key_value_weight() {
  half *transformer_h_26_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_26_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_26_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_26_self_attention_key_value_weight_data,
          transformer_h_26_self_attention_key_value_weight_shape);
}

void delete_transformer_h_26_self_attention_key_value_weight() {
  delete transformer_h_26_self_attention_key_value_weight;
}

// Layer transformer.h.26.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_26_self_attention_dense_weight;
}
void init_transformer_h_26_self_attention_dense_weight() {
  half *transformer_h_26_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_26_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_26_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_26_self_attention_dense_weight_data,
      transformer_h_26_self_attention_dense_weight_shape);
}

void delete_transformer_h_26_self_attention_dense_weight() {
  delete transformer_h_26_self_attention_dense_weight;
}

// Layer transformer.h.26.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_26_self_attention_dense_bias;
}
void init_transformer_h_26_self_attention_dense_bias() {
  half *transformer_h_26_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_26_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_26_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_26_self_attention_dense_bias_data,
      transformer_h_26_self_attention_dense_bias_shape);
}

void delete_transformer_h_26_self_attention_dense_bias() {
  delete transformer_h_26_self_attention_dense_bias;
}

// Layer transformer.h.26.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_26_post_attention_layernorm_weight;
}
void init_transformer_h_26_post_attention_layernorm_weight() {
  half *transformer_h_26_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_26_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_26_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_26_post_attention_layernorm_weight_data,
          transformer_h_26_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_26_post_attention_layernorm_weight() {
  delete transformer_h_26_post_attention_layernorm_weight;
}

// Layer transformer.h.26.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_26_mlp_gate_proj_weight;
}
void init_transformer_h_26_mlp_gate_proj_weight() {
  half *transformer_h_26_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_26_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_26_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_26_mlp_gate_proj_weight_data,
      transformer_h_26_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_26_mlp_gate_proj_weight() {
  delete transformer_h_26_mlp_gate_proj_weight;
}

// Layer transformer.h.26.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_26_mlp_up_proj_weight;
}
void init_transformer_h_26_mlp_up_proj_weight() {
  half *transformer_h_26_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_26_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_26_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_26_mlp_up_proj_weight_data,
                                    transformer_h_26_mlp_up_proj_weight_shape);
}

void delete_transformer_h_26_mlp_up_proj_weight() {
  delete transformer_h_26_mlp_up_proj_weight;
}

// Layer transformer.h.26.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_26_mlp_down_proj_weight;
}
void init_transformer_h_26_mlp_down_proj_weight() {
  half *transformer_h_26_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_26_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_26_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_26_mlp_down_proj_weight_data,
      transformer_h_26_mlp_down_proj_weight_shape);
}

void delete_transformer_h_26_mlp_down_proj_weight() {
  delete transformer_h_26_mlp_down_proj_weight;
}

// Layer transformer.h.26.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_26_mlp_down_proj_bias;
}
void init_transformer_h_26_mlp_down_proj_bias() {
  half *transformer_h_26_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_26_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_26_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_26_mlp_down_proj_bias_data,
                                    transformer_h_26_mlp_down_proj_bias_shape);
}

void delete_transformer_h_26_mlp_down_proj_bias() {
  delete transformer_h_26_mlp_down_proj_bias;
}

// Layer transformer.h.27.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_27_input_layernorm_weight;
}
void init_transformer_h_27_input_layernorm_weight() {
  half *transformer_h_27_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_27_input_layernorm_weight_shape[1] = {5120};
  transformer_h_27_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_27_input_layernorm_weight_data,
      transformer_h_27_input_layernorm_weight_shape);
}

void delete_transformer_h_27_input_layernorm_weight() {
  delete transformer_h_27_input_layernorm_weight;
}

// Layer transformer.h.27.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_27_self_attention_query_weight;
}
void init_transformer_h_27_self_attention_query_weight() {
  half *transformer_h_27_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_27_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_27_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_27_self_attention_query_weight_data,
      transformer_h_27_self_attention_query_weight_shape);
}

void delete_transformer_h_27_self_attention_query_weight() {
  delete transformer_h_27_self_attention_query_weight;
}

// Layer transformer.h.27.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_27_self_attention_key_value_weight;
}
void init_transformer_h_27_self_attention_key_value_weight() {
  half *transformer_h_27_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_27_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_27_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_27_self_attention_key_value_weight_data,
          transformer_h_27_self_attention_key_value_weight_shape);
}

void delete_transformer_h_27_self_attention_key_value_weight() {
  delete transformer_h_27_self_attention_key_value_weight;
}

// Layer transformer.h.27.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_27_self_attention_dense_weight;
}
void init_transformer_h_27_self_attention_dense_weight() {
  half *transformer_h_27_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_27_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_27_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_27_self_attention_dense_weight_data,
      transformer_h_27_self_attention_dense_weight_shape);
}

void delete_transformer_h_27_self_attention_dense_weight() {
  delete transformer_h_27_self_attention_dense_weight;
}

// Layer transformer.h.27.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_27_self_attention_dense_bias;
}
void init_transformer_h_27_self_attention_dense_bias() {
  half *transformer_h_27_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_27_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_27_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_27_self_attention_dense_bias_data,
      transformer_h_27_self_attention_dense_bias_shape);
}

void delete_transformer_h_27_self_attention_dense_bias() {
  delete transformer_h_27_self_attention_dense_bias;
}

// Layer transformer.h.27.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_27_post_attention_layernorm_weight;
}
void init_transformer_h_27_post_attention_layernorm_weight() {
  half *transformer_h_27_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_27_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_27_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_27_post_attention_layernorm_weight_data,
          transformer_h_27_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_27_post_attention_layernorm_weight() {
  delete transformer_h_27_post_attention_layernorm_weight;
}

// Layer transformer.h.27.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_27_mlp_gate_proj_weight;
}
void init_transformer_h_27_mlp_gate_proj_weight() {
  half *transformer_h_27_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_27_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_27_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_27_mlp_gate_proj_weight_data,
      transformer_h_27_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_27_mlp_gate_proj_weight() {
  delete transformer_h_27_mlp_gate_proj_weight;
}

// Layer transformer.h.27.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_27_mlp_up_proj_weight;
}
void init_transformer_h_27_mlp_up_proj_weight() {
  half *transformer_h_27_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_27_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_27_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_27_mlp_up_proj_weight_data,
                                    transformer_h_27_mlp_up_proj_weight_shape);
}

void delete_transformer_h_27_mlp_up_proj_weight() {
  delete transformer_h_27_mlp_up_proj_weight;
}

// Layer transformer.h.27.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_27_mlp_down_proj_weight;
}
void init_transformer_h_27_mlp_down_proj_weight() {
  half *transformer_h_27_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_27_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_27_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_27_mlp_down_proj_weight_data,
      transformer_h_27_mlp_down_proj_weight_shape);
}

void delete_transformer_h_27_mlp_down_proj_weight() {
  delete transformer_h_27_mlp_down_proj_weight;
}

// Layer transformer.h.27.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_27_mlp_down_proj_bias;
}
void init_transformer_h_27_mlp_down_proj_bias() {
  half *transformer_h_27_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_27_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_27_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_27_mlp_down_proj_bias_data,
                                    transformer_h_27_mlp_down_proj_bias_shape);
}

void delete_transformer_h_27_mlp_down_proj_bias() {
  delete transformer_h_27_mlp_down_proj_bias;
}

// Layer transformer.h.28.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_28_input_layernorm_weight;
}
void init_transformer_h_28_input_layernorm_weight() {
  half *transformer_h_28_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_28_input_layernorm_weight_shape[1] = {5120};
  transformer_h_28_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_28_input_layernorm_weight_data,
      transformer_h_28_input_layernorm_weight_shape);
}

void delete_transformer_h_28_input_layernorm_weight() {
  delete transformer_h_28_input_layernorm_weight;
}

// Layer transformer.h.28.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_28_self_attention_query_weight;
}
void init_transformer_h_28_self_attention_query_weight() {
  half *transformer_h_28_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_28_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_28_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_28_self_attention_query_weight_data,
      transformer_h_28_self_attention_query_weight_shape);
}

void delete_transformer_h_28_self_attention_query_weight() {
  delete transformer_h_28_self_attention_query_weight;
}

// Layer transformer.h.28.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_28_self_attention_key_value_weight;
}
void init_transformer_h_28_self_attention_key_value_weight() {
  half *transformer_h_28_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_28_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_28_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_28_self_attention_key_value_weight_data,
          transformer_h_28_self_attention_key_value_weight_shape);
}

void delete_transformer_h_28_self_attention_key_value_weight() {
  delete transformer_h_28_self_attention_key_value_weight;
}

// Layer transformer.h.28.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_28_self_attention_dense_weight;
}
void init_transformer_h_28_self_attention_dense_weight() {
  half *transformer_h_28_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_28_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_28_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_28_self_attention_dense_weight_data,
      transformer_h_28_self_attention_dense_weight_shape);
}

void delete_transformer_h_28_self_attention_dense_weight() {
  delete transformer_h_28_self_attention_dense_weight;
}

// Layer transformer.h.28.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_28_self_attention_dense_bias;
}
void init_transformer_h_28_self_attention_dense_bias() {
  half *transformer_h_28_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_28_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_28_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_28_self_attention_dense_bias_data,
      transformer_h_28_self_attention_dense_bias_shape);
}

void delete_transformer_h_28_self_attention_dense_bias() {
  delete transformer_h_28_self_attention_dense_bias;
}

// Layer transformer.h.28.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_28_post_attention_layernorm_weight;
}
void init_transformer_h_28_post_attention_layernorm_weight() {
  half *transformer_h_28_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_28_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_28_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_28_post_attention_layernorm_weight_data,
          transformer_h_28_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_28_post_attention_layernorm_weight() {
  delete transformer_h_28_post_attention_layernorm_weight;
}

// Layer transformer.h.28.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_28_mlp_gate_proj_weight;
}
void init_transformer_h_28_mlp_gate_proj_weight() {
  half *transformer_h_28_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_28_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_28_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_28_mlp_gate_proj_weight_data,
      transformer_h_28_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_28_mlp_gate_proj_weight() {
  delete transformer_h_28_mlp_gate_proj_weight;
}

// Layer transformer.h.28.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_28_mlp_up_proj_weight;
}
void init_transformer_h_28_mlp_up_proj_weight() {
  half *transformer_h_28_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_28_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_28_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_28_mlp_up_proj_weight_data,
                                    transformer_h_28_mlp_up_proj_weight_shape);
}

void delete_transformer_h_28_mlp_up_proj_weight() {
  delete transformer_h_28_mlp_up_proj_weight;
}

// Layer transformer.h.28.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_28_mlp_down_proj_weight;
}
void init_transformer_h_28_mlp_down_proj_weight() {
  half *transformer_h_28_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_28_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_28_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_28_mlp_down_proj_weight_data,
      transformer_h_28_mlp_down_proj_weight_shape);
}

void delete_transformer_h_28_mlp_down_proj_weight() {
  delete transformer_h_28_mlp_down_proj_weight;
}

// Layer transformer.h.28.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_28_mlp_down_proj_bias;
}
void init_transformer_h_28_mlp_down_proj_bias() {
  half *transformer_h_28_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_28_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_28_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_28_mlp_down_proj_bias_data,
                                    transformer_h_28_mlp_down_proj_bias_shape);
}

void delete_transformer_h_28_mlp_down_proj_bias() {
  delete transformer_h_28_mlp_down_proj_bias;
}

// Layer transformer.h.29.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_29_input_layernorm_weight;
}
void init_transformer_h_29_input_layernorm_weight() {
  half *transformer_h_29_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_29_input_layernorm_weight_shape[1] = {5120};
  transformer_h_29_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_29_input_layernorm_weight_data,
      transformer_h_29_input_layernorm_weight_shape);
}

void delete_transformer_h_29_input_layernorm_weight() {
  delete transformer_h_29_input_layernorm_weight;
}

// Layer transformer.h.29.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_29_self_attention_query_weight;
}
void init_transformer_h_29_self_attention_query_weight() {
  half *transformer_h_29_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_29_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_29_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_29_self_attention_query_weight_data,
      transformer_h_29_self_attention_query_weight_shape);
}

void delete_transformer_h_29_self_attention_query_weight() {
  delete transformer_h_29_self_attention_query_weight;
}

// Layer transformer.h.29.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_29_self_attention_key_value_weight;
}
void init_transformer_h_29_self_attention_key_value_weight() {
  half *transformer_h_29_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_29_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_29_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_29_self_attention_key_value_weight_data,
          transformer_h_29_self_attention_key_value_weight_shape);
}

void delete_transformer_h_29_self_attention_key_value_weight() {
  delete transformer_h_29_self_attention_key_value_weight;
}

// Layer transformer.h.29.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_29_self_attention_dense_weight;
}
void init_transformer_h_29_self_attention_dense_weight() {
  half *transformer_h_29_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_29_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_29_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_29_self_attention_dense_weight_data,
      transformer_h_29_self_attention_dense_weight_shape);
}

void delete_transformer_h_29_self_attention_dense_weight() {
  delete transformer_h_29_self_attention_dense_weight;
}

// Layer transformer.h.29.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_29_self_attention_dense_bias;
}
void init_transformer_h_29_self_attention_dense_bias() {
  half *transformer_h_29_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_29_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_29_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_29_self_attention_dense_bias_data,
      transformer_h_29_self_attention_dense_bias_shape);
}

void delete_transformer_h_29_self_attention_dense_bias() {
  delete transformer_h_29_self_attention_dense_bias;
}

// Layer transformer.h.29.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_29_post_attention_layernorm_weight;
}
void init_transformer_h_29_post_attention_layernorm_weight() {
  half *transformer_h_29_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_29_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_29_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_29_post_attention_layernorm_weight_data,
          transformer_h_29_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_29_post_attention_layernorm_weight() {
  delete transformer_h_29_post_attention_layernorm_weight;
}

// Layer transformer.h.29.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_29_mlp_gate_proj_weight;
}
void init_transformer_h_29_mlp_gate_proj_weight() {
  half *transformer_h_29_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_29_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_29_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_29_mlp_gate_proj_weight_data,
      transformer_h_29_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_29_mlp_gate_proj_weight() {
  delete transformer_h_29_mlp_gate_proj_weight;
}

// Layer transformer.h.29.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_29_mlp_up_proj_weight;
}
void init_transformer_h_29_mlp_up_proj_weight() {
  half *transformer_h_29_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_29_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_29_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_29_mlp_up_proj_weight_data,
                                    transformer_h_29_mlp_up_proj_weight_shape);
}

void delete_transformer_h_29_mlp_up_proj_weight() {
  delete transformer_h_29_mlp_up_proj_weight;
}

// Layer transformer.h.29.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_29_mlp_down_proj_weight;
}
void init_transformer_h_29_mlp_down_proj_weight() {
  half *transformer_h_29_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_29_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_29_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_29_mlp_down_proj_weight_data,
      transformer_h_29_mlp_down_proj_weight_shape);
}

void delete_transformer_h_29_mlp_down_proj_weight() {
  delete transformer_h_29_mlp_down_proj_weight;
}

// Layer transformer.h.29.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_29_mlp_down_proj_bias;
}
void init_transformer_h_29_mlp_down_proj_bias() {
  half *transformer_h_29_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_29_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_29_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_29_mlp_down_proj_bias_data,
                                    transformer_h_29_mlp_down_proj_bias_shape);
}

void delete_transformer_h_29_mlp_down_proj_bias() {
  delete transformer_h_29_mlp_down_proj_bias;
}

// Layer transformer.h.30.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_30_input_layernorm_weight;
}
void init_transformer_h_30_input_layernorm_weight() {
  half *transformer_h_30_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_30_input_layernorm_weight_shape[1] = {5120};
  transformer_h_30_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_30_input_layernorm_weight_data,
      transformer_h_30_input_layernorm_weight_shape);
}

void delete_transformer_h_30_input_layernorm_weight() {
  delete transformer_h_30_input_layernorm_weight;
}

// Layer transformer.h.30.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_30_self_attention_query_weight;
}
void init_transformer_h_30_self_attention_query_weight() {
  half *transformer_h_30_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_30_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_30_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_30_self_attention_query_weight_data,
      transformer_h_30_self_attention_query_weight_shape);
}

void delete_transformer_h_30_self_attention_query_weight() {
  delete transformer_h_30_self_attention_query_weight;
}

// Layer transformer.h.30.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_30_self_attention_key_value_weight;
}
void init_transformer_h_30_self_attention_key_value_weight() {
  half *transformer_h_30_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_30_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_30_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_30_self_attention_key_value_weight_data,
          transformer_h_30_self_attention_key_value_weight_shape);
}

void delete_transformer_h_30_self_attention_key_value_weight() {
  delete transformer_h_30_self_attention_key_value_weight;
}

// Layer transformer.h.30.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_30_self_attention_dense_weight;
}
void init_transformer_h_30_self_attention_dense_weight() {
  half *transformer_h_30_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_30_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_30_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_30_self_attention_dense_weight_data,
      transformer_h_30_self_attention_dense_weight_shape);
}

void delete_transformer_h_30_self_attention_dense_weight() {
  delete transformer_h_30_self_attention_dense_weight;
}

// Layer transformer.h.30.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_30_self_attention_dense_bias;
}
void init_transformer_h_30_self_attention_dense_bias() {
  half *transformer_h_30_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_30_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_30_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_30_self_attention_dense_bias_data,
      transformer_h_30_self_attention_dense_bias_shape);
}

void delete_transformer_h_30_self_attention_dense_bias() {
  delete transformer_h_30_self_attention_dense_bias;
}

// Layer transformer.h.30.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_30_post_attention_layernorm_weight;
}
void init_transformer_h_30_post_attention_layernorm_weight() {
  half *transformer_h_30_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_30_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_30_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_30_post_attention_layernorm_weight_data,
          transformer_h_30_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_30_post_attention_layernorm_weight() {
  delete transformer_h_30_post_attention_layernorm_weight;
}

// Layer transformer.h.30.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_30_mlp_gate_proj_weight;
}
void init_transformer_h_30_mlp_gate_proj_weight() {
  half *transformer_h_30_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_30_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_30_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_30_mlp_gate_proj_weight_data,
      transformer_h_30_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_30_mlp_gate_proj_weight() {
  delete transformer_h_30_mlp_gate_proj_weight;
}

// Layer transformer.h.30.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_30_mlp_up_proj_weight;
}
void init_transformer_h_30_mlp_up_proj_weight() {
  half *transformer_h_30_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_30_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_30_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_30_mlp_up_proj_weight_data,
                                    transformer_h_30_mlp_up_proj_weight_shape);
}

void delete_transformer_h_30_mlp_up_proj_weight() {
  delete transformer_h_30_mlp_up_proj_weight;
}

// Layer transformer.h.30.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_30_mlp_down_proj_weight;
}
void init_transformer_h_30_mlp_down_proj_weight() {
  half *transformer_h_30_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_30_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_30_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_30_mlp_down_proj_weight_data,
      transformer_h_30_mlp_down_proj_weight_shape);
}

void delete_transformer_h_30_mlp_down_proj_weight() {
  delete transformer_h_30_mlp_down_proj_weight;
}

// Layer transformer.h.30.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_30_mlp_down_proj_bias;
}
void init_transformer_h_30_mlp_down_proj_bias() {
  half *transformer_h_30_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_30_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_30_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_30_mlp_down_proj_bias_data,
                                    transformer_h_30_mlp_down_proj_bias_shape);
}

void delete_transformer_h_30_mlp_down_proj_bias() {
  delete transformer_h_30_mlp_down_proj_bias;
}

// Layer transformer.h.31.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_31_input_layernorm_weight;
}
void init_transformer_h_31_input_layernorm_weight() {
  half *transformer_h_31_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_31_input_layernorm_weight_shape[1] = {5120};
  transformer_h_31_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_31_input_layernorm_weight_data,
      transformer_h_31_input_layernorm_weight_shape);
}

void delete_transformer_h_31_input_layernorm_weight() {
  delete transformer_h_31_input_layernorm_weight;
}

// Layer transformer.h.31.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_31_self_attention_query_weight;
}
void init_transformer_h_31_self_attention_query_weight() {
  half *transformer_h_31_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_31_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_31_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_31_self_attention_query_weight_data,
      transformer_h_31_self_attention_query_weight_shape);
}

void delete_transformer_h_31_self_attention_query_weight() {
  delete transformer_h_31_self_attention_query_weight;
}

// Layer transformer.h.31.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_31_self_attention_key_value_weight;
}
void init_transformer_h_31_self_attention_key_value_weight() {
  half *transformer_h_31_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_31_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_31_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_31_self_attention_key_value_weight_data,
          transformer_h_31_self_attention_key_value_weight_shape);
}

void delete_transformer_h_31_self_attention_key_value_weight() {
  delete transformer_h_31_self_attention_key_value_weight;
}

// Layer transformer.h.31.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_31_self_attention_dense_weight;
}
void init_transformer_h_31_self_attention_dense_weight() {
  half *transformer_h_31_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_31_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_31_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_31_self_attention_dense_weight_data,
      transformer_h_31_self_attention_dense_weight_shape);
}

void delete_transformer_h_31_self_attention_dense_weight() {
  delete transformer_h_31_self_attention_dense_weight;
}

// Layer transformer.h.31.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_31_self_attention_dense_bias;
}
void init_transformer_h_31_self_attention_dense_bias() {
  half *transformer_h_31_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_31_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_31_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_31_self_attention_dense_bias_data,
      transformer_h_31_self_attention_dense_bias_shape);
}

void delete_transformer_h_31_self_attention_dense_bias() {
  delete transformer_h_31_self_attention_dense_bias;
}

// Layer transformer.h.31.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_31_post_attention_layernorm_weight;
}
void init_transformer_h_31_post_attention_layernorm_weight() {
  half *transformer_h_31_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_31_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_31_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_31_post_attention_layernorm_weight_data,
          transformer_h_31_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_31_post_attention_layernorm_weight() {
  delete transformer_h_31_post_attention_layernorm_weight;
}

// Layer transformer.h.31.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_31_mlp_gate_proj_weight;
}
void init_transformer_h_31_mlp_gate_proj_weight() {
  half *transformer_h_31_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_31_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_31_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_31_mlp_gate_proj_weight_data,
      transformer_h_31_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_31_mlp_gate_proj_weight() {
  delete transformer_h_31_mlp_gate_proj_weight;
}

// Layer transformer.h.31.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_31_mlp_up_proj_weight;
}
void init_transformer_h_31_mlp_up_proj_weight() {
  half *transformer_h_31_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_31_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_31_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_31_mlp_up_proj_weight_data,
                                    transformer_h_31_mlp_up_proj_weight_shape);
}

void delete_transformer_h_31_mlp_up_proj_weight() {
  delete transformer_h_31_mlp_up_proj_weight;
}

// Layer transformer.h.31.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_31_mlp_down_proj_weight;
}
void init_transformer_h_31_mlp_down_proj_weight() {
  half *transformer_h_31_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_31_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_31_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_31_mlp_down_proj_weight_data,
      transformer_h_31_mlp_down_proj_weight_shape);
}

void delete_transformer_h_31_mlp_down_proj_weight() {
  delete transformer_h_31_mlp_down_proj_weight;
}

// Layer transformer.h.31.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_31_mlp_down_proj_bias;
}
void init_transformer_h_31_mlp_down_proj_bias() {
  half *transformer_h_31_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_31_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_31_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_31_mlp_down_proj_bias_data,
                                    transformer_h_31_mlp_down_proj_bias_shape);
}

void delete_transformer_h_31_mlp_down_proj_bias() {
  delete transformer_h_31_mlp_down_proj_bias;
}

// Layer transformer.h.32.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_32_input_layernorm_weight;
}
void init_transformer_h_32_input_layernorm_weight() {
  half *transformer_h_32_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_32_input_layernorm_weight_shape[1] = {5120};
  transformer_h_32_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_32_input_layernorm_weight_data,
      transformer_h_32_input_layernorm_weight_shape);
}

void delete_transformer_h_32_input_layernorm_weight() {
  delete transformer_h_32_input_layernorm_weight;
}

// Layer transformer.h.32.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_32_self_attention_query_weight;
}
void init_transformer_h_32_self_attention_query_weight() {
  half *transformer_h_32_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_32_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_32_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_32_self_attention_query_weight_data,
      transformer_h_32_self_attention_query_weight_shape);
}

void delete_transformer_h_32_self_attention_query_weight() {
  delete transformer_h_32_self_attention_query_weight;
}

// Layer transformer.h.32.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_32_self_attention_key_value_weight;
}
void init_transformer_h_32_self_attention_key_value_weight() {
  half *transformer_h_32_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_32_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_32_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_32_self_attention_key_value_weight_data,
          transformer_h_32_self_attention_key_value_weight_shape);
}

void delete_transformer_h_32_self_attention_key_value_weight() {
  delete transformer_h_32_self_attention_key_value_weight;
}

// Layer transformer.h.32.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_32_self_attention_dense_weight;
}
void init_transformer_h_32_self_attention_dense_weight() {
  half *transformer_h_32_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_32_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_32_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_32_self_attention_dense_weight_data,
      transformer_h_32_self_attention_dense_weight_shape);
}

void delete_transformer_h_32_self_attention_dense_weight() {
  delete transformer_h_32_self_attention_dense_weight;
}

// Layer transformer.h.32.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_32_self_attention_dense_bias;
}
void init_transformer_h_32_self_attention_dense_bias() {
  half *transformer_h_32_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_32_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_32_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_32_self_attention_dense_bias_data,
      transformer_h_32_self_attention_dense_bias_shape);
}

void delete_transformer_h_32_self_attention_dense_bias() {
  delete transformer_h_32_self_attention_dense_bias;
}

// Layer transformer.h.32.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_32_post_attention_layernorm_weight;
}
void init_transformer_h_32_post_attention_layernorm_weight() {
  half *transformer_h_32_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_32_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_32_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_32_post_attention_layernorm_weight_data,
          transformer_h_32_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_32_post_attention_layernorm_weight() {
  delete transformer_h_32_post_attention_layernorm_weight;
}

// Layer transformer.h.32.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_32_mlp_gate_proj_weight;
}
void init_transformer_h_32_mlp_gate_proj_weight() {
  half *transformer_h_32_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_32_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_32_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_32_mlp_gate_proj_weight_data,
      transformer_h_32_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_32_mlp_gate_proj_weight() {
  delete transformer_h_32_mlp_gate_proj_weight;
}

// Layer transformer.h.32.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_32_mlp_up_proj_weight;
}
void init_transformer_h_32_mlp_up_proj_weight() {
  half *transformer_h_32_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_32_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_32_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_32_mlp_up_proj_weight_data,
                                    transformer_h_32_mlp_up_proj_weight_shape);
}

void delete_transformer_h_32_mlp_up_proj_weight() {
  delete transformer_h_32_mlp_up_proj_weight;
}

// Layer transformer.h.32.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_32_mlp_down_proj_weight;
}
void init_transformer_h_32_mlp_down_proj_weight() {
  half *transformer_h_32_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_32_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_32_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_32_mlp_down_proj_weight_data,
      transformer_h_32_mlp_down_proj_weight_shape);
}

void delete_transformer_h_32_mlp_down_proj_weight() {
  delete transformer_h_32_mlp_down_proj_weight;
}

// Layer transformer.h.32.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_32_mlp_down_proj_bias;
}
void init_transformer_h_32_mlp_down_proj_bias() {
  half *transformer_h_32_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_32_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_32_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_32_mlp_down_proj_bias_data,
                                    transformer_h_32_mlp_down_proj_bias_shape);
}

void delete_transformer_h_32_mlp_down_proj_bias() {
  delete transformer_h_32_mlp_down_proj_bias;
}

// Layer transformer.h.33.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_33_input_layernorm_weight;
}
void init_transformer_h_33_input_layernorm_weight() {
  half *transformer_h_33_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_33_input_layernorm_weight_shape[1] = {5120};
  transformer_h_33_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_33_input_layernorm_weight_data,
      transformer_h_33_input_layernorm_weight_shape);
}

void delete_transformer_h_33_input_layernorm_weight() {
  delete transformer_h_33_input_layernorm_weight;
}

// Layer transformer.h.33.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_33_self_attention_query_weight;
}
void init_transformer_h_33_self_attention_query_weight() {
  half *transformer_h_33_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_33_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_33_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_33_self_attention_query_weight_data,
      transformer_h_33_self_attention_query_weight_shape);
}

void delete_transformer_h_33_self_attention_query_weight() {
  delete transformer_h_33_self_attention_query_weight;
}

// Layer transformer.h.33.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_33_self_attention_key_value_weight;
}
void init_transformer_h_33_self_attention_key_value_weight() {
  half *transformer_h_33_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_33_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_33_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_33_self_attention_key_value_weight_data,
          transformer_h_33_self_attention_key_value_weight_shape);
}

void delete_transformer_h_33_self_attention_key_value_weight() {
  delete transformer_h_33_self_attention_key_value_weight;
}

// Layer transformer.h.33.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_33_self_attention_dense_weight;
}
void init_transformer_h_33_self_attention_dense_weight() {
  half *transformer_h_33_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_33_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_33_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_33_self_attention_dense_weight_data,
      transformer_h_33_self_attention_dense_weight_shape);
}

void delete_transformer_h_33_self_attention_dense_weight() {
  delete transformer_h_33_self_attention_dense_weight;
}

// Layer transformer.h.33.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_33_self_attention_dense_bias;
}
void init_transformer_h_33_self_attention_dense_bias() {
  half *transformer_h_33_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_33_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_33_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_33_self_attention_dense_bias_data,
      transformer_h_33_self_attention_dense_bias_shape);
}

void delete_transformer_h_33_self_attention_dense_bias() {
  delete transformer_h_33_self_attention_dense_bias;
}

// Layer transformer.h.33.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_33_post_attention_layernorm_weight;
}
void init_transformer_h_33_post_attention_layernorm_weight() {
  half *transformer_h_33_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_33_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_33_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_33_post_attention_layernorm_weight_data,
          transformer_h_33_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_33_post_attention_layernorm_weight() {
  delete transformer_h_33_post_attention_layernorm_weight;
}

// Layer transformer.h.33.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_33_mlp_gate_proj_weight;
}
void init_transformer_h_33_mlp_gate_proj_weight() {
  half *transformer_h_33_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_33_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_33_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_33_mlp_gate_proj_weight_data,
      transformer_h_33_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_33_mlp_gate_proj_weight() {
  delete transformer_h_33_mlp_gate_proj_weight;
}

// Layer transformer.h.33.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_33_mlp_up_proj_weight;
}
void init_transformer_h_33_mlp_up_proj_weight() {
  half *transformer_h_33_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_33_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_33_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_33_mlp_up_proj_weight_data,
                                    transformer_h_33_mlp_up_proj_weight_shape);
}

void delete_transformer_h_33_mlp_up_proj_weight() {
  delete transformer_h_33_mlp_up_proj_weight;
}

// Layer transformer.h.33.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_33_mlp_down_proj_weight;
}
void init_transformer_h_33_mlp_down_proj_weight() {
  half *transformer_h_33_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_33_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_33_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_33_mlp_down_proj_weight_data,
      transformer_h_33_mlp_down_proj_weight_shape);
}

void delete_transformer_h_33_mlp_down_proj_weight() {
  delete transformer_h_33_mlp_down_proj_weight;
}

// Layer transformer.h.33.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_33_mlp_down_proj_bias;
}
void init_transformer_h_33_mlp_down_proj_bias() {
  half *transformer_h_33_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_33_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_33_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_33_mlp_down_proj_bias_data,
                                    transformer_h_33_mlp_down_proj_bias_shape);
}

void delete_transformer_h_33_mlp_down_proj_bias() {
  delete transformer_h_33_mlp_down_proj_bias;
}

// Layer transformer.h.34.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_34_input_layernorm_weight;
}
void init_transformer_h_34_input_layernorm_weight() {
  half *transformer_h_34_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_34_input_layernorm_weight_shape[1] = {5120};
  transformer_h_34_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_34_input_layernorm_weight_data,
      transformer_h_34_input_layernorm_weight_shape);
}

void delete_transformer_h_34_input_layernorm_weight() {
  delete transformer_h_34_input_layernorm_weight;
}

// Layer transformer.h.34.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_34_self_attention_query_weight;
}
void init_transformer_h_34_self_attention_query_weight() {
  half *transformer_h_34_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_34_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_34_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_34_self_attention_query_weight_data,
      transformer_h_34_self_attention_query_weight_shape);
}

void delete_transformer_h_34_self_attention_query_weight() {
  delete transformer_h_34_self_attention_query_weight;
}

// Layer transformer.h.34.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_34_self_attention_key_value_weight;
}
void init_transformer_h_34_self_attention_key_value_weight() {
  half *transformer_h_34_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_34_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_34_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_34_self_attention_key_value_weight_data,
          transformer_h_34_self_attention_key_value_weight_shape);
}

void delete_transformer_h_34_self_attention_key_value_weight() {
  delete transformer_h_34_self_attention_key_value_weight;
}

// Layer transformer.h.34.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_34_self_attention_dense_weight;
}
void init_transformer_h_34_self_attention_dense_weight() {
  half *transformer_h_34_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_34_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_34_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_34_self_attention_dense_weight_data,
      transformer_h_34_self_attention_dense_weight_shape);
}

void delete_transformer_h_34_self_attention_dense_weight() {
  delete transformer_h_34_self_attention_dense_weight;
}

// Layer transformer.h.34.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_34_self_attention_dense_bias;
}
void init_transformer_h_34_self_attention_dense_bias() {
  half *transformer_h_34_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_34_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_34_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_34_self_attention_dense_bias_data,
      transformer_h_34_self_attention_dense_bias_shape);
}

void delete_transformer_h_34_self_attention_dense_bias() {
  delete transformer_h_34_self_attention_dense_bias;
}

// Layer transformer.h.34.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_34_post_attention_layernorm_weight;
}
void init_transformer_h_34_post_attention_layernorm_weight() {
  half *transformer_h_34_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_34_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_34_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_34_post_attention_layernorm_weight_data,
          transformer_h_34_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_34_post_attention_layernorm_weight() {
  delete transformer_h_34_post_attention_layernorm_weight;
}

// Layer transformer.h.34.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_34_mlp_gate_proj_weight;
}
void init_transformer_h_34_mlp_gate_proj_weight() {
  half *transformer_h_34_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_34_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_34_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_34_mlp_gate_proj_weight_data,
      transformer_h_34_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_34_mlp_gate_proj_weight() {
  delete transformer_h_34_mlp_gate_proj_weight;
}

// Layer transformer.h.34.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_34_mlp_up_proj_weight;
}
void init_transformer_h_34_mlp_up_proj_weight() {
  half *transformer_h_34_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_34_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_34_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_34_mlp_up_proj_weight_data,
                                    transformer_h_34_mlp_up_proj_weight_shape);
}

void delete_transformer_h_34_mlp_up_proj_weight() {
  delete transformer_h_34_mlp_up_proj_weight;
}

// Layer transformer.h.34.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_34_mlp_down_proj_weight;
}
void init_transformer_h_34_mlp_down_proj_weight() {
  half *transformer_h_34_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_34_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_34_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_34_mlp_down_proj_weight_data,
      transformer_h_34_mlp_down_proj_weight_shape);
}

void delete_transformer_h_34_mlp_down_proj_weight() {
  delete transformer_h_34_mlp_down_proj_weight;
}

// Layer transformer.h.34.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_34_mlp_down_proj_bias;
}
void init_transformer_h_34_mlp_down_proj_bias() {
  half *transformer_h_34_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_34_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_34_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_34_mlp_down_proj_bias_data,
                                    transformer_h_34_mlp_down_proj_bias_shape);
}

void delete_transformer_h_34_mlp_down_proj_bias() {
  delete transformer_h_34_mlp_down_proj_bias;
}

// Layer transformer.h.35.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_35_input_layernorm_weight;
}
void init_transformer_h_35_input_layernorm_weight() {
  half *transformer_h_35_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_35_input_layernorm_weight_shape[1] = {5120};
  transformer_h_35_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_35_input_layernorm_weight_data,
      transformer_h_35_input_layernorm_weight_shape);
}

void delete_transformer_h_35_input_layernorm_weight() {
  delete transformer_h_35_input_layernorm_weight;
}

// Layer transformer.h.35.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_35_self_attention_query_weight;
}
void init_transformer_h_35_self_attention_query_weight() {
  half *transformer_h_35_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_35_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_35_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_35_self_attention_query_weight_data,
      transformer_h_35_self_attention_query_weight_shape);
}

void delete_transformer_h_35_self_attention_query_weight() {
  delete transformer_h_35_self_attention_query_weight;
}

// Layer transformer.h.35.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_35_self_attention_key_value_weight;
}
void init_transformer_h_35_self_attention_key_value_weight() {
  half *transformer_h_35_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_35_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_35_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_35_self_attention_key_value_weight_data,
          transformer_h_35_self_attention_key_value_weight_shape);
}

void delete_transformer_h_35_self_attention_key_value_weight() {
  delete transformer_h_35_self_attention_key_value_weight;
}

// Layer transformer.h.35.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_35_self_attention_dense_weight;
}
void init_transformer_h_35_self_attention_dense_weight() {
  half *transformer_h_35_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_35_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_35_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_35_self_attention_dense_weight_data,
      transformer_h_35_self_attention_dense_weight_shape);
}

void delete_transformer_h_35_self_attention_dense_weight() {
  delete transformer_h_35_self_attention_dense_weight;
}

// Layer transformer.h.35.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_35_self_attention_dense_bias;
}
void init_transformer_h_35_self_attention_dense_bias() {
  half *transformer_h_35_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_35_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_35_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_35_self_attention_dense_bias_data,
      transformer_h_35_self_attention_dense_bias_shape);
}

void delete_transformer_h_35_self_attention_dense_bias() {
  delete transformer_h_35_self_attention_dense_bias;
}

// Layer transformer.h.35.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_35_post_attention_layernorm_weight;
}
void init_transformer_h_35_post_attention_layernorm_weight() {
  half *transformer_h_35_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_35_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_35_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_35_post_attention_layernorm_weight_data,
          transformer_h_35_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_35_post_attention_layernorm_weight() {
  delete transformer_h_35_post_attention_layernorm_weight;
}

// Layer transformer.h.35.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_35_mlp_gate_proj_weight;
}
void init_transformer_h_35_mlp_gate_proj_weight() {
  half *transformer_h_35_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_35_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_35_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_35_mlp_gate_proj_weight_data,
      transformer_h_35_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_35_mlp_gate_proj_weight() {
  delete transformer_h_35_mlp_gate_proj_weight;
}

// Layer transformer.h.35.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_35_mlp_up_proj_weight;
}
void init_transformer_h_35_mlp_up_proj_weight() {
  half *transformer_h_35_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_35_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_35_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_35_mlp_up_proj_weight_data,
                                    transformer_h_35_mlp_up_proj_weight_shape);
}

void delete_transformer_h_35_mlp_up_proj_weight() {
  delete transformer_h_35_mlp_up_proj_weight;
}

// Layer transformer.h.35.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_35_mlp_down_proj_weight;
}
void init_transformer_h_35_mlp_down_proj_weight() {
  half *transformer_h_35_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_35_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_35_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_35_mlp_down_proj_weight_data,
      transformer_h_35_mlp_down_proj_weight_shape);
}

void delete_transformer_h_35_mlp_down_proj_weight() {
  delete transformer_h_35_mlp_down_proj_weight;
}

// Layer transformer.h.35.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_35_mlp_down_proj_bias;
}
void init_transformer_h_35_mlp_down_proj_bias() {
  half *transformer_h_35_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_35_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_35_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_35_mlp_down_proj_bias_data,
                                    transformer_h_35_mlp_down_proj_bias_shape);
}

void delete_transformer_h_35_mlp_down_proj_bias() {
  delete transformer_h_35_mlp_down_proj_bias;
}

// Layer transformer.h.36.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_36_input_layernorm_weight;
}
void init_transformer_h_36_input_layernorm_weight() {
  half *transformer_h_36_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_36_input_layernorm_weight_shape[1] = {5120};
  transformer_h_36_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_36_input_layernorm_weight_data,
      transformer_h_36_input_layernorm_weight_shape);
}

void delete_transformer_h_36_input_layernorm_weight() {
  delete transformer_h_36_input_layernorm_weight;
}

// Layer transformer.h.36.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_36_self_attention_query_weight;
}
void init_transformer_h_36_self_attention_query_weight() {
  half *transformer_h_36_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_36_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_36_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_36_self_attention_query_weight_data,
      transformer_h_36_self_attention_query_weight_shape);
}

void delete_transformer_h_36_self_attention_query_weight() {
  delete transformer_h_36_self_attention_query_weight;
}

// Layer transformer.h.36.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_36_self_attention_key_value_weight;
}
void init_transformer_h_36_self_attention_key_value_weight() {
  half *transformer_h_36_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_36_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_36_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_36_self_attention_key_value_weight_data,
          transformer_h_36_self_attention_key_value_weight_shape);
}

void delete_transformer_h_36_self_attention_key_value_weight() {
  delete transformer_h_36_self_attention_key_value_weight;
}

// Layer transformer.h.36.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_36_self_attention_dense_weight;
}
void init_transformer_h_36_self_attention_dense_weight() {
  half *transformer_h_36_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_36_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_36_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_36_self_attention_dense_weight_data,
      transformer_h_36_self_attention_dense_weight_shape);
}

void delete_transformer_h_36_self_attention_dense_weight() {
  delete transformer_h_36_self_attention_dense_weight;
}

// Layer transformer.h.36.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_36_self_attention_dense_bias;
}
void init_transformer_h_36_self_attention_dense_bias() {
  half *transformer_h_36_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_36_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_36_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_36_self_attention_dense_bias_data,
      transformer_h_36_self_attention_dense_bias_shape);
}

void delete_transformer_h_36_self_attention_dense_bias() {
  delete transformer_h_36_self_attention_dense_bias;
}

// Layer transformer.h.36.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_36_post_attention_layernorm_weight;
}
void init_transformer_h_36_post_attention_layernorm_weight() {
  half *transformer_h_36_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_36_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_36_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_36_post_attention_layernorm_weight_data,
          transformer_h_36_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_36_post_attention_layernorm_weight() {
  delete transformer_h_36_post_attention_layernorm_weight;
}

// Layer transformer.h.36.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_36_mlp_gate_proj_weight;
}
void init_transformer_h_36_mlp_gate_proj_weight() {
  half *transformer_h_36_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_36_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_36_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_36_mlp_gate_proj_weight_data,
      transformer_h_36_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_36_mlp_gate_proj_weight() {
  delete transformer_h_36_mlp_gate_proj_weight;
}

// Layer transformer.h.36.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_36_mlp_up_proj_weight;
}
void init_transformer_h_36_mlp_up_proj_weight() {
  half *transformer_h_36_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_36_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_36_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_36_mlp_up_proj_weight_data,
                                    transformer_h_36_mlp_up_proj_weight_shape);
}

void delete_transformer_h_36_mlp_up_proj_weight() {
  delete transformer_h_36_mlp_up_proj_weight;
}

// Layer transformer.h.36.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_36_mlp_down_proj_weight;
}
void init_transformer_h_36_mlp_down_proj_weight() {
  half *transformer_h_36_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_36_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_36_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_36_mlp_down_proj_weight_data,
      transformer_h_36_mlp_down_proj_weight_shape);
}

void delete_transformer_h_36_mlp_down_proj_weight() {
  delete transformer_h_36_mlp_down_proj_weight;
}

// Layer transformer.h.36.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_36_mlp_down_proj_bias;
}
void init_transformer_h_36_mlp_down_proj_bias() {
  half *transformer_h_36_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_36_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_36_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_36_mlp_down_proj_bias_data,
                                    transformer_h_36_mlp_down_proj_bias_shape);
}

void delete_transformer_h_36_mlp_down_proj_bias() {
  delete transformer_h_36_mlp_down_proj_bias;
}

// Layer transformer.h.37.input_layernorm.weight: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_37_input_layernorm_weight;
}
void init_transformer_h_37_input_layernorm_weight() {
  half *transformer_h_37_input_layernorm_weight_data = new half[5120];
  int64_t transformer_h_37_input_layernorm_weight_shape[1] = {5120};
  transformer_h_37_input_layernorm_weight = new RankedMemRefType<half, 1>(
      transformer_h_37_input_layernorm_weight_data,
      transformer_h_37_input_layernorm_weight_shape);
}

void delete_transformer_h_37_input_layernorm_weight() {
  delete transformer_h_37_input_layernorm_weight;
}

// Layer transformer.h.37.self_attention.query.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_37_self_attention_query_weight;
}
void init_transformer_h_37_self_attention_query_weight() {
  half *transformer_h_37_self_attention_query_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_37_self_attention_query_weight_shape[2] = {5120, 5120};
  transformer_h_37_self_attention_query_weight = new RankedMemRefType<half, 2>(
      transformer_h_37_self_attention_query_weight_data,
      transformer_h_37_self_attention_query_weight_shape);
}

void delete_transformer_h_37_self_attention_query_weight() {
  delete transformer_h_37_self_attention_query_weight;
}

// Layer transformer.h.37.self_attention.key_value.weight: shape
// torch.Size([10240, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_37_self_attention_key_value_weight;
}
void init_transformer_h_37_self_attention_key_value_weight() {
  half *transformer_h_37_self_attention_key_value_weight_data =
      new half[10240 * 5120];
  int64_t transformer_h_37_self_attention_key_value_weight_shape[2] = {10240,
                                                                       5120};
  transformer_h_37_self_attention_key_value_weight =
      new RankedMemRefType<half, 2>(
          transformer_h_37_self_attention_key_value_weight_data,
          transformer_h_37_self_attention_key_value_weight_shape);
}

void delete_transformer_h_37_self_attention_key_value_weight() {
  delete transformer_h_37_self_attention_key_value_weight;
}

// Layer transformer.h.37.self_attention.dense.weight: shape torch.Size([5120,
// 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_37_self_attention_dense_weight;
}
void init_transformer_h_37_self_attention_dense_weight() {
  half *transformer_h_37_self_attention_dense_weight_data =
      new half[5120 * 5120];
  int64_t transformer_h_37_self_attention_dense_weight_shape[2] = {5120, 5120};
  transformer_h_37_self_attention_dense_weight = new RankedMemRefType<half, 2>(
      transformer_h_37_self_attention_dense_weight_data,
      transformer_h_37_self_attention_dense_weight_shape);
}

void delete_transformer_h_37_self_attention_dense_weight() {
  delete transformer_h_37_self_attention_dense_weight;
}

// Layer transformer.h.37.self_attention.dense.bias: shape torch.Size([5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_37_self_attention_dense_bias;
}
void init_transformer_h_37_self_attention_dense_bias() {
  half *transformer_h_37_self_attention_dense_bias_data = new half[5120];
  int64_t transformer_h_37_self_attention_dense_bias_shape[1] = {5120};
  transformer_h_37_self_attention_dense_bias = new RankedMemRefType<half, 1>(
      transformer_h_37_self_attention_dense_bias_data,
      transformer_h_37_self_attention_dense_bias_shape);
}

void delete_transformer_h_37_self_attention_dense_bias() {
  delete transformer_h_37_self_attention_dense_bias;
}

// Layer transformer.h.37.post_attention_layernorm.weight: shape
// torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_37_post_attention_layernorm_weight;
}
void init_transformer_h_37_post_attention_layernorm_weight() {
  half *transformer_h_37_post_attention_layernorm_weight_data = new half[5120];
  int64_t transformer_h_37_post_attention_layernorm_weight_shape[1] = {5120};
  transformer_h_37_post_attention_layernorm_weight =
      new RankedMemRefType<half, 1>(
          transformer_h_37_post_attention_layernorm_weight_data,
          transformer_h_37_post_attention_layernorm_weight_shape);
}

void delete_transformer_h_37_post_attention_layernorm_weight() {
  delete transformer_h_37_post_attention_layernorm_weight;
}

// Layer transformer.h.37.mlp.gate_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_37_mlp_gate_proj_weight;
}
void init_transformer_h_37_mlp_gate_proj_weight() {
  half *transformer_h_37_mlp_gate_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_37_mlp_gate_proj_weight_shape[2] = {12288, 5120};
  transformer_h_37_mlp_gate_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_37_mlp_gate_proj_weight_data,
      transformer_h_37_mlp_gate_proj_weight_shape);
}

void delete_transformer_h_37_mlp_gate_proj_weight() {
  delete transformer_h_37_mlp_gate_proj_weight;
}

// Layer transformer.h.37.mlp.up_proj.weight: shape torch.Size([12288, 5120]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_37_mlp_up_proj_weight;
}
void init_transformer_h_37_mlp_up_proj_weight() {
  half *transformer_h_37_mlp_up_proj_weight_data = new half[12288 * 5120];
  int64_t transformer_h_37_mlp_up_proj_weight_shape[2] = {12288, 5120};
  transformer_h_37_mlp_up_proj_weight =
      new RankedMemRefType<half, 2>(transformer_h_37_mlp_up_proj_weight_data,
                                    transformer_h_37_mlp_up_proj_weight_shape);
}

void delete_transformer_h_37_mlp_up_proj_weight() {
  delete transformer_h_37_mlp_up_proj_weight;
}

// Layer transformer.h.37.mlp.down_proj.weight: shape torch.Size([5120, 12288]),
// dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *transformer_h_37_mlp_down_proj_weight;
}
void init_transformer_h_37_mlp_down_proj_weight() {
  half *transformer_h_37_mlp_down_proj_weight_data = new half[5120 * 12288];
  int64_t transformer_h_37_mlp_down_proj_weight_shape[2] = {5120, 12288};
  transformer_h_37_mlp_down_proj_weight = new RankedMemRefType<half, 2>(
      transformer_h_37_mlp_down_proj_weight_data,
      transformer_h_37_mlp_down_proj_weight_shape);
}

void delete_transformer_h_37_mlp_down_proj_weight() {
  delete transformer_h_37_mlp_down_proj_weight;
}

// Layer transformer.h.37.mlp.down_proj.bias: shape torch.Size([5120]), dtype
// torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_h_37_mlp_down_proj_bias;
}
void init_transformer_h_37_mlp_down_proj_bias() {
  half *transformer_h_37_mlp_down_proj_bias_data = new half[5120];
  int64_t transformer_h_37_mlp_down_proj_bias_shape[1] = {5120};
  transformer_h_37_mlp_down_proj_bias =
      new RankedMemRefType<half, 1>(transformer_h_37_mlp_down_proj_bias_data,
                                    transformer_h_37_mlp_down_proj_bias_shape);
}

void delete_transformer_h_37_mlp_down_proj_bias() {
  delete transformer_h_37_mlp_down_proj_bias;
}

// Layer transformer.ln_f.weight: shape torch.Size([5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 1> *transformer_ln_f_weight;
}
void init_transformer_ln_f_weight() {
  half *transformer_ln_f_weight_data = new half[5120];
  int64_t transformer_ln_f_weight_shape[1] = {5120};
  transformer_ln_f_weight = new RankedMemRefType<half, 1>(
      transformer_ln_f_weight_data, transformer_ln_f_weight_shape);
}

void delete_transformer_ln_f_weight() { delete transformer_ln_f_weight; }

// Layer lm_head.weight: shape torch.Size([120000, 5120]), dtype torch.float16
extern "C" {
RankedMemRefType<half, 2> *lm_head_weight;
}
void init_lm_head_weight() {
  half *lm_head_weight_data = new half[120000 * 5120];
  int64_t lm_head_weight_shape[2] = {120000, 5120};
  lm_head_weight =
      new RankedMemRefType<half, 2>(lm_head_weight_data, lm_head_weight_shape);
}

void delete_lm_head_weight() { delete lm_head_weight; }

void init_all_globals() {
  init_transformer_word_embeddings_weight();
  init_transformer_h_0_input_layernorm_weight();
  init_transformer_h_0_self_attention_query_weight();
  init_transformer_h_0_self_attention_key_value_weight();
  init_transformer_h_0_self_attention_dense_weight();
  init_transformer_h_0_self_attention_dense_bias();
  init_transformer_h_0_post_attention_layernorm_weight();
  init_transformer_h_0_mlp_gate_proj_weight();
  init_transformer_h_0_mlp_up_proj_weight();
  init_transformer_h_0_mlp_down_proj_weight();
  init_transformer_h_0_mlp_down_proj_bias();
  init_transformer_h_1_input_layernorm_weight();
  init_transformer_h_1_self_attention_query_weight();
  init_transformer_h_1_self_attention_key_value_weight();
  init_transformer_h_1_self_attention_dense_weight();
  init_transformer_h_1_self_attention_dense_bias();
  init_transformer_h_1_post_attention_layernorm_weight();
  init_transformer_h_1_mlp_gate_proj_weight();
  init_transformer_h_1_mlp_up_proj_weight();
  init_transformer_h_1_mlp_down_proj_weight();
  init_transformer_h_1_mlp_down_proj_bias();
  init_transformer_h_2_input_layernorm_weight();
  init_transformer_h_2_self_attention_query_weight();
  init_transformer_h_2_self_attention_key_value_weight();
  init_transformer_h_2_self_attention_dense_weight();
  init_transformer_h_2_self_attention_dense_bias();
  init_transformer_h_2_post_attention_layernorm_weight();
  init_transformer_h_2_mlp_gate_proj_weight();
  init_transformer_h_2_mlp_up_proj_weight();
  init_transformer_h_2_mlp_down_proj_weight();
  init_transformer_h_2_mlp_down_proj_bias();
  init_transformer_h_3_input_layernorm_weight();
  init_transformer_h_3_self_attention_query_weight();
  init_transformer_h_3_self_attention_key_value_weight();
  init_transformer_h_3_self_attention_dense_weight();
  init_transformer_h_3_self_attention_dense_bias();
  init_transformer_h_3_post_attention_layernorm_weight();
  init_transformer_h_3_mlp_gate_proj_weight();
  init_transformer_h_3_mlp_up_proj_weight();
  init_transformer_h_3_mlp_down_proj_weight();
  init_transformer_h_3_mlp_down_proj_bias();
  init_transformer_h_4_input_layernorm_weight();
  init_transformer_h_4_self_attention_query_weight();
  init_transformer_h_4_self_attention_key_value_weight();
  init_transformer_h_4_self_attention_dense_weight();
  init_transformer_h_4_self_attention_dense_bias();
  init_transformer_h_4_post_attention_layernorm_weight();
  init_transformer_h_4_mlp_gate_proj_weight();
  init_transformer_h_4_mlp_up_proj_weight();
  init_transformer_h_4_mlp_down_proj_weight();
  init_transformer_h_4_mlp_down_proj_bias();
  init_transformer_h_5_input_layernorm_weight();
  init_transformer_h_5_self_attention_query_weight();
  init_transformer_h_5_self_attention_key_value_weight();
  init_transformer_h_5_self_attention_dense_weight();
  init_transformer_h_5_self_attention_dense_bias();
  init_transformer_h_5_post_attention_layernorm_weight();
  init_transformer_h_5_mlp_gate_proj_weight();
  init_transformer_h_5_mlp_up_proj_weight();
  init_transformer_h_5_mlp_down_proj_weight();
  init_transformer_h_5_mlp_down_proj_bias();
  init_transformer_h_6_input_layernorm_weight();
  init_transformer_h_6_self_attention_query_weight();
  init_transformer_h_6_self_attention_key_value_weight();
  init_transformer_h_6_self_attention_dense_weight();
  init_transformer_h_6_self_attention_dense_bias();
  init_transformer_h_6_post_attention_layernorm_weight();
  init_transformer_h_6_mlp_gate_proj_weight();
  init_transformer_h_6_mlp_up_proj_weight();
  init_transformer_h_6_mlp_down_proj_weight();
  init_transformer_h_6_mlp_down_proj_bias();
  init_transformer_h_7_input_layernorm_weight();
  init_transformer_h_7_self_attention_query_weight();
  init_transformer_h_7_self_attention_key_value_weight();
  init_transformer_h_7_self_attention_dense_weight();
  init_transformer_h_7_self_attention_dense_bias();
  init_transformer_h_7_post_attention_layernorm_weight();
  init_transformer_h_7_mlp_gate_proj_weight();
  init_transformer_h_7_mlp_up_proj_weight();
  init_transformer_h_7_mlp_down_proj_weight();
  init_transformer_h_7_mlp_down_proj_bias();
  init_transformer_h_8_input_layernorm_weight();
  init_transformer_h_8_self_attention_query_weight();
  init_transformer_h_8_self_attention_key_value_weight();
  init_transformer_h_8_self_attention_dense_weight();
  init_transformer_h_8_self_attention_dense_bias();
  init_transformer_h_8_post_attention_layernorm_weight();
  init_transformer_h_8_mlp_gate_proj_weight();
  init_transformer_h_8_mlp_up_proj_weight();
  init_transformer_h_8_mlp_down_proj_weight();
  init_transformer_h_8_mlp_down_proj_bias();
  init_transformer_h_9_input_layernorm_weight();
  init_transformer_h_9_self_attention_query_weight();
  init_transformer_h_9_self_attention_key_value_weight();
  init_transformer_h_9_self_attention_dense_weight();
  init_transformer_h_9_self_attention_dense_bias();
  init_transformer_h_9_post_attention_layernorm_weight();
  init_transformer_h_9_mlp_gate_proj_weight();
  init_transformer_h_9_mlp_up_proj_weight();
  init_transformer_h_9_mlp_down_proj_weight();
  init_transformer_h_9_mlp_down_proj_bias();
  init_transformer_h_10_input_layernorm_weight();
  init_transformer_h_10_self_attention_query_weight();
  init_transformer_h_10_self_attention_key_value_weight();
  init_transformer_h_10_self_attention_dense_weight();
  init_transformer_h_10_self_attention_dense_bias();
  init_transformer_h_10_post_attention_layernorm_weight();
  init_transformer_h_10_mlp_gate_proj_weight();
  init_transformer_h_10_mlp_up_proj_weight();
  init_transformer_h_10_mlp_down_proj_weight();
  init_transformer_h_10_mlp_down_proj_bias();
  init_transformer_h_11_input_layernorm_weight();
  init_transformer_h_11_self_attention_query_weight();
  init_transformer_h_11_self_attention_key_value_weight();
  init_transformer_h_11_self_attention_dense_weight();
  init_transformer_h_11_self_attention_dense_bias();
  init_transformer_h_11_post_attention_layernorm_weight();
  init_transformer_h_11_mlp_gate_proj_weight();
  init_transformer_h_11_mlp_up_proj_weight();
  init_transformer_h_11_mlp_down_proj_weight();
  init_transformer_h_11_mlp_down_proj_bias();
  init_transformer_h_12_input_layernorm_weight();
  init_transformer_h_12_self_attention_query_weight();
  init_transformer_h_12_self_attention_key_value_weight();
  init_transformer_h_12_self_attention_dense_weight();
  init_transformer_h_12_self_attention_dense_bias();
  init_transformer_h_12_post_attention_layernorm_weight();
  init_transformer_h_12_mlp_gate_proj_weight();
  init_transformer_h_12_mlp_up_proj_weight();
  init_transformer_h_12_mlp_down_proj_weight();
  init_transformer_h_12_mlp_down_proj_bias();
  init_transformer_h_13_input_layernorm_weight();
  init_transformer_h_13_self_attention_query_weight();
  init_transformer_h_13_self_attention_key_value_weight();
  init_transformer_h_13_self_attention_dense_weight();
  init_transformer_h_13_self_attention_dense_bias();
  init_transformer_h_13_post_attention_layernorm_weight();
  init_transformer_h_13_mlp_gate_proj_weight();
  init_transformer_h_13_mlp_up_proj_weight();
  init_transformer_h_13_mlp_down_proj_weight();
  init_transformer_h_13_mlp_down_proj_bias();
  init_transformer_h_14_input_layernorm_weight();
  init_transformer_h_14_self_attention_query_weight();
  init_transformer_h_14_self_attention_key_value_weight();
  init_transformer_h_14_self_attention_dense_weight();
  init_transformer_h_14_self_attention_dense_bias();
  init_transformer_h_14_post_attention_layernorm_weight();
  init_transformer_h_14_mlp_gate_proj_weight();
  init_transformer_h_14_mlp_up_proj_weight();
  init_transformer_h_14_mlp_down_proj_weight();
  init_transformer_h_14_mlp_down_proj_bias();
  init_transformer_h_15_input_layernorm_weight();
  init_transformer_h_15_self_attention_query_weight();
  init_transformer_h_15_self_attention_key_value_weight();
  init_transformer_h_15_self_attention_dense_weight();
  init_transformer_h_15_self_attention_dense_bias();
  init_transformer_h_15_post_attention_layernorm_weight();
  init_transformer_h_15_mlp_gate_proj_weight();
  init_transformer_h_15_mlp_up_proj_weight();
  init_transformer_h_15_mlp_down_proj_weight();
  init_transformer_h_15_mlp_down_proj_bias();
  init_transformer_h_16_input_layernorm_weight();
  init_transformer_h_16_self_attention_query_weight();
  init_transformer_h_16_self_attention_key_value_weight();
  init_transformer_h_16_self_attention_dense_weight();
  init_transformer_h_16_self_attention_dense_bias();
  init_transformer_h_16_post_attention_layernorm_weight();
  init_transformer_h_16_mlp_gate_proj_weight();
  init_transformer_h_16_mlp_up_proj_weight();
  init_transformer_h_16_mlp_down_proj_weight();
  init_transformer_h_16_mlp_down_proj_bias();
  init_transformer_h_17_input_layernorm_weight();
  init_transformer_h_17_self_attention_query_weight();
  init_transformer_h_17_self_attention_key_value_weight();
  init_transformer_h_17_self_attention_dense_weight();
  init_transformer_h_17_self_attention_dense_bias();
  init_transformer_h_17_post_attention_layernorm_weight();
  init_transformer_h_17_mlp_gate_proj_weight();
  init_transformer_h_17_mlp_up_proj_weight();
  init_transformer_h_17_mlp_down_proj_weight();
  init_transformer_h_17_mlp_down_proj_bias();
  init_transformer_h_18_input_layernorm_weight();
  init_transformer_h_18_self_attention_query_weight();
  init_transformer_h_18_self_attention_key_value_weight();
  init_transformer_h_18_self_attention_dense_weight();
  init_transformer_h_18_self_attention_dense_bias();
  init_transformer_h_18_post_attention_layernorm_weight();
  init_transformer_h_18_mlp_gate_proj_weight();
  init_transformer_h_18_mlp_up_proj_weight();
  init_transformer_h_18_mlp_down_proj_weight();
  init_transformer_h_18_mlp_down_proj_bias();
  init_transformer_h_19_input_layernorm_weight();
  init_transformer_h_19_self_attention_query_weight();
  init_transformer_h_19_self_attention_key_value_weight();
  init_transformer_h_19_self_attention_dense_weight();
  init_transformer_h_19_self_attention_dense_bias();
  init_transformer_h_19_post_attention_layernorm_weight();
  init_transformer_h_19_mlp_gate_proj_weight();
  init_transformer_h_19_mlp_up_proj_weight();
  init_transformer_h_19_mlp_down_proj_weight();
  init_transformer_h_19_mlp_down_proj_bias();
  init_transformer_h_20_input_layernorm_weight();
  init_transformer_h_20_self_attention_query_weight();
  init_transformer_h_20_self_attention_key_value_weight();
  init_transformer_h_20_self_attention_dense_weight();
  init_transformer_h_20_self_attention_dense_bias();
  init_transformer_h_20_post_attention_layernorm_weight();
  init_transformer_h_20_mlp_gate_proj_weight();
  init_transformer_h_20_mlp_up_proj_weight();
  init_transformer_h_20_mlp_down_proj_weight();
  init_transformer_h_20_mlp_down_proj_bias();
  init_transformer_h_21_input_layernorm_weight();
  init_transformer_h_21_self_attention_query_weight();
  init_transformer_h_21_self_attention_key_value_weight();
  init_transformer_h_21_self_attention_dense_weight();
  init_transformer_h_21_self_attention_dense_bias();
  init_transformer_h_21_post_attention_layernorm_weight();
  init_transformer_h_21_mlp_gate_proj_weight();
  init_transformer_h_21_mlp_up_proj_weight();
  init_transformer_h_21_mlp_down_proj_weight();
  init_transformer_h_21_mlp_down_proj_bias();
  init_transformer_h_22_input_layernorm_weight();
  init_transformer_h_22_self_attention_query_weight();
  init_transformer_h_22_self_attention_key_value_weight();
  init_transformer_h_22_self_attention_dense_weight();
  init_transformer_h_22_self_attention_dense_bias();
  init_transformer_h_22_post_attention_layernorm_weight();
  init_transformer_h_22_mlp_gate_proj_weight();
  init_transformer_h_22_mlp_up_proj_weight();
  init_transformer_h_22_mlp_down_proj_weight();
  init_transformer_h_22_mlp_down_proj_bias();
  init_transformer_h_23_input_layernorm_weight();
  init_transformer_h_23_self_attention_query_weight();
  init_transformer_h_23_self_attention_key_value_weight();
  init_transformer_h_23_self_attention_dense_weight();
  init_transformer_h_23_self_attention_dense_bias();
  init_transformer_h_23_post_attention_layernorm_weight();
  init_transformer_h_23_mlp_gate_proj_weight();
  init_transformer_h_23_mlp_up_proj_weight();
  init_transformer_h_23_mlp_down_proj_weight();
  init_transformer_h_23_mlp_down_proj_bias();
  init_transformer_h_24_input_layernorm_weight();
  init_transformer_h_24_self_attention_query_weight();
  init_transformer_h_24_self_attention_key_value_weight();
  init_transformer_h_24_self_attention_dense_weight();
  init_transformer_h_24_self_attention_dense_bias();
  init_transformer_h_24_post_attention_layernorm_weight();
  init_transformer_h_24_mlp_gate_proj_weight();
  init_transformer_h_24_mlp_up_proj_weight();
  init_transformer_h_24_mlp_down_proj_weight();
  init_transformer_h_24_mlp_down_proj_bias();
  init_transformer_h_25_input_layernorm_weight();
  init_transformer_h_25_self_attention_query_weight();
  init_transformer_h_25_self_attention_key_value_weight();
  init_transformer_h_25_self_attention_dense_weight();
  init_transformer_h_25_self_attention_dense_bias();
  init_transformer_h_25_post_attention_layernorm_weight();
  init_transformer_h_25_mlp_gate_proj_weight();
  init_transformer_h_25_mlp_up_proj_weight();
  init_transformer_h_25_mlp_down_proj_weight();
  init_transformer_h_25_mlp_down_proj_bias();
  init_transformer_h_26_input_layernorm_weight();
  init_transformer_h_26_self_attention_query_weight();
  init_transformer_h_26_self_attention_key_value_weight();
  init_transformer_h_26_self_attention_dense_weight();
  init_transformer_h_26_self_attention_dense_bias();
  init_transformer_h_26_post_attention_layernorm_weight();
  init_transformer_h_26_mlp_gate_proj_weight();
  init_transformer_h_26_mlp_up_proj_weight();
  init_transformer_h_26_mlp_down_proj_weight();
  init_transformer_h_26_mlp_down_proj_bias();
  init_transformer_h_27_input_layernorm_weight();
  init_transformer_h_27_self_attention_query_weight();
  init_transformer_h_27_self_attention_key_value_weight();
  init_transformer_h_27_self_attention_dense_weight();
  init_transformer_h_27_self_attention_dense_bias();
  init_transformer_h_27_post_attention_layernorm_weight();
  init_transformer_h_27_mlp_gate_proj_weight();
  init_transformer_h_27_mlp_up_proj_weight();
  init_transformer_h_27_mlp_down_proj_weight();
  init_transformer_h_27_mlp_down_proj_bias();
  init_transformer_h_28_input_layernorm_weight();
  init_transformer_h_28_self_attention_query_weight();
  init_transformer_h_28_self_attention_key_value_weight();
  init_transformer_h_28_self_attention_dense_weight();
  init_transformer_h_28_self_attention_dense_bias();
  init_transformer_h_28_post_attention_layernorm_weight();
  init_transformer_h_28_mlp_gate_proj_weight();
  init_transformer_h_28_mlp_up_proj_weight();
  init_transformer_h_28_mlp_down_proj_weight();
  init_transformer_h_28_mlp_down_proj_bias();
  init_transformer_h_29_input_layernorm_weight();
  init_transformer_h_29_self_attention_query_weight();
  init_transformer_h_29_self_attention_key_value_weight();
  init_transformer_h_29_self_attention_dense_weight();
  init_transformer_h_29_self_attention_dense_bias();
  init_transformer_h_29_post_attention_layernorm_weight();
  init_transformer_h_29_mlp_gate_proj_weight();
  init_transformer_h_29_mlp_up_proj_weight();
  init_transformer_h_29_mlp_down_proj_weight();
  init_transformer_h_29_mlp_down_proj_bias();
  init_transformer_h_30_input_layernorm_weight();
  init_transformer_h_30_self_attention_query_weight();
  init_transformer_h_30_self_attention_key_value_weight();
  init_transformer_h_30_self_attention_dense_weight();
  init_transformer_h_30_self_attention_dense_bias();
  init_transformer_h_30_post_attention_layernorm_weight();
  init_transformer_h_30_mlp_gate_proj_weight();
  init_transformer_h_30_mlp_up_proj_weight();
  init_transformer_h_30_mlp_down_proj_weight();
  init_transformer_h_30_mlp_down_proj_bias();
  init_transformer_h_31_input_layernorm_weight();
  init_transformer_h_31_self_attention_query_weight();
  init_transformer_h_31_self_attention_key_value_weight();
  init_transformer_h_31_self_attention_dense_weight();
  init_transformer_h_31_self_attention_dense_bias();
  init_transformer_h_31_post_attention_layernorm_weight();
  init_transformer_h_31_mlp_gate_proj_weight();
  init_transformer_h_31_mlp_up_proj_weight();
  init_transformer_h_31_mlp_down_proj_weight();
  init_transformer_h_31_mlp_down_proj_bias();
  init_transformer_h_32_input_layernorm_weight();
  init_transformer_h_32_self_attention_query_weight();
  init_transformer_h_32_self_attention_key_value_weight();
  init_transformer_h_32_self_attention_dense_weight();
  init_transformer_h_32_self_attention_dense_bias();
  init_transformer_h_32_post_attention_layernorm_weight();
  init_transformer_h_32_mlp_gate_proj_weight();
  init_transformer_h_32_mlp_up_proj_weight();
  init_transformer_h_32_mlp_down_proj_weight();
  init_transformer_h_32_mlp_down_proj_bias();
  init_transformer_h_33_input_layernorm_weight();
  init_transformer_h_33_self_attention_query_weight();
  init_transformer_h_33_self_attention_key_value_weight();
  init_transformer_h_33_self_attention_dense_weight();
  init_transformer_h_33_self_attention_dense_bias();
  init_transformer_h_33_post_attention_layernorm_weight();
  init_transformer_h_33_mlp_gate_proj_weight();
  init_transformer_h_33_mlp_up_proj_weight();
  init_transformer_h_33_mlp_down_proj_weight();
  init_transformer_h_33_mlp_down_proj_bias();
  init_transformer_h_34_input_layernorm_weight();
  init_transformer_h_34_self_attention_query_weight();
  init_transformer_h_34_self_attention_key_value_weight();
  init_transformer_h_34_self_attention_dense_weight();
  init_transformer_h_34_self_attention_dense_bias();
  init_transformer_h_34_post_attention_layernorm_weight();
  init_transformer_h_34_mlp_gate_proj_weight();
  init_transformer_h_34_mlp_up_proj_weight();
  init_transformer_h_34_mlp_down_proj_weight();
  init_transformer_h_34_mlp_down_proj_bias();
  init_transformer_h_35_input_layernorm_weight();
  init_transformer_h_35_self_attention_query_weight();
  init_transformer_h_35_self_attention_key_value_weight();
  init_transformer_h_35_self_attention_dense_weight();
  init_transformer_h_35_self_attention_dense_bias();
  init_transformer_h_35_post_attention_layernorm_weight();
  init_transformer_h_35_mlp_gate_proj_weight();
  init_transformer_h_35_mlp_up_proj_weight();
  init_transformer_h_35_mlp_down_proj_weight();
  init_transformer_h_35_mlp_down_proj_bias();
  init_transformer_h_36_input_layernorm_weight();
  init_transformer_h_36_self_attention_query_weight();
  init_transformer_h_36_self_attention_key_value_weight();
  init_transformer_h_36_self_attention_dense_weight();
  init_transformer_h_36_self_attention_dense_bias();
  init_transformer_h_36_post_attention_layernorm_weight();
  init_transformer_h_36_mlp_gate_proj_weight();
  init_transformer_h_36_mlp_up_proj_weight();
  init_transformer_h_36_mlp_down_proj_weight();
  init_transformer_h_36_mlp_down_proj_bias();
  init_transformer_h_37_input_layernorm_weight();
  init_transformer_h_37_self_attention_query_weight();
  init_transformer_h_37_self_attention_key_value_weight();
  init_transformer_h_37_self_attention_dense_weight();
  init_transformer_h_37_self_attention_dense_bias();
  init_transformer_h_37_post_attention_layernorm_weight();
  init_transformer_h_37_mlp_gate_proj_weight();
  init_transformer_h_37_mlp_up_proj_weight();
  init_transformer_h_37_mlp_down_proj_weight();
  init_transformer_h_37_mlp_down_proj_bias();
  init_transformer_ln_f_weight();
  init_lm_head_weight();
  std::vector<std::string> model_names = {"./pytorch_model_00001-of-00004.bin",
                                          "./pytorch_model_00002-of-00004.bin",
                                          "./pytorch_model_00003-of-00004.bin",
                                          "./pytorch_model_00004-of-00004.bin"};
  std::map<std::string, half *> param_and_loc = {
      {"transformer.word_embeddings.weight",
       transformer_word_embeddings_weight->data},
      {"transformer.h.0.input_layernorm.weight",
       transformer_h_0_input_layernorm_weight->data},
      {"transformer.h.0.self_attention.query.weight",
       transformer_h_0_self_attention_query_weight->data},
      {"transformer.h.0.self_attention.key_value.weight",
       transformer_h_0_self_attention_key_value_weight->data},
      {"transformer.h.0.self_attention.dense.weight",
       transformer_h_0_self_attention_dense_weight->data},
      {"transformer.h.0.self_attention.dense.bias",
       transformer_h_0_self_attention_dense_bias->data},
      {"transformer.h.0.post_attention_layernorm.weight",
       transformer_h_0_post_attention_layernorm_weight->data},
      {"transformer.h.0.mlp.gate_proj.weight",
       transformer_h_0_mlp_gate_proj_weight->data},
      {"transformer.h.0.mlp.up_proj.weight",
       transformer_h_0_mlp_up_proj_weight->data},
      {"transformer.h.0.mlp.down_proj.weight",
       transformer_h_0_mlp_down_proj_weight->data},
      {"transformer.h.0.mlp.down_proj.bias",
       transformer_h_0_mlp_down_proj_bias->data},
      {"transformer.h.1.input_layernorm.weight",
       transformer_h_1_input_layernorm_weight->data},
      {"transformer.h.1.self_attention.query.weight",
       transformer_h_1_self_attention_query_weight->data},
      {"transformer.h.1.self_attention.key_value.weight",
       transformer_h_1_self_attention_key_value_weight->data},
      {"transformer.h.1.self_attention.dense.weight",
       transformer_h_1_self_attention_dense_weight->data},
      {"transformer.h.1.self_attention.dense.bias",
       transformer_h_1_self_attention_dense_bias->data},
      {"transformer.h.1.post_attention_layernorm.weight",
       transformer_h_1_post_attention_layernorm_weight->data},
      {"transformer.h.1.mlp.gate_proj.weight",
       transformer_h_1_mlp_gate_proj_weight->data},
      {"transformer.h.1.mlp.up_proj.weight",
       transformer_h_1_mlp_up_proj_weight->data},
      {"transformer.h.1.mlp.down_proj.weight",
       transformer_h_1_mlp_down_proj_weight->data},
      {"transformer.h.1.mlp.down_proj.bias",
       transformer_h_1_mlp_down_proj_bias->data},
      {"transformer.h.2.input_layernorm.weight",
       transformer_h_2_input_layernorm_weight->data},
      {"transformer.h.2.self_attention.query.weight",
       transformer_h_2_self_attention_query_weight->data},
      {"transformer.h.2.self_attention.key_value.weight",
       transformer_h_2_self_attention_key_value_weight->data},
      {"transformer.h.2.self_attention.dense.weight",
       transformer_h_2_self_attention_dense_weight->data},
      {"transformer.h.2.self_attention.dense.bias",
       transformer_h_2_self_attention_dense_bias->data},
      {"transformer.h.2.post_attention_layernorm.weight",
       transformer_h_2_post_attention_layernorm_weight->data},
      {"transformer.h.2.mlp.gate_proj.weight",
       transformer_h_2_mlp_gate_proj_weight->data},
      {"transformer.h.2.mlp.up_proj.weight",
       transformer_h_2_mlp_up_proj_weight->data},
      {"transformer.h.2.mlp.down_proj.weight",
       transformer_h_2_mlp_down_proj_weight->data},
      {"transformer.h.2.mlp.down_proj.bias",
       transformer_h_2_mlp_down_proj_bias->data},
      {"transformer.h.3.input_layernorm.weight",
       transformer_h_3_input_layernorm_weight->data},
      {"transformer.h.3.self_attention.query.weight",
       transformer_h_3_self_attention_query_weight->data},
      {"transformer.h.3.self_attention.key_value.weight",
       transformer_h_3_self_attention_key_value_weight->data},
      {"transformer.h.3.self_attention.dense.weight",
       transformer_h_3_self_attention_dense_weight->data},
      {"transformer.h.3.self_attention.dense.bias",
       transformer_h_3_self_attention_dense_bias->data},
      {"transformer.h.3.post_attention_layernorm.weight",
       transformer_h_3_post_attention_layernorm_weight->data},
      {"transformer.h.3.mlp.gate_proj.weight",
       transformer_h_3_mlp_gate_proj_weight->data},
      {"transformer.h.3.mlp.up_proj.weight",
       transformer_h_3_mlp_up_proj_weight->data},
      {"transformer.h.3.mlp.down_proj.weight",
       transformer_h_3_mlp_down_proj_weight->data},
      {"transformer.h.3.mlp.down_proj.bias",
       transformer_h_3_mlp_down_proj_bias->data},
      {"transformer.h.4.input_layernorm.weight",
       transformer_h_4_input_layernorm_weight->data},
      {"transformer.h.4.self_attention.query.weight",
       transformer_h_4_self_attention_query_weight->data},
      {"transformer.h.4.self_attention.key_value.weight",
       transformer_h_4_self_attention_key_value_weight->data},
      {"transformer.h.4.self_attention.dense.weight",
       transformer_h_4_self_attention_dense_weight->data},
      {"transformer.h.4.self_attention.dense.bias",
       transformer_h_4_self_attention_dense_bias->data},
      {"transformer.h.4.post_attention_layernorm.weight",
       transformer_h_4_post_attention_layernorm_weight->data},
      {"transformer.h.4.mlp.gate_proj.weight",
       transformer_h_4_mlp_gate_proj_weight->data},
      {"transformer.h.4.mlp.up_proj.weight",
       transformer_h_4_mlp_up_proj_weight->data},
      {"transformer.h.4.mlp.down_proj.weight",
       transformer_h_4_mlp_down_proj_weight->data},
      {"transformer.h.4.mlp.down_proj.bias",
       transformer_h_4_mlp_down_proj_bias->data},
      {"transformer.h.5.input_layernorm.weight",
       transformer_h_5_input_layernorm_weight->data},
      {"transformer.h.5.self_attention.query.weight",
       transformer_h_5_self_attention_query_weight->data},
      {"transformer.h.5.self_attention.key_value.weight",
       transformer_h_5_self_attention_key_value_weight->data},
      {"transformer.h.5.self_attention.dense.weight",
       transformer_h_5_self_attention_dense_weight->data},
      {"transformer.h.5.self_attention.dense.bias",
       transformer_h_5_self_attention_dense_bias->data},
      {"transformer.h.5.post_attention_layernorm.weight",
       transformer_h_5_post_attention_layernorm_weight->data},
      {"transformer.h.5.mlp.gate_proj.weight",
       transformer_h_5_mlp_gate_proj_weight->data},
      {"transformer.h.5.mlp.up_proj.weight",
       transformer_h_5_mlp_up_proj_weight->data},
      {"transformer.h.5.mlp.down_proj.weight",
       transformer_h_5_mlp_down_proj_weight->data},
      {"transformer.h.5.mlp.down_proj.bias",
       transformer_h_5_mlp_down_proj_bias->data},
      {"transformer.h.6.input_layernorm.weight",
       transformer_h_6_input_layernorm_weight->data},
      {"transformer.h.6.self_attention.query.weight",
       transformer_h_6_self_attention_query_weight->data},
      {"transformer.h.6.self_attention.key_value.weight",
       transformer_h_6_self_attention_key_value_weight->data},
      {"transformer.h.6.self_attention.dense.weight",
       transformer_h_6_self_attention_dense_weight->data},
      {"transformer.h.6.self_attention.dense.bias",
       transformer_h_6_self_attention_dense_bias->data},
      {"transformer.h.6.post_attention_layernorm.weight",
       transformer_h_6_post_attention_layernorm_weight->data},
      {"transformer.h.6.mlp.gate_proj.weight",
       transformer_h_6_mlp_gate_proj_weight->data},
      {"transformer.h.6.mlp.up_proj.weight",
       transformer_h_6_mlp_up_proj_weight->data},
      {"transformer.h.6.mlp.down_proj.weight",
       transformer_h_6_mlp_down_proj_weight->data},
      {"transformer.h.6.mlp.down_proj.bias",
       transformer_h_6_mlp_down_proj_bias->data},
      {"transformer.h.7.input_layernorm.weight",
       transformer_h_7_input_layernorm_weight->data},
      {"transformer.h.7.self_attention.query.weight",
       transformer_h_7_self_attention_query_weight->data},
      {"transformer.h.7.self_attention.key_value.weight",
       transformer_h_7_self_attention_key_value_weight->data},
      {"transformer.h.7.self_attention.dense.weight",
       transformer_h_7_self_attention_dense_weight->data},
      {"transformer.h.7.self_attention.dense.bias",
       transformer_h_7_self_attention_dense_bias->data},
      {"transformer.h.7.post_attention_layernorm.weight",
       transformer_h_7_post_attention_layernorm_weight->data},
      {"transformer.h.7.mlp.gate_proj.weight",
       transformer_h_7_mlp_gate_proj_weight->data},
      {"transformer.h.7.mlp.up_proj.weight",
       transformer_h_7_mlp_up_proj_weight->data},
      {"transformer.h.7.mlp.down_proj.weight",
       transformer_h_7_mlp_down_proj_weight->data},
      {"transformer.h.7.mlp.down_proj.bias",
       transformer_h_7_mlp_down_proj_bias->data},
      {"transformer.h.8.input_layernorm.weight",
       transformer_h_8_input_layernorm_weight->data},
      {"transformer.h.8.self_attention.query.weight",
       transformer_h_8_self_attention_query_weight->data},
      {"transformer.h.8.self_attention.key_value.weight",
       transformer_h_8_self_attention_key_value_weight->data},
      {"transformer.h.8.self_attention.dense.weight",
       transformer_h_8_self_attention_dense_weight->data},
      {"transformer.h.8.self_attention.dense.bias",
       transformer_h_8_self_attention_dense_bias->data},
      {"transformer.h.8.post_attention_layernorm.weight",
       transformer_h_8_post_attention_layernorm_weight->data},
      {"transformer.h.8.mlp.gate_proj.weight",
       transformer_h_8_mlp_gate_proj_weight->data},
      {"transformer.h.8.mlp.up_proj.weight",
       transformer_h_8_mlp_up_proj_weight->data},
      {"transformer.h.8.mlp.down_proj.weight",
       transformer_h_8_mlp_down_proj_weight->data},
      {"transformer.h.8.mlp.down_proj.bias",
       transformer_h_8_mlp_down_proj_bias->data},
      {"transformer.h.9.input_layernorm.weight",
       transformer_h_9_input_layernorm_weight->data},
      {"transformer.h.9.self_attention.query.weight",
       transformer_h_9_self_attention_query_weight->data},
      {"transformer.h.9.self_attention.key_value.weight",
       transformer_h_9_self_attention_key_value_weight->data},
      {"transformer.h.9.self_attention.dense.weight",
       transformer_h_9_self_attention_dense_weight->data},
      {"transformer.h.9.self_attention.dense.bias",
       transformer_h_9_self_attention_dense_bias->data},
      {"transformer.h.9.post_attention_layernorm.weight",
       transformer_h_9_post_attention_layernorm_weight->data},
      {"transformer.h.9.mlp.gate_proj.weight",
       transformer_h_9_mlp_gate_proj_weight->data},
      {"transformer.h.9.mlp.up_proj.weight",
       transformer_h_9_mlp_up_proj_weight->data},
      {"transformer.h.9.mlp.down_proj.weight",
       transformer_h_9_mlp_down_proj_weight->data},
      {"transformer.h.9.mlp.down_proj.bias",
       transformer_h_9_mlp_down_proj_bias->data},
      {"transformer.h.10.input_layernorm.weight",
       transformer_h_10_input_layernorm_weight->data},
      {"transformer.h.10.self_attention.query.weight",
       transformer_h_10_self_attention_query_weight->data},
      {"transformer.h.10.self_attention.key_value.weight",
       transformer_h_10_self_attention_key_value_weight->data},
      {"transformer.h.10.self_attention.dense.weight",
       transformer_h_10_self_attention_dense_weight->data},
      {"transformer.h.10.self_attention.dense.bias",
       transformer_h_10_self_attention_dense_bias->data},
      {"transformer.h.10.post_attention_layernorm.weight",
       transformer_h_10_post_attention_layernorm_weight->data},
      {"transformer.h.10.mlp.gate_proj.weight",
       transformer_h_10_mlp_gate_proj_weight->data},
      {"transformer.h.10.mlp.up_proj.weight",
       transformer_h_10_mlp_up_proj_weight->data},
      {"transformer.h.10.mlp.down_proj.weight",
       transformer_h_10_mlp_down_proj_weight->data},
      {"transformer.h.10.mlp.down_proj.bias",
       transformer_h_10_mlp_down_proj_bias->data},
      {"transformer.h.11.input_layernorm.weight",
       transformer_h_11_input_layernorm_weight->data},
      {"transformer.h.11.self_attention.query.weight",
       transformer_h_11_self_attention_query_weight->data},
      {"transformer.h.11.self_attention.key_value.weight",
       transformer_h_11_self_attention_key_value_weight->data},
      {"transformer.h.11.self_attention.dense.weight",
       transformer_h_11_self_attention_dense_weight->data},
      {"transformer.h.11.self_attention.dense.bias",
       transformer_h_11_self_attention_dense_bias->data},
      {"transformer.h.11.post_attention_layernorm.weight",
       transformer_h_11_post_attention_layernorm_weight->data},
      {"transformer.h.11.mlp.gate_proj.weight",
       transformer_h_11_mlp_gate_proj_weight->data},
      {"transformer.h.11.mlp.up_proj.weight",
       transformer_h_11_mlp_up_proj_weight->data},
      {"transformer.h.11.mlp.down_proj.weight",
       transformer_h_11_mlp_down_proj_weight->data},
      {"transformer.h.11.mlp.down_proj.bias",
       transformer_h_11_mlp_down_proj_bias->data},
      {"transformer.h.12.input_layernorm.weight",
       transformer_h_12_input_layernorm_weight->data},
      {"transformer.h.12.self_attention.query.weight",
       transformer_h_12_self_attention_query_weight->data},
      {"transformer.h.12.self_attention.key_value.weight",
       transformer_h_12_self_attention_key_value_weight->data},
      {"transformer.h.12.self_attention.dense.weight",
       transformer_h_12_self_attention_dense_weight->data},
      {"transformer.h.12.self_attention.dense.bias",
       transformer_h_12_self_attention_dense_bias->data},
      {"transformer.h.12.post_attention_layernorm.weight",
       transformer_h_12_post_attention_layernorm_weight->data},
      {"transformer.h.12.mlp.gate_proj.weight",
       transformer_h_12_mlp_gate_proj_weight->data},
      {"transformer.h.12.mlp.up_proj.weight",
       transformer_h_12_mlp_up_proj_weight->data},
      {"transformer.h.12.mlp.down_proj.weight",
       transformer_h_12_mlp_down_proj_weight->data},
      {"transformer.h.12.mlp.down_proj.bias",
       transformer_h_12_mlp_down_proj_bias->data},
      {"transformer.h.13.input_layernorm.weight",
       transformer_h_13_input_layernorm_weight->data},
      {"transformer.h.13.self_attention.query.weight",
       transformer_h_13_self_attention_query_weight->data},
      {"transformer.h.13.self_attention.key_value.weight",
       transformer_h_13_self_attention_key_value_weight->data},
      {"transformer.h.13.self_attention.dense.weight",
       transformer_h_13_self_attention_dense_weight->data},
      {"transformer.h.13.self_attention.dense.bias",
       transformer_h_13_self_attention_dense_bias->data},
      {"transformer.h.13.post_attention_layernorm.weight",
       transformer_h_13_post_attention_layernorm_weight->data},
      {"transformer.h.13.mlp.gate_proj.weight",
       transformer_h_13_mlp_gate_proj_weight->data},
      {"transformer.h.13.mlp.up_proj.weight",
       transformer_h_13_mlp_up_proj_weight->data},
      {"transformer.h.13.mlp.down_proj.weight",
       transformer_h_13_mlp_down_proj_weight->data},
      {"transformer.h.13.mlp.down_proj.bias",
       transformer_h_13_mlp_down_proj_bias->data},
      {"transformer.h.14.input_layernorm.weight",
       transformer_h_14_input_layernorm_weight->data},
      {"transformer.h.14.self_attention.query.weight",
       transformer_h_14_self_attention_query_weight->data},
      {"transformer.h.14.self_attention.key_value.weight",
       transformer_h_14_self_attention_key_value_weight->data},
      {"transformer.h.14.self_attention.dense.weight",
       transformer_h_14_self_attention_dense_weight->data},
      {"transformer.h.14.self_attention.dense.bias",
       transformer_h_14_self_attention_dense_bias->data},
      {"transformer.h.14.post_attention_layernorm.weight",
       transformer_h_14_post_attention_layernorm_weight->data},
      {"transformer.h.14.mlp.gate_proj.weight",
       transformer_h_14_mlp_gate_proj_weight->data},
      {"transformer.h.14.mlp.up_proj.weight",
       transformer_h_14_mlp_up_proj_weight->data},
      {"transformer.h.14.mlp.down_proj.weight",
       transformer_h_14_mlp_down_proj_weight->data},
      {"transformer.h.14.mlp.down_proj.bias",
       transformer_h_14_mlp_down_proj_bias->data},
      {"transformer.h.15.input_layernorm.weight",
       transformer_h_15_input_layernorm_weight->data},
      {"transformer.h.15.self_attention.query.weight",
       transformer_h_15_self_attention_query_weight->data},
      {"transformer.h.15.self_attention.key_value.weight",
       transformer_h_15_self_attention_key_value_weight->data},
      {"transformer.h.15.self_attention.dense.weight",
       transformer_h_15_self_attention_dense_weight->data},
      {"transformer.h.15.self_attention.dense.bias",
       transformer_h_15_self_attention_dense_bias->data},
      {"transformer.h.15.post_attention_layernorm.weight",
       transformer_h_15_post_attention_layernorm_weight->data},
      {"transformer.h.15.mlp.gate_proj.weight",
       transformer_h_15_mlp_gate_proj_weight->data},
      {"transformer.h.15.mlp.up_proj.weight",
       transformer_h_15_mlp_up_proj_weight->data},
      {"transformer.h.15.mlp.down_proj.weight",
       transformer_h_15_mlp_down_proj_weight->data},
      {"transformer.h.15.mlp.down_proj.bias",
       transformer_h_15_mlp_down_proj_bias->data},
      {"transformer.h.16.input_layernorm.weight",
       transformer_h_16_input_layernorm_weight->data},
      {"transformer.h.16.self_attention.query.weight",
       transformer_h_16_self_attention_query_weight->data},
      {"transformer.h.16.self_attention.key_value.weight",
       transformer_h_16_self_attention_key_value_weight->data},
      {"transformer.h.16.self_attention.dense.weight",
       transformer_h_16_self_attention_dense_weight->data},
      {"transformer.h.16.self_attention.dense.bias",
       transformer_h_16_self_attention_dense_bias->data},
      {"transformer.h.16.post_attention_layernorm.weight",
       transformer_h_16_post_attention_layernorm_weight->data},
      {"transformer.h.16.mlp.gate_proj.weight",
       transformer_h_16_mlp_gate_proj_weight->data},
      {"transformer.h.16.mlp.up_proj.weight",
       transformer_h_16_mlp_up_proj_weight->data},
      {"transformer.h.16.mlp.down_proj.weight",
       transformer_h_16_mlp_down_proj_weight->data},
      {"transformer.h.16.mlp.down_proj.bias",
       transformer_h_16_mlp_down_proj_bias->data},
      {"transformer.h.17.input_layernorm.weight",
       transformer_h_17_input_layernorm_weight->data},
      {"transformer.h.17.self_attention.query.weight",
       transformer_h_17_self_attention_query_weight->data},
      {"transformer.h.17.self_attention.key_value.weight",
       transformer_h_17_self_attention_key_value_weight->data},
      {"transformer.h.17.self_attention.dense.weight",
       transformer_h_17_self_attention_dense_weight->data},
      {"transformer.h.17.self_attention.dense.bias",
       transformer_h_17_self_attention_dense_bias->data},
      {"transformer.h.17.post_attention_layernorm.weight",
       transformer_h_17_post_attention_layernorm_weight->data},
      {"transformer.h.17.mlp.gate_proj.weight",
       transformer_h_17_mlp_gate_proj_weight->data},
      {"transformer.h.17.mlp.up_proj.weight",
       transformer_h_17_mlp_up_proj_weight->data},
      {"transformer.h.17.mlp.down_proj.weight",
       transformer_h_17_mlp_down_proj_weight->data},
      {"transformer.h.17.mlp.down_proj.bias",
       transformer_h_17_mlp_down_proj_bias->data},
      {"transformer.h.18.input_layernorm.weight",
       transformer_h_18_input_layernorm_weight->data},
      {"transformer.h.18.self_attention.query.weight",
       transformer_h_18_self_attention_query_weight->data},
      {"transformer.h.18.self_attention.key_value.weight",
       transformer_h_18_self_attention_key_value_weight->data},
      {"transformer.h.18.self_attention.dense.weight",
       transformer_h_18_self_attention_dense_weight->data},
      {"transformer.h.18.self_attention.dense.bias",
       transformer_h_18_self_attention_dense_bias->data},
      {"transformer.h.18.post_attention_layernorm.weight",
       transformer_h_18_post_attention_layernorm_weight->data},
      {"transformer.h.18.mlp.gate_proj.weight",
       transformer_h_18_mlp_gate_proj_weight->data},
      {"transformer.h.18.mlp.up_proj.weight",
       transformer_h_18_mlp_up_proj_weight->data},
      {"transformer.h.18.mlp.down_proj.weight",
       transformer_h_18_mlp_down_proj_weight->data},
      {"transformer.h.18.mlp.down_proj.bias",
       transformer_h_18_mlp_down_proj_bias->data},
      {"transformer.h.19.input_layernorm.weight",
       transformer_h_19_input_layernorm_weight->data},
      {"transformer.h.19.self_attention.query.weight",
       transformer_h_19_self_attention_query_weight->data},
      {"transformer.h.19.self_attention.key_value.weight",
       transformer_h_19_self_attention_key_value_weight->data},
      {"transformer.h.19.self_attention.dense.weight",
       transformer_h_19_self_attention_dense_weight->data},
      {"transformer.h.19.self_attention.dense.bias",
       transformer_h_19_self_attention_dense_bias->data},
      {"transformer.h.19.post_attention_layernorm.weight",
       transformer_h_19_post_attention_layernorm_weight->data},
      {"transformer.h.19.mlp.gate_proj.weight",
       transformer_h_19_mlp_gate_proj_weight->data},
      {"transformer.h.19.mlp.up_proj.weight",
       transformer_h_19_mlp_up_proj_weight->data},
      {"transformer.h.19.mlp.down_proj.weight",
       transformer_h_19_mlp_down_proj_weight->data},
      {"transformer.h.19.mlp.down_proj.bias",
       transformer_h_19_mlp_down_proj_bias->data},
      {"transformer.h.20.input_layernorm.weight",
       transformer_h_20_input_layernorm_weight->data},
      {"transformer.h.20.self_attention.query.weight",
       transformer_h_20_self_attention_query_weight->data},
      {"transformer.h.20.self_attention.key_value.weight",
       transformer_h_20_self_attention_key_value_weight->data},
      {"transformer.h.20.self_attention.dense.weight",
       transformer_h_20_self_attention_dense_weight->data},
      {"transformer.h.20.self_attention.dense.bias",
       transformer_h_20_self_attention_dense_bias->data},
      {"transformer.h.20.post_attention_layernorm.weight",
       transformer_h_20_post_attention_layernorm_weight->data},
      {"transformer.h.20.mlp.gate_proj.weight",
       transformer_h_20_mlp_gate_proj_weight->data},
      {"transformer.h.20.mlp.up_proj.weight",
       transformer_h_20_mlp_up_proj_weight->data},
      {"transformer.h.20.mlp.down_proj.weight",
       transformer_h_20_mlp_down_proj_weight->data},
      {"transformer.h.20.mlp.down_proj.bias",
       transformer_h_20_mlp_down_proj_bias->data},
      {"transformer.h.21.input_layernorm.weight",
       transformer_h_21_input_layernorm_weight->data},
      {"transformer.h.21.self_attention.query.weight",
       transformer_h_21_self_attention_query_weight->data},
      {"transformer.h.21.self_attention.key_value.weight",
       transformer_h_21_self_attention_key_value_weight->data},
      {"transformer.h.21.self_attention.dense.weight",
       transformer_h_21_self_attention_dense_weight->data},
      {"transformer.h.21.self_attention.dense.bias",
       transformer_h_21_self_attention_dense_bias->data},
      {"transformer.h.21.post_attention_layernorm.weight",
       transformer_h_21_post_attention_layernorm_weight->data},
      {"transformer.h.21.mlp.gate_proj.weight",
       transformer_h_21_mlp_gate_proj_weight->data},
      {"transformer.h.21.mlp.up_proj.weight",
       transformer_h_21_mlp_up_proj_weight->data},
      {"transformer.h.21.mlp.down_proj.weight",
       transformer_h_21_mlp_down_proj_weight->data},
      {"transformer.h.21.mlp.down_proj.bias",
       transformer_h_21_mlp_down_proj_bias->data},
      {"transformer.h.22.input_layernorm.weight",
       transformer_h_22_input_layernorm_weight->data},
      {"transformer.h.22.self_attention.query.weight",
       transformer_h_22_self_attention_query_weight->data},
      {"transformer.h.22.self_attention.key_value.weight",
       transformer_h_22_self_attention_key_value_weight->data},
      {"transformer.h.22.self_attention.dense.weight",
       transformer_h_22_self_attention_dense_weight->data},
      {"transformer.h.22.self_attention.dense.bias",
       transformer_h_22_self_attention_dense_bias->data},
      {"transformer.h.22.post_attention_layernorm.weight",
       transformer_h_22_post_attention_layernorm_weight->data},
      {"transformer.h.22.mlp.gate_proj.weight",
       transformer_h_22_mlp_gate_proj_weight->data},
      {"transformer.h.22.mlp.up_proj.weight",
       transformer_h_22_mlp_up_proj_weight->data},
      {"transformer.h.22.mlp.down_proj.weight",
       transformer_h_22_mlp_down_proj_weight->data},
      {"transformer.h.22.mlp.down_proj.bias",
       transformer_h_22_mlp_down_proj_bias->data},
      {"transformer.h.23.input_layernorm.weight",
       transformer_h_23_input_layernorm_weight->data},
      {"transformer.h.23.self_attention.query.weight",
       transformer_h_23_self_attention_query_weight->data},
      {"transformer.h.23.self_attention.key_value.weight",
       transformer_h_23_self_attention_key_value_weight->data},
      {"transformer.h.23.self_attention.dense.weight",
       transformer_h_23_self_attention_dense_weight->data},
      {"transformer.h.23.self_attention.dense.bias",
       transformer_h_23_self_attention_dense_bias->data},
      {"transformer.h.23.post_attention_layernorm.weight",
       transformer_h_23_post_attention_layernorm_weight->data},
      {"transformer.h.23.mlp.gate_proj.weight",
       transformer_h_23_mlp_gate_proj_weight->data},
      {"transformer.h.23.mlp.up_proj.weight",
       transformer_h_23_mlp_up_proj_weight->data},
      {"transformer.h.23.mlp.down_proj.weight",
       transformer_h_23_mlp_down_proj_weight->data},
      {"transformer.h.23.mlp.down_proj.bias",
       transformer_h_23_mlp_down_proj_bias->data},
      {"transformer.h.24.input_layernorm.weight",
       transformer_h_24_input_layernorm_weight->data},
      {"transformer.h.24.self_attention.query.weight",
       transformer_h_24_self_attention_query_weight->data},
      {"transformer.h.24.self_attention.key_value.weight",
       transformer_h_24_self_attention_key_value_weight->data},
      {"transformer.h.24.self_attention.dense.weight",
       transformer_h_24_self_attention_dense_weight->data},
      {"transformer.h.24.self_attention.dense.bias",
       transformer_h_24_self_attention_dense_bias->data},
      {"transformer.h.24.post_attention_layernorm.weight",
       transformer_h_24_post_attention_layernorm_weight->data},
      {"transformer.h.24.mlp.gate_proj.weight",
       transformer_h_24_mlp_gate_proj_weight->data},
      {"transformer.h.24.mlp.up_proj.weight",
       transformer_h_24_mlp_up_proj_weight->data},
      {"transformer.h.24.mlp.down_proj.weight",
       transformer_h_24_mlp_down_proj_weight->data},
      {"transformer.h.24.mlp.down_proj.bias",
       transformer_h_24_mlp_down_proj_bias->data},
      {"transformer.h.25.input_layernorm.weight",
       transformer_h_25_input_layernorm_weight->data},
      {"transformer.h.25.self_attention.query.weight",
       transformer_h_25_self_attention_query_weight->data},
      {"transformer.h.25.self_attention.key_value.weight",
       transformer_h_25_self_attention_key_value_weight->data},
      {"transformer.h.25.self_attention.dense.weight",
       transformer_h_25_self_attention_dense_weight->data},
      {"transformer.h.25.self_attention.dense.bias",
       transformer_h_25_self_attention_dense_bias->data},
      {"transformer.h.25.post_attention_layernorm.weight",
       transformer_h_25_post_attention_layernorm_weight->data},
      {"transformer.h.25.mlp.gate_proj.weight",
       transformer_h_25_mlp_gate_proj_weight->data},
      {"transformer.h.25.mlp.up_proj.weight",
       transformer_h_25_mlp_up_proj_weight->data},
      {"transformer.h.25.mlp.down_proj.weight",
       transformer_h_25_mlp_down_proj_weight->data},
      {"transformer.h.25.mlp.down_proj.bias",
       transformer_h_25_mlp_down_proj_bias->data},
      {"transformer.h.26.input_layernorm.weight",
       transformer_h_26_input_layernorm_weight->data},
      {"transformer.h.26.self_attention.query.weight",
       transformer_h_26_self_attention_query_weight->data},
      {"transformer.h.26.self_attention.key_value.weight",
       transformer_h_26_self_attention_key_value_weight->data},
      {"transformer.h.26.self_attention.dense.weight",
       transformer_h_26_self_attention_dense_weight->data},
      {"transformer.h.26.self_attention.dense.bias",
       transformer_h_26_self_attention_dense_bias->data},
      {"transformer.h.26.post_attention_layernorm.weight",
       transformer_h_26_post_attention_layernorm_weight->data},
      {"transformer.h.26.mlp.gate_proj.weight",
       transformer_h_26_mlp_gate_proj_weight->data},
      {"transformer.h.26.mlp.up_proj.weight",
       transformer_h_26_mlp_up_proj_weight->data},
      {"transformer.h.26.mlp.down_proj.weight",
       transformer_h_26_mlp_down_proj_weight->data},
      {"transformer.h.26.mlp.down_proj.bias",
       transformer_h_26_mlp_down_proj_bias->data},
      {"transformer.h.27.input_layernorm.weight",
       transformer_h_27_input_layernorm_weight->data},
      {"transformer.h.27.self_attention.query.weight",
       transformer_h_27_self_attention_query_weight->data},
      {"transformer.h.27.self_attention.key_value.weight",
       transformer_h_27_self_attention_key_value_weight->data},
      {"transformer.h.27.self_attention.dense.weight",
       transformer_h_27_self_attention_dense_weight->data},
      {"transformer.h.27.self_attention.dense.bias",
       transformer_h_27_self_attention_dense_bias->data},
      {"transformer.h.27.post_attention_layernorm.weight",
       transformer_h_27_post_attention_layernorm_weight->data},
      {"transformer.h.27.mlp.gate_proj.weight",
       transformer_h_27_mlp_gate_proj_weight->data},
      {"transformer.h.27.mlp.up_proj.weight",
       transformer_h_27_mlp_up_proj_weight->data},
      {"transformer.h.27.mlp.down_proj.weight",
       transformer_h_27_mlp_down_proj_weight->data},
      {"transformer.h.27.mlp.down_proj.bias",
       transformer_h_27_mlp_down_proj_bias->data},
      {"transformer.h.28.input_layernorm.weight",
       transformer_h_28_input_layernorm_weight->data},
      {"transformer.h.28.self_attention.query.weight",
       transformer_h_28_self_attention_query_weight->data},
      {"transformer.h.28.self_attention.key_value.weight",
       transformer_h_28_self_attention_key_value_weight->data},
      {"transformer.h.28.self_attention.dense.weight",
       transformer_h_28_self_attention_dense_weight->data},
      {"transformer.h.28.self_attention.dense.bias",
       transformer_h_28_self_attention_dense_bias->data},
      {"transformer.h.28.post_attention_layernorm.weight",
       transformer_h_28_post_attention_layernorm_weight->data},
      {"transformer.h.28.mlp.gate_proj.weight",
       transformer_h_28_mlp_gate_proj_weight->data},
      {"transformer.h.28.mlp.up_proj.weight",
       transformer_h_28_mlp_up_proj_weight->data},
      {"transformer.h.28.mlp.down_proj.weight",
       transformer_h_28_mlp_down_proj_weight->data},
      {"transformer.h.28.mlp.down_proj.bias",
       transformer_h_28_mlp_down_proj_bias->data},
      {"transformer.h.29.input_layernorm.weight",
       transformer_h_29_input_layernorm_weight->data},
      {"transformer.h.29.self_attention.query.weight",
       transformer_h_29_self_attention_query_weight->data},
      {"transformer.h.29.self_attention.key_value.weight",
       transformer_h_29_self_attention_key_value_weight->data},
      {"transformer.h.29.self_attention.dense.weight",
       transformer_h_29_self_attention_dense_weight->data},
      {"transformer.h.29.self_attention.dense.bias",
       transformer_h_29_self_attention_dense_bias->data},
      {"transformer.h.29.post_attention_layernorm.weight",
       transformer_h_29_post_attention_layernorm_weight->data},
      {"transformer.h.29.mlp.gate_proj.weight",
       transformer_h_29_mlp_gate_proj_weight->data},
      {"transformer.h.29.mlp.up_proj.weight",
       transformer_h_29_mlp_up_proj_weight->data},
      {"transformer.h.29.mlp.down_proj.weight",
       transformer_h_29_mlp_down_proj_weight->data},
      {"transformer.h.29.mlp.down_proj.bias",
       transformer_h_29_mlp_down_proj_bias->data},
      {"transformer.h.30.input_layernorm.weight",
       transformer_h_30_input_layernorm_weight->data},
      {"transformer.h.30.self_attention.query.weight",
       transformer_h_30_self_attention_query_weight->data},
      {"transformer.h.30.self_attention.key_value.weight",
       transformer_h_30_self_attention_key_value_weight->data},
      {"transformer.h.30.self_attention.dense.weight",
       transformer_h_30_self_attention_dense_weight->data},
      {"transformer.h.30.self_attention.dense.bias",
       transformer_h_30_self_attention_dense_bias->data},
      {"transformer.h.30.post_attention_layernorm.weight",
       transformer_h_30_post_attention_layernorm_weight->data},
      {"transformer.h.30.mlp.gate_proj.weight",
       transformer_h_30_mlp_gate_proj_weight->data},
      {"transformer.h.30.mlp.up_proj.weight",
       transformer_h_30_mlp_up_proj_weight->data},
      {"transformer.h.30.mlp.down_proj.weight",
       transformer_h_30_mlp_down_proj_weight->data},
      {"transformer.h.30.mlp.down_proj.bias",
       transformer_h_30_mlp_down_proj_bias->data},
      {"transformer.h.31.input_layernorm.weight",
       transformer_h_31_input_layernorm_weight->data},
      {"transformer.h.31.self_attention.query.weight",
       transformer_h_31_self_attention_query_weight->data},
      {"transformer.h.31.self_attention.key_value.weight",
       transformer_h_31_self_attention_key_value_weight->data},
      {"transformer.h.31.self_attention.dense.weight",
       transformer_h_31_self_attention_dense_weight->data},
      {"transformer.h.31.self_attention.dense.bias",
       transformer_h_31_self_attention_dense_bias->data},
      {"transformer.h.31.post_attention_layernorm.weight",
       transformer_h_31_post_attention_layernorm_weight->data},
      {"transformer.h.31.mlp.gate_proj.weight",
       transformer_h_31_mlp_gate_proj_weight->data},
      {"transformer.h.31.mlp.up_proj.weight",
       transformer_h_31_mlp_up_proj_weight->data},
      {"transformer.h.31.mlp.down_proj.weight",
       transformer_h_31_mlp_down_proj_weight->data},
      {"transformer.h.31.mlp.down_proj.bias",
       transformer_h_31_mlp_down_proj_bias->data},
      {"transformer.h.32.input_layernorm.weight",
       transformer_h_32_input_layernorm_weight->data},
      {"transformer.h.32.self_attention.query.weight",
       transformer_h_32_self_attention_query_weight->data},
      {"transformer.h.32.self_attention.key_value.weight",
       transformer_h_32_self_attention_key_value_weight->data},
      {"transformer.h.32.self_attention.dense.weight",
       transformer_h_32_self_attention_dense_weight->data},
      {"transformer.h.32.self_attention.dense.bias",
       transformer_h_32_self_attention_dense_bias->data},
      {"transformer.h.32.post_attention_layernorm.weight",
       transformer_h_32_post_attention_layernorm_weight->data},
      {"transformer.h.32.mlp.gate_proj.weight",
       transformer_h_32_mlp_gate_proj_weight->data},
      {"transformer.h.32.mlp.up_proj.weight",
       transformer_h_32_mlp_up_proj_weight->data},
      {"transformer.h.32.mlp.down_proj.weight",
       transformer_h_32_mlp_down_proj_weight->data},
      {"transformer.h.32.mlp.down_proj.bias",
       transformer_h_32_mlp_down_proj_bias->data},
      {"transformer.h.33.input_layernorm.weight",
       transformer_h_33_input_layernorm_weight->data},
      {"transformer.h.33.self_attention.query.weight",
       transformer_h_33_self_attention_query_weight->data},
      {"transformer.h.33.self_attention.key_value.weight",
       transformer_h_33_self_attention_key_value_weight->data},
      {"transformer.h.33.self_attention.dense.weight",
       transformer_h_33_self_attention_dense_weight->data},
      {"transformer.h.33.self_attention.dense.bias",
       transformer_h_33_self_attention_dense_bias->data},
      {"transformer.h.33.post_attention_layernorm.weight",
       transformer_h_33_post_attention_layernorm_weight->data},
      {"transformer.h.33.mlp.gate_proj.weight",
       transformer_h_33_mlp_gate_proj_weight->data},
      {"transformer.h.33.mlp.up_proj.weight",
       transformer_h_33_mlp_up_proj_weight->data},
      {"transformer.h.33.mlp.down_proj.weight",
       transformer_h_33_mlp_down_proj_weight->data},
      {"transformer.h.33.mlp.down_proj.bias",
       transformer_h_33_mlp_down_proj_bias->data},
      {"transformer.h.34.input_layernorm.weight",
       transformer_h_34_input_layernorm_weight->data},
      {"transformer.h.34.self_attention.query.weight",
       transformer_h_34_self_attention_query_weight->data},
      {"transformer.h.34.self_attention.key_value.weight",
       transformer_h_34_self_attention_key_value_weight->data},
      {"transformer.h.34.self_attention.dense.weight",
       transformer_h_34_self_attention_dense_weight->data},
      {"transformer.h.34.self_attention.dense.bias",
       transformer_h_34_self_attention_dense_bias->data},
      {"transformer.h.34.post_attention_layernorm.weight",
       transformer_h_34_post_attention_layernorm_weight->data},
      {"transformer.h.34.mlp.gate_proj.weight",
       transformer_h_34_mlp_gate_proj_weight->data},
      {"transformer.h.34.mlp.up_proj.weight",
       transformer_h_34_mlp_up_proj_weight->data},
      {"transformer.h.34.mlp.down_proj.weight",
       transformer_h_34_mlp_down_proj_weight->data},
      {"transformer.h.34.mlp.down_proj.bias",
       transformer_h_34_mlp_down_proj_bias->data},
      {"transformer.h.35.input_layernorm.weight",
       transformer_h_35_input_layernorm_weight->data},
      {"transformer.h.35.self_attention.query.weight",
       transformer_h_35_self_attention_query_weight->data},
      {"transformer.h.35.self_attention.key_value.weight",
       transformer_h_35_self_attention_key_value_weight->data},
      {"transformer.h.35.self_attention.dense.weight",
       transformer_h_35_self_attention_dense_weight->data},
      {"transformer.h.35.self_attention.dense.bias",
       transformer_h_35_self_attention_dense_bias->data},
      {"transformer.h.35.post_attention_layernorm.weight",
       transformer_h_35_post_attention_layernorm_weight->data},
      {"transformer.h.35.mlp.gate_proj.weight",
       transformer_h_35_mlp_gate_proj_weight->data},
      {"transformer.h.35.mlp.up_proj.weight",
       transformer_h_35_mlp_up_proj_weight->data},
      {"transformer.h.35.mlp.down_proj.weight",
       transformer_h_35_mlp_down_proj_weight->data},
      {"transformer.h.35.mlp.down_proj.bias",
       transformer_h_35_mlp_down_proj_bias->data},
      {"transformer.h.36.input_layernorm.weight",
       transformer_h_36_input_layernorm_weight->data},
      {"transformer.h.36.self_attention.query.weight",
       transformer_h_36_self_attention_query_weight->data},
      {"transformer.h.36.self_attention.key_value.weight",
       transformer_h_36_self_attention_key_value_weight->data},
      {"transformer.h.36.self_attention.dense.weight",
       transformer_h_36_self_attention_dense_weight->data},
      {"transformer.h.36.self_attention.dense.bias",
       transformer_h_36_self_attention_dense_bias->data},
      {"transformer.h.36.post_attention_layernorm.weight",
       transformer_h_36_post_attention_layernorm_weight->data},
      {"transformer.h.36.mlp.gate_proj.weight",
       transformer_h_36_mlp_gate_proj_weight->data},
      {"transformer.h.36.mlp.up_proj.weight",
       transformer_h_36_mlp_up_proj_weight->data},
      {"transformer.h.36.mlp.down_proj.weight",
       transformer_h_36_mlp_down_proj_weight->data},
      {"transformer.h.36.mlp.down_proj.bias",
       transformer_h_36_mlp_down_proj_bias->data},
      {"transformer.h.37.input_layernorm.weight",
       transformer_h_37_input_layernorm_weight->data},
      {"transformer.h.37.self_attention.query.weight",
       transformer_h_37_self_attention_query_weight->data},
      {"transformer.h.37.self_attention.key_value.weight",
       transformer_h_37_self_attention_key_value_weight->data},
      {"transformer.h.37.self_attention.dense.weight",
       transformer_h_37_self_attention_dense_weight->data},
      {"transformer.h.37.self_attention.dense.bias",
       transformer_h_37_self_attention_dense_bias->data},
      {"transformer.h.37.post_attention_layernorm.weight",
       transformer_h_37_post_attention_layernorm_weight->data},
      {"transformer.h.37.mlp.gate_proj.weight",
       transformer_h_37_mlp_gate_proj_weight->data},
      {"transformer.h.37.mlp.up_proj.weight",
       transformer_h_37_mlp_up_proj_weight->data},
      {"transformer.h.37.mlp.down_proj.weight",
       transformer_h_37_mlp_down_proj_weight->data},
      {"transformer.h.37.mlp.down_proj.bias",
       transformer_h_37_mlp_down_proj_bias->data},
      {"transformer.ln_f.weight", transformer_ln_f_weight->data},
      {"lm_head.weight", lm_head_weight->data}};
  mix::utils::load_model_f16(model_names, param_and_loc);
}
void delete_all_globals() {
  delete_transformer_word_embeddings_weight();
  delete_transformer_h_0_input_layernorm_weight();
  delete_transformer_h_0_self_attention_query_weight();
  delete_transformer_h_0_self_attention_key_value_weight();
  delete_transformer_h_0_self_attention_dense_weight();
  delete_transformer_h_0_self_attention_dense_bias();
  delete_transformer_h_0_post_attention_layernorm_weight();
  delete_transformer_h_0_mlp_gate_proj_weight();
  delete_transformer_h_0_mlp_up_proj_weight();
  delete_transformer_h_0_mlp_down_proj_weight();
  delete_transformer_h_0_mlp_down_proj_bias();
  delete_transformer_h_1_input_layernorm_weight();
  delete_transformer_h_1_self_attention_query_weight();
  delete_transformer_h_1_self_attention_key_value_weight();
  delete_transformer_h_1_self_attention_dense_weight();
  delete_transformer_h_1_self_attention_dense_bias();
  delete_transformer_h_1_post_attention_layernorm_weight();
  delete_transformer_h_1_mlp_gate_proj_weight();
  delete_transformer_h_1_mlp_up_proj_weight();
  delete_transformer_h_1_mlp_down_proj_weight();
  delete_transformer_h_1_mlp_down_proj_bias();
  delete_transformer_h_2_input_layernorm_weight();
  delete_transformer_h_2_self_attention_query_weight();
  delete_transformer_h_2_self_attention_key_value_weight();
  delete_transformer_h_2_self_attention_dense_weight();
  delete_transformer_h_2_self_attention_dense_bias();
  delete_transformer_h_2_post_attention_layernorm_weight();
  delete_transformer_h_2_mlp_gate_proj_weight();
  delete_transformer_h_2_mlp_up_proj_weight();
  delete_transformer_h_2_mlp_down_proj_weight();
  delete_transformer_h_2_mlp_down_proj_bias();
  delete_transformer_h_3_input_layernorm_weight();
  delete_transformer_h_3_self_attention_query_weight();
  delete_transformer_h_3_self_attention_key_value_weight();
  delete_transformer_h_3_self_attention_dense_weight();
  delete_transformer_h_3_self_attention_dense_bias();
  delete_transformer_h_3_post_attention_layernorm_weight();
  delete_transformer_h_3_mlp_gate_proj_weight();
  delete_transformer_h_3_mlp_up_proj_weight();
  delete_transformer_h_3_mlp_down_proj_weight();
  delete_transformer_h_3_mlp_down_proj_bias();
  delete_transformer_h_4_input_layernorm_weight();
  delete_transformer_h_4_self_attention_query_weight();
  delete_transformer_h_4_self_attention_key_value_weight();
  delete_transformer_h_4_self_attention_dense_weight();
  delete_transformer_h_4_self_attention_dense_bias();
  delete_transformer_h_4_post_attention_layernorm_weight();
  delete_transformer_h_4_mlp_gate_proj_weight();
  delete_transformer_h_4_mlp_up_proj_weight();
  delete_transformer_h_4_mlp_down_proj_weight();
  delete_transformer_h_4_mlp_down_proj_bias();
  delete_transformer_h_5_input_layernorm_weight();
  delete_transformer_h_5_self_attention_query_weight();
  delete_transformer_h_5_self_attention_key_value_weight();
  delete_transformer_h_5_self_attention_dense_weight();
  delete_transformer_h_5_self_attention_dense_bias();
  delete_transformer_h_5_post_attention_layernorm_weight();
  delete_transformer_h_5_mlp_gate_proj_weight();
  delete_transformer_h_5_mlp_up_proj_weight();
  delete_transformer_h_5_mlp_down_proj_weight();
  delete_transformer_h_5_mlp_down_proj_bias();
  delete_transformer_h_6_input_layernorm_weight();
  delete_transformer_h_6_self_attention_query_weight();
  delete_transformer_h_6_self_attention_key_value_weight();
  delete_transformer_h_6_self_attention_dense_weight();
  delete_transformer_h_6_self_attention_dense_bias();
  delete_transformer_h_6_post_attention_layernorm_weight();
  delete_transformer_h_6_mlp_gate_proj_weight();
  delete_transformer_h_6_mlp_up_proj_weight();
  delete_transformer_h_6_mlp_down_proj_weight();
  delete_transformer_h_6_mlp_down_proj_bias();
  delete_transformer_h_7_input_layernorm_weight();
  delete_transformer_h_7_self_attention_query_weight();
  delete_transformer_h_7_self_attention_key_value_weight();
  delete_transformer_h_7_self_attention_dense_weight();
  delete_transformer_h_7_self_attention_dense_bias();
  delete_transformer_h_7_post_attention_layernorm_weight();
  delete_transformer_h_7_mlp_gate_proj_weight();
  delete_transformer_h_7_mlp_up_proj_weight();
  delete_transformer_h_7_mlp_down_proj_weight();
  delete_transformer_h_7_mlp_down_proj_bias();
  delete_transformer_h_8_input_layernorm_weight();
  delete_transformer_h_8_self_attention_query_weight();
  delete_transformer_h_8_self_attention_key_value_weight();
  delete_transformer_h_8_self_attention_dense_weight();
  delete_transformer_h_8_self_attention_dense_bias();
  delete_transformer_h_8_post_attention_layernorm_weight();
  delete_transformer_h_8_mlp_gate_proj_weight();
  delete_transformer_h_8_mlp_up_proj_weight();
  delete_transformer_h_8_mlp_down_proj_weight();
  delete_transformer_h_8_mlp_down_proj_bias();
  delete_transformer_h_9_input_layernorm_weight();
  delete_transformer_h_9_self_attention_query_weight();
  delete_transformer_h_9_self_attention_key_value_weight();
  delete_transformer_h_9_self_attention_dense_weight();
  delete_transformer_h_9_self_attention_dense_bias();
  delete_transformer_h_9_post_attention_layernorm_weight();
  delete_transformer_h_9_mlp_gate_proj_weight();
  delete_transformer_h_9_mlp_up_proj_weight();
  delete_transformer_h_9_mlp_down_proj_weight();
  delete_transformer_h_9_mlp_down_proj_bias();
  delete_transformer_h_10_input_layernorm_weight();
  delete_transformer_h_10_self_attention_query_weight();
  delete_transformer_h_10_self_attention_key_value_weight();
  delete_transformer_h_10_self_attention_dense_weight();
  delete_transformer_h_10_self_attention_dense_bias();
  delete_transformer_h_10_post_attention_layernorm_weight();
  delete_transformer_h_10_mlp_gate_proj_weight();
  delete_transformer_h_10_mlp_up_proj_weight();
  delete_transformer_h_10_mlp_down_proj_weight();
  delete_transformer_h_10_mlp_down_proj_bias();
  delete_transformer_h_11_input_layernorm_weight();
  delete_transformer_h_11_self_attention_query_weight();
  delete_transformer_h_11_self_attention_key_value_weight();
  delete_transformer_h_11_self_attention_dense_weight();
  delete_transformer_h_11_self_attention_dense_bias();
  delete_transformer_h_11_post_attention_layernorm_weight();
  delete_transformer_h_11_mlp_gate_proj_weight();
  delete_transformer_h_11_mlp_up_proj_weight();
  delete_transformer_h_11_mlp_down_proj_weight();
  delete_transformer_h_11_mlp_down_proj_bias();
  delete_transformer_h_12_input_layernorm_weight();
  delete_transformer_h_12_self_attention_query_weight();
  delete_transformer_h_12_self_attention_key_value_weight();
  delete_transformer_h_12_self_attention_dense_weight();
  delete_transformer_h_12_self_attention_dense_bias();
  delete_transformer_h_12_post_attention_layernorm_weight();
  delete_transformer_h_12_mlp_gate_proj_weight();
  delete_transformer_h_12_mlp_up_proj_weight();
  delete_transformer_h_12_mlp_down_proj_weight();
  delete_transformer_h_12_mlp_down_proj_bias();
  delete_transformer_h_13_input_layernorm_weight();
  delete_transformer_h_13_self_attention_query_weight();
  delete_transformer_h_13_self_attention_key_value_weight();
  delete_transformer_h_13_self_attention_dense_weight();
  delete_transformer_h_13_self_attention_dense_bias();
  delete_transformer_h_13_post_attention_layernorm_weight();
  delete_transformer_h_13_mlp_gate_proj_weight();
  delete_transformer_h_13_mlp_up_proj_weight();
  delete_transformer_h_13_mlp_down_proj_weight();
  delete_transformer_h_13_mlp_down_proj_bias();
  delete_transformer_h_14_input_layernorm_weight();
  delete_transformer_h_14_self_attention_query_weight();
  delete_transformer_h_14_self_attention_key_value_weight();
  delete_transformer_h_14_self_attention_dense_weight();
  delete_transformer_h_14_self_attention_dense_bias();
  delete_transformer_h_14_post_attention_layernorm_weight();
  delete_transformer_h_14_mlp_gate_proj_weight();
  delete_transformer_h_14_mlp_up_proj_weight();
  delete_transformer_h_14_mlp_down_proj_weight();
  delete_transformer_h_14_mlp_down_proj_bias();
  delete_transformer_h_15_input_layernorm_weight();
  delete_transformer_h_15_self_attention_query_weight();
  delete_transformer_h_15_self_attention_key_value_weight();
  delete_transformer_h_15_self_attention_dense_weight();
  delete_transformer_h_15_self_attention_dense_bias();
  delete_transformer_h_15_post_attention_layernorm_weight();
  delete_transformer_h_15_mlp_gate_proj_weight();
  delete_transformer_h_15_mlp_up_proj_weight();
  delete_transformer_h_15_mlp_down_proj_weight();
  delete_transformer_h_15_mlp_down_proj_bias();
  delete_transformer_h_16_input_layernorm_weight();
  delete_transformer_h_16_self_attention_query_weight();
  delete_transformer_h_16_self_attention_key_value_weight();
  delete_transformer_h_16_self_attention_dense_weight();
  delete_transformer_h_16_self_attention_dense_bias();
  delete_transformer_h_16_post_attention_layernorm_weight();
  delete_transformer_h_16_mlp_gate_proj_weight();
  delete_transformer_h_16_mlp_up_proj_weight();
  delete_transformer_h_16_mlp_down_proj_weight();
  delete_transformer_h_16_mlp_down_proj_bias();
  delete_transformer_h_17_input_layernorm_weight();
  delete_transformer_h_17_self_attention_query_weight();
  delete_transformer_h_17_self_attention_key_value_weight();
  delete_transformer_h_17_self_attention_dense_weight();
  delete_transformer_h_17_self_attention_dense_bias();
  delete_transformer_h_17_post_attention_layernorm_weight();
  delete_transformer_h_17_mlp_gate_proj_weight();
  delete_transformer_h_17_mlp_up_proj_weight();
  delete_transformer_h_17_mlp_down_proj_weight();
  delete_transformer_h_17_mlp_down_proj_bias();
  delete_transformer_h_18_input_layernorm_weight();
  delete_transformer_h_18_self_attention_query_weight();
  delete_transformer_h_18_self_attention_key_value_weight();
  delete_transformer_h_18_self_attention_dense_weight();
  delete_transformer_h_18_self_attention_dense_bias();
  delete_transformer_h_18_post_attention_layernorm_weight();
  delete_transformer_h_18_mlp_gate_proj_weight();
  delete_transformer_h_18_mlp_up_proj_weight();
  delete_transformer_h_18_mlp_down_proj_weight();
  delete_transformer_h_18_mlp_down_proj_bias();
  delete_transformer_h_19_input_layernorm_weight();
  delete_transformer_h_19_self_attention_query_weight();
  delete_transformer_h_19_self_attention_key_value_weight();
  delete_transformer_h_19_self_attention_dense_weight();
  delete_transformer_h_19_self_attention_dense_bias();
  delete_transformer_h_19_post_attention_layernorm_weight();
  delete_transformer_h_19_mlp_gate_proj_weight();
  delete_transformer_h_19_mlp_up_proj_weight();
  delete_transformer_h_19_mlp_down_proj_weight();
  delete_transformer_h_19_mlp_down_proj_bias();
  delete_transformer_h_20_input_layernorm_weight();
  delete_transformer_h_20_self_attention_query_weight();
  delete_transformer_h_20_self_attention_key_value_weight();
  delete_transformer_h_20_self_attention_dense_weight();
  delete_transformer_h_20_self_attention_dense_bias();
  delete_transformer_h_20_post_attention_layernorm_weight();
  delete_transformer_h_20_mlp_gate_proj_weight();
  delete_transformer_h_20_mlp_up_proj_weight();
  delete_transformer_h_20_mlp_down_proj_weight();
  delete_transformer_h_20_mlp_down_proj_bias();
  delete_transformer_h_21_input_layernorm_weight();
  delete_transformer_h_21_self_attention_query_weight();
  delete_transformer_h_21_self_attention_key_value_weight();
  delete_transformer_h_21_self_attention_dense_weight();
  delete_transformer_h_21_self_attention_dense_bias();
  delete_transformer_h_21_post_attention_layernorm_weight();
  delete_transformer_h_21_mlp_gate_proj_weight();
  delete_transformer_h_21_mlp_up_proj_weight();
  delete_transformer_h_21_mlp_down_proj_weight();
  delete_transformer_h_21_mlp_down_proj_bias();
  delete_transformer_h_22_input_layernorm_weight();
  delete_transformer_h_22_self_attention_query_weight();
  delete_transformer_h_22_self_attention_key_value_weight();
  delete_transformer_h_22_self_attention_dense_weight();
  delete_transformer_h_22_self_attention_dense_bias();
  delete_transformer_h_22_post_attention_layernorm_weight();
  delete_transformer_h_22_mlp_gate_proj_weight();
  delete_transformer_h_22_mlp_up_proj_weight();
  delete_transformer_h_22_mlp_down_proj_weight();
  delete_transformer_h_22_mlp_down_proj_bias();
  delete_transformer_h_23_input_layernorm_weight();
  delete_transformer_h_23_self_attention_query_weight();
  delete_transformer_h_23_self_attention_key_value_weight();
  delete_transformer_h_23_self_attention_dense_weight();
  delete_transformer_h_23_self_attention_dense_bias();
  delete_transformer_h_23_post_attention_layernorm_weight();
  delete_transformer_h_23_mlp_gate_proj_weight();
  delete_transformer_h_23_mlp_up_proj_weight();
  delete_transformer_h_23_mlp_down_proj_weight();
  delete_transformer_h_23_mlp_down_proj_bias();
  delete_transformer_h_24_input_layernorm_weight();
  delete_transformer_h_24_self_attention_query_weight();
  delete_transformer_h_24_self_attention_key_value_weight();
  delete_transformer_h_24_self_attention_dense_weight();
  delete_transformer_h_24_self_attention_dense_bias();
  delete_transformer_h_24_post_attention_layernorm_weight();
  delete_transformer_h_24_mlp_gate_proj_weight();
  delete_transformer_h_24_mlp_up_proj_weight();
  delete_transformer_h_24_mlp_down_proj_weight();
  delete_transformer_h_24_mlp_down_proj_bias();
  delete_transformer_h_25_input_layernorm_weight();
  delete_transformer_h_25_self_attention_query_weight();
  delete_transformer_h_25_self_attention_key_value_weight();
  delete_transformer_h_25_self_attention_dense_weight();
  delete_transformer_h_25_self_attention_dense_bias();
  delete_transformer_h_25_post_attention_layernorm_weight();
  delete_transformer_h_25_mlp_gate_proj_weight();
  delete_transformer_h_25_mlp_up_proj_weight();
  delete_transformer_h_25_mlp_down_proj_weight();
  delete_transformer_h_25_mlp_down_proj_bias();
  delete_transformer_h_26_input_layernorm_weight();
  delete_transformer_h_26_self_attention_query_weight();
  delete_transformer_h_26_self_attention_key_value_weight();
  delete_transformer_h_26_self_attention_dense_weight();
  delete_transformer_h_26_self_attention_dense_bias();
  delete_transformer_h_26_post_attention_layernorm_weight();
  delete_transformer_h_26_mlp_gate_proj_weight();
  delete_transformer_h_26_mlp_up_proj_weight();
  delete_transformer_h_26_mlp_down_proj_weight();
  delete_transformer_h_26_mlp_down_proj_bias();
  delete_transformer_h_27_input_layernorm_weight();
  delete_transformer_h_27_self_attention_query_weight();
  delete_transformer_h_27_self_attention_key_value_weight();
  delete_transformer_h_27_self_attention_dense_weight();
  delete_transformer_h_27_self_attention_dense_bias();
  delete_transformer_h_27_post_attention_layernorm_weight();
  delete_transformer_h_27_mlp_gate_proj_weight();
  delete_transformer_h_27_mlp_up_proj_weight();
  delete_transformer_h_27_mlp_down_proj_weight();
  delete_transformer_h_27_mlp_down_proj_bias();
  delete_transformer_h_28_input_layernorm_weight();
  delete_transformer_h_28_self_attention_query_weight();
  delete_transformer_h_28_self_attention_key_value_weight();
  delete_transformer_h_28_self_attention_dense_weight();
  delete_transformer_h_28_self_attention_dense_bias();
  delete_transformer_h_28_post_attention_layernorm_weight();
  delete_transformer_h_28_mlp_gate_proj_weight();
  delete_transformer_h_28_mlp_up_proj_weight();
  delete_transformer_h_28_mlp_down_proj_weight();
  delete_transformer_h_28_mlp_down_proj_bias();
  delete_transformer_h_29_input_layernorm_weight();
  delete_transformer_h_29_self_attention_query_weight();
  delete_transformer_h_29_self_attention_key_value_weight();
  delete_transformer_h_29_self_attention_dense_weight();
  delete_transformer_h_29_self_attention_dense_bias();
  delete_transformer_h_29_post_attention_layernorm_weight();
  delete_transformer_h_29_mlp_gate_proj_weight();
  delete_transformer_h_29_mlp_up_proj_weight();
  delete_transformer_h_29_mlp_down_proj_weight();
  delete_transformer_h_29_mlp_down_proj_bias();
  delete_transformer_h_30_input_layernorm_weight();
  delete_transformer_h_30_self_attention_query_weight();
  delete_transformer_h_30_self_attention_key_value_weight();
  delete_transformer_h_30_self_attention_dense_weight();
  delete_transformer_h_30_self_attention_dense_bias();
  delete_transformer_h_30_post_attention_layernorm_weight();
  delete_transformer_h_30_mlp_gate_proj_weight();
  delete_transformer_h_30_mlp_up_proj_weight();
  delete_transformer_h_30_mlp_down_proj_weight();
  delete_transformer_h_30_mlp_down_proj_bias();
  delete_transformer_h_31_input_layernorm_weight();
  delete_transformer_h_31_self_attention_query_weight();
  delete_transformer_h_31_self_attention_key_value_weight();
  delete_transformer_h_31_self_attention_dense_weight();
  delete_transformer_h_31_self_attention_dense_bias();
  delete_transformer_h_31_post_attention_layernorm_weight();
  delete_transformer_h_31_mlp_gate_proj_weight();
  delete_transformer_h_31_mlp_up_proj_weight();
  delete_transformer_h_31_mlp_down_proj_weight();
  delete_transformer_h_31_mlp_down_proj_bias();
  delete_transformer_h_32_input_layernorm_weight();
  delete_transformer_h_32_self_attention_query_weight();
  delete_transformer_h_32_self_attention_key_value_weight();
  delete_transformer_h_32_self_attention_dense_weight();
  delete_transformer_h_32_self_attention_dense_bias();
  delete_transformer_h_32_post_attention_layernorm_weight();
  delete_transformer_h_32_mlp_gate_proj_weight();
  delete_transformer_h_32_mlp_up_proj_weight();
  delete_transformer_h_32_mlp_down_proj_weight();
  delete_transformer_h_32_mlp_down_proj_bias();
  delete_transformer_h_33_input_layernorm_weight();
  delete_transformer_h_33_self_attention_query_weight();
  delete_transformer_h_33_self_attention_key_value_weight();
  delete_transformer_h_33_self_attention_dense_weight();
  delete_transformer_h_33_self_attention_dense_bias();
  delete_transformer_h_33_post_attention_layernorm_weight();
  delete_transformer_h_33_mlp_gate_proj_weight();
  delete_transformer_h_33_mlp_up_proj_weight();
  delete_transformer_h_33_mlp_down_proj_weight();
  delete_transformer_h_33_mlp_down_proj_bias();
  delete_transformer_h_34_input_layernorm_weight();
  delete_transformer_h_34_self_attention_query_weight();
  delete_transformer_h_34_self_attention_key_value_weight();
  delete_transformer_h_34_self_attention_dense_weight();
  delete_transformer_h_34_self_attention_dense_bias();
  delete_transformer_h_34_post_attention_layernorm_weight();
  delete_transformer_h_34_mlp_gate_proj_weight();
  delete_transformer_h_34_mlp_up_proj_weight();
  delete_transformer_h_34_mlp_down_proj_weight();
  delete_transformer_h_34_mlp_down_proj_bias();
  delete_transformer_h_35_input_layernorm_weight();
  delete_transformer_h_35_self_attention_query_weight();
  delete_transformer_h_35_self_attention_key_value_weight();
  delete_transformer_h_35_self_attention_dense_weight();
  delete_transformer_h_35_self_attention_dense_bias();
  delete_transformer_h_35_post_attention_layernorm_weight();
  delete_transformer_h_35_mlp_gate_proj_weight();
  delete_transformer_h_35_mlp_up_proj_weight();
  delete_transformer_h_35_mlp_down_proj_weight();
  delete_transformer_h_35_mlp_down_proj_bias();
  delete_transformer_h_36_input_layernorm_weight();
  delete_transformer_h_36_self_attention_query_weight();
  delete_transformer_h_36_self_attention_key_value_weight();
  delete_transformer_h_36_self_attention_dense_weight();
  delete_transformer_h_36_self_attention_dense_bias();
  delete_transformer_h_36_post_attention_layernorm_weight();
  delete_transformer_h_36_mlp_gate_proj_weight();
  delete_transformer_h_36_mlp_up_proj_weight();
  delete_transformer_h_36_mlp_down_proj_weight();
  delete_transformer_h_36_mlp_down_proj_bias();
  delete_transformer_h_37_input_layernorm_weight();
  delete_transformer_h_37_self_attention_query_weight();
  delete_transformer_h_37_self_attention_key_value_weight();
  delete_transformer_h_37_self_attention_dense_weight();
  delete_transformer_h_37_self_attention_dense_bias();
  delete_transformer_h_37_post_attention_layernorm_weight();
  delete_transformer_h_37_mlp_gate_proj_weight();
  delete_transformer_h_37_mlp_up_proj_weight();
  delete_transformer_h_37_mlp_down_proj_weight();
  delete_transformer_h_37_mlp_down_proj_bias();
  delete_transformer_ln_f_weight();
  delete_lm_head_weight();
}
#endif
