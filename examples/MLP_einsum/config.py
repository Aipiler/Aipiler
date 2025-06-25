from dataclasses import dataclass

batch_size = 1
seq_len = 16


@dataclass
class Config:
    hidden_size = 7168
    intermediate_size = 18432
    hidden_act = "silu"
