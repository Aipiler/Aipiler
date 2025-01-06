
import os
import torch

from torch.export import export

from typing import List
import torch
from torch import _dynamo as torchdynamo
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable


import sys
sys.path.append("../models_src_code")  # 添加子目录路径

from telechat_src.modeling_telechat import TelechatPreTrainedModel, GenerationConfig,TelechatForCausalLM,TelechatModel
from telechat_src.tokenization_telechat import TelechatTokenizer
from telechat_src.configuration_telechat import TelechatConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_path = "权重所在的路径"

def telechat_infer():
    tokenizer = TelechatTokenizer.from_pretrained(model_path)
    model = TelechatModel.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float16)
    # config = TelechatConfig()
    # model = TelechatModel(config)
    
    print(model)
    print(type(model))

    generate_config = GenerationConfig.from_pretrained(model_path)
    question="生抽与老抽的区别？"

    inputs = build_inputs_for_chat(tokenizer, question)
    print("inputs:",inputs)
    answer = model(inputs)


    print("responese:\n")
    print(answer[0][len(inputs[0]):-1])



def build_inputs_for_chat(tokenizer, question):
        """
        check history and  build inputs here
        """
        # first tokenize question
        q_token = tokenizer(question)

        # get the max length we should build our inputs in
        # model_max_length = self.config.seq_length
        build_max_length = 100

        # trunc left
        input_tokens =  q_token["input_ids"][-build_max_length + 1:] 


        return torch.tensor([input_tokens], dtype=torch.int64)



if __name__ == '__main__':
    telechat_infer()
