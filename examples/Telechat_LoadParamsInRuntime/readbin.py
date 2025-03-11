from typing import Dict
import numpy as np
import logging
import os
from typing import List
import torch


def load_model_weights(model_paths):
    logging.info(f"Loading binary files: {model_paths}")
    try:
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        total_params = 0
        with open("globals.h", "w") as f:
            print("#ifndef GLOBALS_H", file=f)
            print("#define GLOBALS_H", file=f)
            print("#include \"memref.h\"", file=f)
            global_names: List[str] = []
            # key : global_name
            key_names : List[str] = []
            type_name = "half"
            for model_path in model_paths:
                if not os.path.exists(model_path):
                    logging.error(f"Can not find file {model_path}")
                    raise RuntimeError(f"Can not find file {model_path}")
                logging.info(f"Loading weight from path: {model_path}")
                model_weights = torch.load(model_path, weights_only=True)
                logging.info("Successfully loaded model weights")

                for key, value in model_weights.items():
                    if not torch.is_tensor(value):
                        logging.warning(f"Skipping non-tensor value for key: {key}")
                        continue
                    assert isinstance(key, str)
                    assert isinstance(value, torch.Tensor)
                    numpy_array = value.cpu().numpy()
                    total_params += numpy_array.size

                    logging.info(
                        f"Layer {key}: shape {value.shape}, "
                        f"dtype {value.dtype}, "
                        f"parameters {numpy_array.size}"
                    )
                    
                    global_name = key.replace(".", "_")
                    global_names.append(global_name)
                    key_names.append(key)
                    list_shape = [str(e) for e in value.shape]
                    shape_size = len(list_shape)
                    print("""
// Layer {key}: shape {shape}, dtype {value_type}
extern "C" {{ RankedMemRefType<{type_name}, {shape_size}> *{global_name}; }}
void init_{global_name}(){{  
    {type_name} *{global_name}_data = new {type_name}[{shapes_multi}];
    int64_t {global_name}_shape[{shape_size}] = {{{shapes_comma}}};
    {global_name} =
        new RankedMemRefType<{type_name}, {shape_size}>({global_name}_data, {global_name}_shape);
}}

void delete_{global_name}(){{  
    delete {global_name};
}}
""".format(key = key, shape = value.shape, value_type = value.dtype, type_name = type_name, shape_size = shape_size, global_name = global_name, shapes_multi = " * ".join(list_shape), shapes_comma = ", ".join(list_shape)), file=f)

            logging.info(f"Total parameters: {total_params}")

            # init_all
            print("void init_all_globals() {", file=f)
            for g_name in global_names:
                print(f"\tinit_{g_name}();", file=f)
            
            print("std::vector<std::string> model_names = {{{model_names}}};".format(model_names = ", ".join([f"\"{model_name}\"" for model_name in model_paths])), file = f)
            print(f"std::map<std::string, {type_name}*> param_and_loc =", file = f)
            print("{", file=f)
            map_elements = []
            for global_name, key_name in zip(global_names, key_names):
                map_elements.append("{{\"{key}\", {name}->data}}".format(key=key_name, name=global_name))
            print(", \n".join(map_elements), file=f)
            print("};", file = f)
            print("mix::utils::load_model_f16(model_names, param_and_loc);", file=f)
            # end of function init_all            
            print("}", file=f)

            # del_all
            print("void delete_all_globals() {", file=f)
            for g_name in global_names:
                print(f"\tdelete_{g_name}();", file=f)
            print("}", file=f)
            
            print("#endif", file=f)
    except Exception as e:
        logging.error(f"Error loading model weights: {str(e)}")
        raise

logging.basicConfig(level=logging.DEBUG)
load_model_weights([f"./pytorch_model_0000{n}-of-00004.bin" for n in [1,2,3,4]])