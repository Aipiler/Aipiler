import torch
import logging
import sys
from typing import Dict, List, Union
import numpy as np
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def load_model_weights(model_paths) -> Dict[str, np.ndarray]:
    logging.info(f"Loading binary files: {model_paths}")
    try:
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        weights_dict = {}
        total_params = 0
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

                numpy_array = value.cpu().numpy()
                total_params += numpy_array.size
                weights_dict[key] = numpy_array

                logging.info(
                    f"Layer {key}: shape {value.shape}, "
                    f"dtype {value.dtype}, "
                    f"parameters {numpy_array.size}"
                )

        logging.info(f"Total parameters: {total_params}")
        return weights_dict

    except Exception as e:
        logging.error(f"Error loading model weights: {str(e)}")
        raise

