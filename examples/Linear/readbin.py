from typing import Dict
import numpy as np
import logging
import os

import torch


def load_model_weights(model_paths):
    logging.info(f"Loading binary files: {model_paths}")
    try:
        if isinstance(model_paths, str):
            model_paths = [model_paths]
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

                logging.info(
                    f"Layer {key}: shape {value.shape}, "
                    f"dtype {value.dtype}, "
                    f"parameters {numpy_array.size}"
                )

        logging.info(f"Total parameters: {total_params}")

    except Exception as e:
        logging.error(f"Error loading model weights: {str(e)}")
        raise

logging.basicConfig(level=logging.DEBUG)
load_model_weights("./linear_model.bin")