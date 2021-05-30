import os
from datetime import datetime


def extract_model_path_for_hyperparams(start_path, model):
    relevant_hparams = {}
    for key in [
        "arch",
        "dataset_name",
        "test_only"
    ]:
        if hasattr(model.hparams, key):
            relevant_hparams[key] = eval(f"model.hparams.{key}")

    path = os.path.join(start_path, *["{}_{}".format(key, val) for key, val in relevant_hparams.items()])
    
    dt_string = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    path = os.path.join(path, dt_string)

    os.makedirs(path, exist_ok=True)

    return path