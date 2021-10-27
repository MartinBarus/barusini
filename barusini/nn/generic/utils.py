import json
import os
import random

import numpy as np
import pandas as pd

import torch


def set_seed(seed=1234):
    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def expand_classification_label(label, max_size=None):
    label = label.astype(np.int64)
    if max_size is None:
        max_size = label.max()

    new_label = np.zeros((label.size, max_size + 1))
    new_label[np.arange(label.size), label] = 1
    return new_label


def get_data(x):
    if type(x) is str:
        return pd.read_csv(x)
    return x


def get_real_n_classes(x):
    if x == 2:
        return 1  # Binary classification will only return probability of class 1
    return x


def parse_config(config_path, **overrides):
    with open(config_path, "r") as file:
        config = json.load(file)
        config["model_id"] = os.path.basename(config_path).split(".")[0]
        return {**config, **overrides}
