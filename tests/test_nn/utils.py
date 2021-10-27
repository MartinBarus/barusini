import os

import pandas as pd

import torch


def get_gpus():
    if torch.cuda.is_available():
        return [0]
    return None


def run_nn_test(
    data, model_class, scorer_class, config, label_col, proba, **config_overrides
):
    train_path, val_path, test_path = data

    # Create Model object (used for training) and fit it
    model = model_class.from_config(config, label=label_col, **config_overrides)
    model.fit(train_path, val_path, gpus=get_gpus())  # use gpus=[0] to use GPU 0

    # Create Scorer (used for predicting) from NLP Model checkpoint folder
    model_folder = model.ckpt_save_path.format(val=os.path.basename(val_path))
    scorer = scorer_class.from_folder(model_folder)

    # Load test data, make predictions
    test = pd.read_csv(test_path)
    if proba:
        preds = scorer.predict_proba(test)
    else:
        preds = scorer.predict(test)
    label = test[label_col]
    return preds, label
