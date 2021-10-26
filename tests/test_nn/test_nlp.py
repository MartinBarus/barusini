import json
import os

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from barusini.nn.nlp.nlp_model import NlpModel, NlpScorer
from datasets import load_dataset


@pytest.fixture(scope="session")
def nlp_data():
    train_path = "nlp_train.csv"
    val_path = "nlp_val.csv"
    test_path = "nlp_test.csv"
    file_paths = [train_path, val_path, test_path]
    if all([os.path.exists(path) for path in file_paths]):
        return train_path, val_path, test_path

    dataset = load_dataset("ethos", "binary")
    train = dataset["train"].to_pandas().sample(frac=0.5, random_state=42)

    train_bound = int(0.6 * train.shape[0])
    val_bound = int(0.8 * train.shape[0])
    test = train.iloc[val_bound:].reset_index(drop=True)
    val = train.iloc[train_bound:val_bound].reset_index(drop=True)
    train = train.iloc[:train_bound].reset_index(drop=True)

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, val_path, test_path


@pytest.fixture(scope="session")
def nlp_config():
    config_path = "cfg_nlp.json"
    config = {
        "n_classes": 2,
        "n_tokens": 256,
        "backbone": "distilbert-base-uncased",
        "batch_size": 16,
        "metric": "roc_auc_score",
        "input_cols": ["text"],
        "label": "label",
        "precision": 32,  # without GPU only 32 precision is supported
        "max_epochs": 1,
    }

    with open(config_path, "w") as file:
        json.dump(config, file)

    return config_path


def test_nlp(nlp_data, nlp_config):
    train_path, val_path, test_path = nlp_data

    # Create NLP Model object (used for training) and fit it
    model = NlpModel.from_config(nlp_config)
    model.fit(train_path, val_path, gpus=None)  # use gpus=[0] to use GPU 0

    # Create NLP Scorer (used for predicting) from NLP Model checkpoint folder
    model_folder = model.ckpt_save_path.format(val=val_path)
    scorer = NlpScorer.from_folder(model_folder)

    # Load test data, make predictions, compute AUC
    test = pd.read_csv(test_path)
    preds = scorer.predict_proba(test)
    label = test["label"]
    auc = roc_auc_score(label, preds.iloc[:, 1])
    assert auc > 0.81
