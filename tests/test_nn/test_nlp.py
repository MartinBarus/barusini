import json
import os

import pytest
from sklearn.metrics import roc_auc_score
from tests.test_nn.utils import run_nn_test

from barusini.constants import rmse
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


def run_nlp_test(data, config, label_col, proba, **config_overrides):
    return run_nn_test(
        data, NlpModel, NlpScorer, config, label_col, proba, **config_overrides
    )


def test_nlp_binary(nlp_data, nlp_config):
    preds, label = run_nlp_test(nlp_data, nlp_config, label_col="label", proba=True)
    auc = roc_auc_score(label, preds.iloc[:, 0])
    assert auc > 0.7


def test_nlp_regression(nlp_data, nlp_config):
    preds, label = run_nlp_test(
        nlp_data, nlp_config, label_col="label", proba=False, n_classes=1, metric="rmse"
    )
    score = rmse(label, preds.iloc[:, 0])
    assert score < 0.5
