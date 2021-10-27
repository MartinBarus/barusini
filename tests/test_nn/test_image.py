import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from tests.test_nn.utils import run_nn_test
from tqdm import tqdm

from barusini.constants import rmse
from barusini.nn.image.image_model import ImageModel, ImageScorer
from torchvision import datasets


def read_fl(fl):
    with open(fl, "rb") as file:
        return file.read()


def get_mnist_data(root, N=1200):
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    val_path = os.path.join(root, f"val_{N}.csv")
    train_path = os.path.join(root, f"train_{N}.csv")
    image_dir = os.path.join(root, f"images")

    if (
        os.path.exists(val_path)
        and os.path.exists(train_path)
        and os.path.exists(image_dir)
        and len(os.listdir(image_dir)) >= N
    ):
        print("Mnist Dataset Already exists")
        return train_path, val_path

    os.makedirs(image_dir, exist_ok=True)

    # Download MNIST data
    mnist = datasets.MNIST(root=root, train=True, download=True, transform=None)
    mnist_img_path = os.path.join(mnist.raw_folder, "train-images-idx3-ubyte")
    mnist_labels_path = os.path.join(mnist.raw_folder, "train-labels-idx1-ubyte")

    # Load Images
    mnist = read_fl(mnist_img_path)
    images = [int(x) for x in mnist[16:]]
    images = np.array(images, dtype=np.uint8)
    images = images.reshape((-1, 28, 28))

    # Load labels
    labels = read_fl(mnist_labels_path)
    labels = [int(x) for x in labels[8:]]

    # Create train and validation dataframes
    paths = [f"{i}_{labels[i]}.jpg" for i in range(len(labels))]
    df = pd.DataFrame({"path": paths[:N], "label": labels[:N]})
    df["label2"] = (df["label"] > 4).astype(int)
    val_start = int(N * 0.8)

    train_df = df.iloc[:val_start]
    val_df = df.iloc[val_start:]

    # Save images as jpgs
    for i in tqdm(df.index):
        plt.imsave(os.path.join(image_dir, paths[i]), images[i])

    # Save train and validation csv
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Remove temporary downloaded files
    shutil.rmtree(os.path.join(root, "MNIST"))

    return train_path, val_path


@pytest.fixture(scope="session")
def mnist():
    train_path, val_path = get_mnist_data("image_data")
    return train_path, val_path, val_path  # test is same as validation


@pytest.fixture(scope="session")
def image_config():
    config_path = "cfg_image.json"
    config = {
        "n_classes": 10,
        "backbone": "resnet34",
        "batch_size": 16,
        "metric": "roc_auc_score",
        "path_col": "path",
        "label": "label",
        "precision": 32,  # without GPU only 32 precision is supported
        "max_epochs": 1,
        "data_folder": "image_data/images",
        "lr": 0.001,
        "seed": 7,
        "pretrained_weights": True,
        "num_workers": 0,
    }

    with open(config_path, "w") as file:
        json.dump(config, file)

    return config_path


def run_image_test(mnist, config, label_col, proba, **config_overrides):
    return run_nn_test(
        mnist, ImageModel, ImageScorer, config, label_col, proba, **config_overrides
    )


def test_image_multiclass(mnist, image_config):
    preds, label = run_image_test(mnist, image_config, label_col="label", proba=True)
    auc = roc_auc_score(label, preds.values, multi_class="ovo")
    assert auc > 0.81


def test_image_binary(mnist, image_config):
    preds, label = run_image_test(
        mnist,
        image_config,
        label_col="label2",
        proba=True,
        n_classes=2,
        label_smoothing=0.01,
    )
    auc = roc_auc_score(label, preds.values)
    assert auc > 0.81


def test_image_regression(mnist, image_config):
    preds, label = run_image_test(
        mnist, image_config, label_col="label", proba=False, n_classes=1, metric="rmse"
    )
    score = rmse(preds, label)
    assert score < 2
