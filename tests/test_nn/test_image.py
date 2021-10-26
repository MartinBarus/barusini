import json
import os

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from barusini.nn.image.image_model import ImageModel, ImageScorer
from datasets import load_dataset

from torchvision import datasets
import os
import numpy as np
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import shutil


def read_fl(fl):
    with open(fl, 'rb') as file:
        return file.read()


def get_mnist_data(root, N=1200):
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    val_path = os.path.join(root, f"val_{N}.csv")
    train_path = os.path.join(root, f"train_{N}.csv")
    image_dir = os.path.join(root, f"images")

    if os.path.exists(val_path) and os.path.exists(train_path) and os.path.exists(
            image_dir) and len(os.listdir(image_dir)) >= N:
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
    return get_mnist_data("image_data")


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


def test_image(mnist, image_config):
    train_path, val_path = mnist

    # Create NLP Model object (used for training) and fit it
    model = ImageModel.from_config(image_config)
    model.fit(train_path, val_path, gpus=None)  # use gpus=[0] to use GPU 0

    # Create NLP Scorer (used for predicting) from NLP Model checkpoint folder
    model_folder = model.ckpt_save_path.format(val=os.path.basename(val_path))
    scorer = ImageScorer.from_folder(model_folder)

    # Load test data, make predictions, compute AUC
    test = pd.read_csv(val_path)
    preds = scorer.predict_proba(test)
    label = test["label"]
    auc = roc_auc_score(label, preds.values, multi_class='ovo')
    assert auc > 0.81
