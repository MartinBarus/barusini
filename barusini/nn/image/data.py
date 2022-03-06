import os

import pandas as pd

import albumentations as A
import cv2
import torch
from barusini.constants import TEST_MODE, TRAIN_MODE
from barusini.nn.generic.loading import Serializable


class ImageDataset(torch.utils.data.Dataset, Serializable):
    def __init__(
        self,
        df,
        path_col,
        label=None,
        data_folder="",
        label_smoothing=None,
        additional_augment=[A.HorizontalFlip(p=0.5)],
        image_height=224,
        image_width=224,
        mode=TEST_MODE,
        **kwargs,
    ):
        if type(df) is str:
            df = pd.read_csv(df)

        self.image_paths = df[path_col]
        self.labels = label
        if self.labels is not None:
            self.labels = df[self.labels].values

        self.augment_test = A.Compose(
            [A.Resize(height=image_height, width=image_width, p=1)]
        )
        self.augment_train = A.Compose(
            [aug for aug in self.augment_test] + additional_augment
        )
        self.label_smoothing = label_smoothing
        self.mode = mode
        self.data_folder = data_folder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = os.path.join(self.data_folder, self.image_paths.iloc[index])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image / 255

        if self.mode == TEST_MODE:
            image = self.augment_test(image=image)["image"]
        elif self.mode == TRAIN_MODE:
            image = self.augment_train(image=image)["image"]
        else:
            raise ValueError(f"Unsupported data mode {self.mode}!")

        feature_dict = {
            "idx": torch.tensor(index).long(),
            "input": torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float),
        }

        if self.labels is not None:
            target = self.labels[index]
            if self.label_smoothing:
                feature_dict["target"] = torch.tensor(
                    abs(target - self.label_smoothing)
                )
            else:
                feature_dict["target"] = torch.tensor(target, dtype=torch.float)

        return feature_dict