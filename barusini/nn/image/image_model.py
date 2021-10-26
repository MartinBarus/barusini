import os

from barusini.nn.generic.high_level_model import HighLevelModel
from barusini.nn.generic.ensemble import Ensemble
from barusini.nn.generic.scorer import Scorer
from barusini.nn.image.data import ImageDataset
from barusini.nn.image.low_level_model import ImageNet


class ImageModel(HighLevelModel):
    model_class = ImageNet
    dataset_class = ImageDataset


class ImageScorer(Scorer):
    model_class = ImageNet
    dataset_class = ImageDataset


class ImageEnsemble(Ensemble):
    high_level_model_class = ImageModel
