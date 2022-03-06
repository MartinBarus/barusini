import os

from barusini.nn.generic.ensemble import Ensemble
from barusini.nn.generic.high_level_model import HighLevelModel
from barusini.nn.generic.scorer import Scorer
from barusini.nn.image.data import ImageDataset
from barusini.nn.image.low_level_model import ImageNet, ImageSimilarityNet


class ImageModel(HighLevelModel):
    model_class = ImageNet
    train_dataset_class = ImageDataset
    val_dataset_class = ImageDataset


class ImageScorer(Scorer):
    model_class = ImageNet
    dataset_class = ImageDataset


class ImageEnsemble(Ensemble):
    high_level_model_class = ImageModel


class ImageSimilarityModel(ImageModel):
    model_class = ImageSimilarityNet


class ImageSimilarityScorer(ImageScorer):
    model_class = ImageSimilarityNet


class ImageSimilarityEnsemble(ImageEnsemble):
    high_level_model_class = ImageSimilarityModel
