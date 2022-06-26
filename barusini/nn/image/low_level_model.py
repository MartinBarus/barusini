import os

import timm
import torch
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.utils import get_real_n_classes
from torch import nn


class ImageNet(nn.Module, Serializable):
    def __init__(
        self,
        backbone="resnet34",
        n_classes=2,
        in_channels=3,
        pretrained_weights=True,
        classification=None,
        **kwargs
    ):
        super().__init__()

        if classification:
            n_classes = get_real_n_classes(n_classes)

        self.net = timm.create_model(
            model_name=backbone,
            pretrained=pretrained_weights,
            num_classes=n_classes,
            drop_rate=0,
            in_chans=in_channels,
        )

    def forward(self, input_dict, mode):
        return {"logits": self.net(input_dict)}

    def load_weights(self, pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        state_dict = {k[6:]: v for k, v in state_dict["state_dict"].items()}
        self.load_state_dict(state_dict, strict=True)
        print("weights loaded from", pretrained_weights)

    @classmethod
    def from_folder(
        cls, folder_path=None, best=False, pretrained_weights=None, **overrides
    ):
        config_path = Serializable.get_config_path(folder_path)
        if best:
            pretrained_weights = Serializable.find_best_ckpt(folder_path)
        else:
            pretrained_weights = os.path.join(folder_path, "last.ckpt")

        obj = cls.from_config(config_path, pretrained_weights=False, **overrides)

        obj.load_weights(pretrained_weights)
        return obj
