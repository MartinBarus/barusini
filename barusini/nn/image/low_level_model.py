import os

import timm
import torch
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.nn_utils import ArcMarginProduct, GeM
from barusini.nn.generic.utils import get_real_n_classes
from torch import nn


class ImageNet(nn.Module, Serializable):
    def __init__(
        self,
        backbone="resnet34",
        n_classes=2,
        in_channels=3,
        pretrained_weights=None,
        **kwargs
    ):
        super().__init__()
        self.net = timm.create_model(
            model_name=backbone,
            pretrained=pretrained_weights,
            num_classes=get_real_n_classes(n_classes),
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

        obj = cls.from_config(config_path, **overrides)

        obj.load_weights(pretrained_weights)
        return obj


class ImageSimilarityNet(ImageNet):
    def __init__(
        self,
        backbone="resnet34",
        n_classes=2,
        in_channels=3,
        pretrained_weights=None,
        embedding_size=512,
        p_trainable=True,
        drop_out=0,
        batch_norm=True,
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            n_classes=n_classes,
            in_channels=in_channels,
            pretrained_weights=pretrained_weights,
        )
        self.pool = GeM(p_trainable=p_trainable)
        out_features = self.get_out_features(backbone)
        post_pool = [nn.Linear(out_features, embedding_size, bias=True)]
        if drop_out > 0:
            post_pool.insert(0, nn.Dropout(drop_out))
        if batch_norm:
            post_pool.extend([nn.BatchNorm1d(embedding_size), torch.nn.PReLU()])
        self.pots_pool = nn.Sequential(*post_pool)
        self.head = ArcMarginProduct(embedding_size, n_classes)

    def forward(self, input_dict, mode):
        emb = self.net.forward_features(input_dict)
        emb = self.pool(emb)
        emb = self.pots_pool(emb)
        logits = self.head(emb)
        return {"logits": logits, "embeddings": emb}

    def get_out_features(self, name):
        if "regnet" in name or "csp" in name:
            return self.net.head.fc.in_features
        if "res" in name or "senet" in name:
            return self.net.fc.in_features
        if "inception" in name:
            return self.net.last_linear.in_features
        return self.net.classifier.in_features
