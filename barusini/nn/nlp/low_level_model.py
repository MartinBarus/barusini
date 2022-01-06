import torch
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.utils import get_real_n_classes
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class NlpNet(nn.Module, Serializable):
    def __init__(
        self,
        backbone=None,
        n_classes=None,
        pretrained_weights=None,  # only the weights
        model_folder=None,  # model config, vocab, weights
        **kwargs
    ):

        super(NlpNet, self).__init__()
        self.backbone_name = backbone
        self.pretrained_weights = pretrained_weights
        self.model_folder = model_folder
        if model_folder is not None:
            self.model_config = None
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_folder
            )
            print("Model loaded from", model_folder)
        else:
            self.n_classes = get_real_n_classes(n_classes)
            self.model_config = AutoConfig.from_pretrained(backbone)
            self.model_config.num_labels = self.n_classes
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                self.backbone_name, config=self.model_config
            )
        if self.pretrained_weights is not None:
            self.load_weights(self.pretrained_weights)

    def load_weights(self, pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        # remove "model." from the keys
        state_dict = {k[6:]: v for k, v in state_dict["state_dict"].items()}
        self.load_state_dict(state_dict, strict=True)
        print("weights loaded from", pretrained_weights)

    def forward(self, input_dict, mode):
        return self.backbone(**input_dict)

    def to_folder(self, folder_path=None):
        self.backbone.save_pretrained(folder_path)

    @classmethod
    def from_folder(
        cls,
        folder_path=None,
        best=False,
        pretrained_weights=None,
        original_config_path=None,
        **overrides
    ):
        if folder_path is None:
            return cls.from_config(
                original_config_path, pretrained_weights=pretrained_weights, **overrides
            )

        config_path = Serializable.get_config_path(folder_path)
        if best:
            pretrained_weights = Serializable.find_best_ckpt(folder_path)

        return cls.from_config(
            config_path,
            model_folder=folder_path,
            pretrained_weights=pretrained_weights,
            **overrides,
        )
