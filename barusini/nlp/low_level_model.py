import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class NlpNet(nn.Module):
    def __init__(
        self, backbone, n_classes, pretrained_weights,
    ):

        super(NlpNet, self).__init__()
        self.backbone_name = backbone
        self.n_classes = n_classes
        self.pretrained_weights = pretrained_weights
        self.model_config = AutoConfig.from_pretrained(backbone)
        self.model_config.num_labels = n_classes
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            self.backbone_name, config=self.model_config
        )

        if self.pretrained_weights is not None:
            self.load_state_dict(
                torch.load(self.pretrained_weights, map_location="cpu"),
                strict=False,
            )
            print("weights loaded from", self.pretrained_weights)

    def forward(self, input_dict):
        return self.backbone(**input_dict)
