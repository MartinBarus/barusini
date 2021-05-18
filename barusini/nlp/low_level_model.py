import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class NlpNet(nn.Module):
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
        if model_folder is not None:
            self.model_config = None
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_folder
            )
        else:
            self.n_classes = n_classes
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

    def save_to_folder(self, folder_path):
        self.backbone.save_pretrained(folder_path)
        tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        tokenizer.save_pretrained(folder_path)
