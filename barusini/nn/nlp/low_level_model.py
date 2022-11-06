import torch
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.utils import get_real_n_classes
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

SIMPLE_HEAD = "simple"
LINEAR_HEAD = "linear"
HEADS = [SIMPLE_HEAD, LINEAR_HEAD]

CLS_POOLING = "cls"
MAX_POOLING = "max"
AVG_POOLING = "avg"
POOLINGS = [CLS_POOLING, MAX_POOLING, AVG_POOLING]


def cls_pooling(x, mask):
    return x[:, 0, :]


def avg_pooling(x, mask):
    denom = torch.sum(mask, -1, keepdim=True)
    return torch.sum(x * mask.unsqueeze(-1), dim=1) / denom


def get_pooling(pooling):
    if pooling == CLS_POOLING:
        return cls_pooling
    elif pooling == AVG_POOLING:
        return avg_pooling
    else:
        raise ValueError(f"Unsupported pooling {pooling}")


class NlpNet(nn.Module, Serializable):
    def __init__(
        self,
        backbone=None,
        n_classes=None,
        pretrained_weights=None,  # only the weights
        model_folder=None,  # model config, vocab, weights
        dropout=0,
        classification=None,
        head=SIMPLE_HEAD,
        pooling=CLS_POOLING,
        **kwargs,
    ):

        super(NlpNet, self).__init__()
        self.backbone_name = backbone
        self.head_type = head
        self.pool_type = pooling
        self.pretrained_weights = pretrained_weights
        self.model_folder = model_folder
        self.dropout = dropout
        self.backbone = None
        self.head = None
        self.pooling = get_pooling(pooling)
        if model_folder is not None:
            self.model_config = None
            self.get_architecture(head, model_folder=model_folder)
            print("Model loaded from", model_folder)
        else:

            if classification:
                n_classes = get_real_n_classes(n_classes)

            self.n_classes = n_classes
            self.model_config = AutoConfig.from_pretrained(backbone)
            self.model_config.num_labels = self.n_classes
            if self.dropout is not None:
                self.model_config.hidden_dropout_prob = self.dropout
                self.model_config.attention_probs_dropout_prob = self.dropout

            self.get_architecture(head)

        if self.pretrained_weights is not None:
            self.load_weights(self.pretrained_weights)

    def get_architecture(self, head, model_folder=None):
        if head == SIMPLE_HEAD:
            if model_folder:
                self.backbone = AutoModelForSequenceClassification.from_pretrained(
                    model_folder
                )
            else:
                self.backbone = AutoModelForSequenceClassification.from_pretrained(
                    self.backbone_name,
                    config=self.model_config,
                    ignore_mismatched_sizes=True,
                )
        elif head == LINEAR_HEAD:
            if model_folder:
                self.backbone = AutoModel.from_pretrained(model_folder)
                self.model_config = AutoConfig.from_pretrained(model_folder)
            else:
                self.backbone = AutoModel.from_pretrained(
                    self.backbone_name,
                    config=self.model_config,
                    ignore_mismatched_sizes=True,
                )
            self.head = nn.Linear(self.model_config.hidden_size, self.n_classes)
        else:
            raise ValueError()

    def load_weights(self, pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        # remove "model." from the keys
        state_dict = {k[6:]: v for k, v in state_dict["state_dict"].items()}
        self.load_state_dict(state_dict, strict=True)
        print("weights loaded from", pretrained_weights)

    def forward(self, input_dict, mode):
        x = self.backbone(**input_dict)

        if self.head is not None:
            mask = input_dict["attention_mask"]
            x = self.pooling(x["last_hidden_state"], mask)
            x = self.head(x)
            return {"logits": x}

        return x

    def to_folder(self, folder_path=None):
        self.backbone.save_pretrained(folder_path)

    @classmethod
    def from_folder(
        cls,
        folder_path=None,
        best=False,
        pretrained_weights=None,
        original_config_path=None,
        **overrides,
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
