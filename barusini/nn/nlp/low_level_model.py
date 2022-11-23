import os

import torch
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.utils import get_real_n_classes
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

SIMPLE_HEAD = "simple"
LINEAR_HEAD = "linear"
HEADS = [SIMPLE_HEAD, LINEAR_HEAD]

CLS_POOLING = "cls"
CLS_LAST_POOLING = "cls_last_"
MAX_POOLING = "max"
AVG_POOLING = "avg"
ATTENTION_POOLING = "attention"
ATTENTION_LAST_POOLING = "attention_last_"
POOLINGS = [
    CLS_POOLING,
    MAX_POOLING,
    AVG_POOLING,
    ATTENTION_POOLING,
    CLS_LAST_POOLING,
]


def cls_pooling(x, mask):
    return x[:, 0, :]


def get_cls_last_pooling(top_n):
    assert top_n > 1, f"Incorrect number of last layers to use: {top_n}"

    def cls_last_pooling(x, mask):
        return torch.concat([x[-i][:, 0] for i in range(1, top_n + 1)], dim=1)

    return cls_last_pooling


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class LastNAttentionPooling(AttentionPooling):
    def __init__(self, in_dim, last_n):
        super().__init__(in_dim * last_n)
        self.last_n = last_n

    def forward(self, last_hidden_state, attention_mask):
        x = torch.concat(
            [last_hidden_state[-i] for i in range(1, self.last_n + 1)], dim=2
        )
        return super().forward(x, attention_mask)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


def get_last_n(pooling):
    return int(pooling.split("_last_")[1])


def get_pooling(pooling, hidden_size):
    if pooling == CLS_POOLING:
        return cls_pooling
    elif pooling == AVG_POOLING:
        return MeanPooling()
    elif pooling == MAX_POOLING:
        return MaxPooling()
    elif pooling == ATTENTION_POOLING:
        return AttentionPooling(hidden_size)
    elif ATTENTION_LAST_POOLING in pooling:
        return LastNAttentionPooling(hidden_size, get_last_n(pooling))
    elif CLS_LAST_POOLING in pooling:
        return get_cls_last_pooling(get_last_n(pooling))
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

        if classification:
            n_classes = get_real_n_classes(n_classes)

        self.n_classes = n_classes

        if "_last_" in pooling:
            self.backbone_output = "hidden_states"
            self.output_coef = get_last_n(pooling)
        else:
            self.backbone_output = "last_hidden_state"
            self.output_coef = 1

        if model_folder is not None:
            self.model_config = None
            self.get_architecture(head, model_folder=model_folder)
            print("Model loaded from", model_folder)
        else:
            self.model_config = AutoConfig.from_pretrained(backbone)
            self.model_config.num_labels = self.n_classes
            if self.dropout is not None:
                self.model_config.hidden_dropout_prob = self.dropout
                self.model_config.attention_probs_dropout_prob = self.dropout

            self.get_architecture(head)

        if "_last_" in pooling:
            self.model_config.output_hidden_states = True

        self.pooling = get_pooling(pooling, self.model_config.hidden_size)

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
            self.head = nn.Linear(
                self.model_config.hidden_size * self.output_coef, self.n_classes
            )
        else:
            raise ValueError()

    def load_weights(self, pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        # remove "model." from the keys
        state_dict = {k[6:]: v for k, v in state_dict["state_dict"].items()}
        self.load_state_dict(state_dict, strict=False)
        print("weights loaded from", pretrained_weights)

    def forward(self, input_dict, mode):
        x = self.backbone(**input_dict)

        if self.head is not None:
            mask = input_dict["attention_mask"]
            x = self.pooling(x[self.backbone_output], mask)
            x = self.head(x)
            return {"logits": x}

        return x

    @staticmethod
    def remove_duplicate_weights(folder_path):
        hf_bin = os.path.join(folder_path, "pytorch_model.bin")
        hf_bin = torch.load(hf_bin, map_location="cpu")
        hf_keys = list(hf_bin.keys())
        del hf_bin

        for checkpoint in ["last.ckpt"]:
            pretrained_weights = os.path.join(folder_path, checkpoint)

            if not os.path.exists(pretrained_weights):
                continue

            state_dict = torch.load(pretrained_weights, map_location="cpu")

            del_keys = [
                k
                for k in state_dict["state_dict"].keys()
                if k.replace("model.backbone.", "") in hf_keys
            ]

            for k in del_keys:
                del state_dict["state_dict"][k]

            print("Removing duplicate weights in", pretrained_weights)
            torch.save(state_dict, pretrained_weights)

    def to_folder(self, folder_path=None):
        self.backbone.save_pretrained(folder_path)
        self.remove_duplicate_weights(folder_path)

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
        elif pretrained_weights is None and os.path.join(folder_path, "last.ckpt"):
            pretrained_weights = os.path.join(folder_path, "last.ckpt")

        return cls.from_config(
            config_path,
            model_folder=folder_path,
            pretrained_weights=pretrained_weights,
            **overrides,
        )
