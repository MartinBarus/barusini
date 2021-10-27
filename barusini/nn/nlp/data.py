import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from barusini.nn.generic.loading import Serializable


class NLPDataset(Dataset, Serializable):
    def __init__(self, df, backbone, input_cols, label, n_tokens, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        text_values = df[input_cols].fillna("").sum(axis=1).values
        self.tokenized_texts = self.tokenizer(
            text_values.tolist(),
            add_special_tokens=True,
            padding=True,
            max_length=n_tokens,
            truncation=True,
            return_tensors="pt",
        )
        self.text = text_values
        self.eps = 1e-6
        self.labels = label
        if self.labels is not None:
            self.labels = df[self.labels].values

    def __getitem__(self, idx):
        inpt = {
            key: self.tokenized_texts[key][idx] for key in self.tokenized_texts
        }

        feature_dict = {
            "idx": torch.tensor(idx).long(),
            "input": inpt,
        }

        if self.labels is not None:
            target = torch.tensor(self.labels[idx])
            feature_dict["target"] = target.float()

        return feature_dict

    def __len__(self):
        return len(self.text)

    def to_folder(self, folder_path=None):
        self.tokenizer.save_pretrained(folder_path)

    @classmethod
    def from_folder(cls, folder_path=None, **overrides):
        config_path = Serializable.get_config_path(folder_path)
        return cls.from_config(config_path, backbone=folder_path, **overrides)
