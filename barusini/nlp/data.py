import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NLPDataset(Dataset):
    def __init__(self, df, model_name, input_cols, label_col, max_length):

        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_values = df[input_cols].fillna("").sum(axis=1).values
        self.tokenized_texts = self.tokenizer(
            text_values.tolist(),
            add_special_tokens=True,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        self.text = text_values
        self.eps = 1e-6
        self.labels = label_col
        if self.labels is not None:
            self.labels = self.df[self.labels].values

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
        return self.df.shape[0]
