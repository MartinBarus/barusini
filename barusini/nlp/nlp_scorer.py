import json
import os
from contextlib import ExitStack

import pandas as pd
from tqdm import tqdm

import torch
from barusini.nlp.data import NLPDataset
from barusini.nlp.low_level_model import NlpNet
from torch.utils.data import DataLoader, SequentialSampler


class NlpScorer(torch.nn.Module):
    def __init__(
        self,
        model_class=NlpNet,
        input_cols=None,
        n_tokens=None,
        pretrained_weights=None,
        precision=16,
        model_folder=None,
        **kwargs,
    ):
        super(NlpScorer, self).__init__()
        valid = (model_folder is not None) ^ (pretrained_weights is not None)
        assert valid, "Provide either model folder or pretrained weights!"
        self.model = model_class(
            **kwargs, pretrained_weights=None, model_folder=model_folder
        )
        self.input_cols = input_cols
        self.n_tokens = n_tokens
        self.precision = precision
        self.model_folder = model_folder
        if pretrained_weights is not None:
            state = torch.load(pretrained_weights, map_location="cpu")
            self.load_state_dict(state["state_dict"])
            print("Loaded weights", pretrained_weights)
        if model_folder is not None:
            print("Using model folder", model_folder)

    @staticmethod
    def parse_config(config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
            config["model_id"] = os.path.basename(config_path).split(".")[0]
            return config

    @staticmethod
    def from_config(config_path):
        config = NlpScorer.parse_config(config_path)
        return NlpScorer(**config)

    @staticmethod
    def from_folder(folder_path):
        config_path = os.path.join(folder_path, "high_level_config.json")
        config = NlpScorer.parse_config(config_path)
        config["model_folder"] = folder_path
        return NlpScorer(**config)

    def save_to_folder(self, folder_path):
        self.model.save_to_folder(folder_path)

    def predict(self, test_file_path, num_workers=8, batch_size=16):
        test = pd.read_csv(test_file_path)

        if self.model_folder is not None:
            load_from = self.model_folder
        else:
            load_from = self.model.backbone_name

        test_ds = NLPDataset(
            test, load_from, self.input_cols, None, self.n_tokens
        )

        test_dl = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            sampler=SequentialSampler(test_ds),
            num_workers=num_workers,
            pin_memory=False,
            multiprocessing_context="fork",
        )
        return self._predict(test_dl)

    def _predict(self, dl):
        cuda = torch.cuda.is_available()
        if cuda:
            self.model.cuda()

        self.model.eval()
        all_preds = []
        if self.precision == 16 and cuda:
            context_managers = (torch.no_grad(), torch.cuda.amp.autocast())
        else:
            context_managers = (torch.no_grad(),)
        with ExitStack() as stack:
            for cm in context_managers:
                stack.enter_context(cm)

            for batch_idx, inpt in tqdm(enumerate(dl), total=len(dl)):
                input_dct = inpt["input"]
                if cuda:
                    for key in input_dct:
                        input_dct[key] = input_dct[key].cuda()
                preds = self.model(input_dct)["logits"]
                if cuda:
                    all_preds.extend(preds.cpu().tolist())
                else:
                    all_preds.extend(preds.tolist())

                del input_dct
                del inpt
                del preds
        if cuda:
            self.model.cpu()

        res = pd.DataFrame(all_preds)
        return res
