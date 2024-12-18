from contextlib import ExitStack

import pandas as pd
from scipy.special import expit as sigmoid
from scipy.special import softmax
from tqdm import tqdm

import torch
from barusini.constants import TEST_MODE
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.utils import get_data
from torch.utils.data import DataLoader, SequentialSampler


class Scorer(torch.nn.Module, Serializable):

    model_class = None
    data_class = None

    def __init__(self, precision=16, model_folder=None, **model_overrides):
        super(Scorer, self).__init__()
        self.model_folder = model_folder
        self.model_overrides = model_overrides
        self.model = self.model_class.from_folder(
            folder_path=model_folder, **model_overrides
        )
        self.precision = precision

    def predict(self, test_file_path, num_workers=8, batch_size=16, precision=None):
        test = get_data(test_file_path)

        if self.model_folder is None and "original_config_path" in self.model_overrides:
            # Load scorer and data from config
            cfg_path = self.model_overrides["original_config_path"]
            test_ds = self.dataset_class.from_config(
                cfg_path, df=test, mode=TEST_MODE, **self.model_overrides
            )
        else:
            # Load scorer and data with tokenizers saved in same folder
            test_ds = self.dataset_class.from_folder(
                folder_path=self.model_folder, df=test, mode=TEST_MODE
            )

        test_dl = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            sampler=SequentialSampler(test_ds),
            num_workers=num_workers,
            pin_memory=False,
            multiprocessing_context="fork",
        )
        return self._predict(test_dl, precision=precision)

    def predict_proba(
        self, test_file_path, num_workers=8, batch_size=16, precision=None
    ):
        logits = self.predict(
            test_file_path, num_workers, batch_size, precision=precision
        )
        if len(logits.shape) == 1:
            return sigmoid(logits)
        if logits.shape[1] == 1:
            return logits.apply(sigmoid, axis=1)
        return pd.DataFrame(softmax(logits.values, axis=1))

    def _predict(self, dl, precision=None):
        if precision is None:
            precision = self.precision

        cuda = torch.cuda.is_available()
        if cuda:
            self.model.cuda()

        self.model.eval()
        all_preds = []
        if precision == 16 and cuda:
            context_managers = (torch.no_grad(), torch.cuda.amp.autocast())
        else:
            context_managers = (torch.no_grad(),)
        with ExitStack() as stack:
            for cm in context_managers:
                stack.enter_context(cm)

            for batch_idx, inpt in tqdm(enumerate(dl), total=len(dl)):
                input_dct = inpt["input"]
                if cuda:
                    if type(input_dct) is dict:
                        for key in input_dct:
                            input_dct[key] = input_dct[key].cuda()
                    else:
                        input_dct = input_dct.cuda()
                preds = self.model(input_dct, TEST_MODE)["logits"]
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

    @classmethod
    def from_folder(cls, folder_path=None, **overrides):  # load fitted
        config_path = Serializable.get_config_path(folder_path)
        return cls.from_config(config_path, model_folder=folder_path, **overrides)
