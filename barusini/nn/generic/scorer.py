from contextlib import ExitStack

import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

import torch
from barusini.nn.generic.utils import get_data
from barusini.nn.generic.loading import Serializable
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

    def predict(
        self, test_file_path, num_workers=8, batch_size=16, precision=None
    ):
        test = get_data(test_file_path)
        test_ds = self.dataset_class.from_folder(
            folder_path=self.model_folder, df=test,
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
        return logits.apply(softmax, axis=1)

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

    @classmethod
    def from_folder(cls, folder_path=None, **overrides):  # load fitted
        config_path = Serializable.get_config_path(folder_path)
        return cls.from_config(
            config_path, model_folder=folder_path, **overrides
        )
