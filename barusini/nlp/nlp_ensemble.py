import pandas as pd
from joblib import Parallel, delayed

from barusini.nlp.nlp_model import NlpModel
from barusini.utils import copy_signature


class NlpEnsemble:
    @copy_signature(NlpModel.__init__)
    def __init__(self, *args, **kwargs):
        self.model_args = args
        self.model_kwargs = kwargs
        self.models = None

    @staticmethod
    def from_config(config_path):
        config = NlpModel.parse_config(config_path)
        return NlpEnsemble(**config)

    @copy_signature(NlpModel.fit)
    def fit(self, train, val, *args, **kwargs):
        assert type(train) in [list, tuple], "train must be list or tuple"
        assert type(val) in [list, tuple], "val must be list or tuple"
        assert len(train) == len(val), "train and val must be of same size"

        gpus = kwargs.get("gpus", ("0",))
        if len(gpus) > 1:
            self._fit_parallel(train, val, *args, **kwargs)
        else:
            self._fit_sequential(train, val, *args, **kwargs)

    def _fit_parallel(self, train, val, *args, gpus=("0",), **kwargs):
        n_gpus = len(gpus)
        parallel = Parallel(
            n_jobs=n_gpus, verbose=False, pre_dispatch="2*n_jobs"
        )
        self.models = [
            NlpModel(*self.model_args, **self.model_kwargs)
            for i in range(len(train))
        ]
        parallel(
            delayed(self.models[i].fit)(
                train[i], val[i], *args, **kwargs, gpus=gpus[i % n_gpus]
            )
            for i in range(len(train))
        )

    def _fit_sequential(self, train, val, *args, gpus=("0",), **kwargs):
        n_gpus = len(gpus)
        self.models = []
        for i in range(len(train)):
            m = NlpModel(*self.model_args, **self.model_kwargs)
            m.fit(train[i], val[i], *args, gpus=gpus[i % n_gpus], **kwargs)
            self.models.append(m)

    def get_oof_dict(self):
        return [m.get_oof_dict() for m in self.models]

    def oof_dataset(self):
        oof_datasets = [m.oof_dataset() for m in self.models]
        for i, oof_df in enumerate(oof_datasets):
            oof_df["fold"] = i
        return pd.concat(oof_datasets).reset_index(drop=True)
