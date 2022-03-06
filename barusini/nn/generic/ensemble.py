import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from barusini.nn.generic.high_level_model import HighLevelModel
from barusini.utils import copy_signature
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.utils import get_data


class Ensemble(Serializable):
    high_level_model_class = HighLevelModel

    @copy_signature(HighLevelModel.__init__)
    def __init__(self, *args, seed=(1234,), **kwargs):
        self.model_args = args
        self.model_kwargs = kwargs
        self.models = None
        if type(seed) not in [list, tuple]:
            self.seeds = [seed]
        else:
            self.seeds = seed

    def fit(self, train, val, gpus=("0",), fold_col=None, folds=None, **kwargs):
        if fold_col is None:
            assert type(train) in [list, tuple], "train must be list or tuple"
            assert type(val) in [list, tuple], "val must be list or tuple"
            assert len(train) == len(val), "train and val must be of same size"
            val_splits = [None] * len(train)
        else:
            assert type(train) in [str, pd.DataFrame], (
                "if fold is provided, " "train should be a string or " "dataframe"
            )
            name = train
            all_data = get_data(train)
            val_splits = sorted(all_data[fold_col].unique())
            if folds is not None:
                assert all([x in val_splits for x in folds]), "Incorrect folds"
                val_splits = folds

            print(f"Using fold column {fold_col}, training folds {val_splits}")
            train = [all_data.query(f"{fold_col}!={fld}") for fld in val_splits]
            val = [all_data.query(f"{fold_col}=={fld}") for fld in val_splits]
            name = f"{all_data.shape[0]}_rows" if type(name) is not str else name
            val_splits = [f"{name}_{splt}" for splt in val_splits]

        if len(gpus) > 1 and len(val_splits) > 1:
            self._fit_parallel(train, val, val_splits, gpus=gpus, **kwargs)
        else:
            self._fit_sequential(train, val, val_splits, gpus=gpus, **kwargs)

    def _fit_parallel(self, train, val, val_splits, gpus=("0",), **kwargs):
        n_gpus = len(gpus)
        parallel = Parallel(n_jobs=n_gpus, verbose=False, pre_dispatch="2*n_jobs")
        self.models = []
        for seed in self.seeds:
            act_models = [
                self.high_level_model_class(
                    *self.model_args, seed=seed, **self.model_kwargs
                )
                for _ in range(len(train))
            ]
            self.models.extend(act_models)

        n_splits = len(train)
        parallel(
            delayed(self.models[i].fit)(
                train[i % n_splits],
                val[i % n_splits],
                val_splits[i % n_splits],
                **kwargs,
                gpus=gpus[i % n_gpus],
            )
            for i in range(len(self.models))
        )

    def _fit_sequential(self, train, val, val_splits, gpus=("0",), **kwargs):
        n_gpus = len(gpus)
        assert n_gpus < 2, "use parallel fit for multi gpus"
        self.models = []
        for seed in self.seeds:
            for i in range(len(train)):
                m = self.high_level_model_class(
                    *self.model_args, seed=seed, **self.model_kwargs
                )
                m.fit(
                    train[i], val[i], val_splits[i], gpus=gpus[0], **kwargs,
                )
                self.models.append(m)

    def get_oof_dict(self):
        return [m.get_oof_dict() for m in self.models]

    @staticmethod
    def _get_oof_pred_col(seed):
        return f"oof_pred_{seed}"

    def oof_dataset_single_seed(self, seed_index):
        n_splits = len(self.models) // len(self.seeds)
        lower_bound = n_splits * seed_index
        upper_bound = lower_bound + n_splits

        oof_datasets = [
            m.oof_dataset()
            for i, m in enumerate(self.models)
            if lower_bound <= i < upper_bound
        ]

        for i, oof_df in enumerate(oof_datasets):
            oof_df["fold"] = i % n_splits

        x = pd.concat(oof_datasets)
        x[self._get_oof_pred_col(self.seeds[seed_index])] = x["oof_pred"]
        return x.drop("oof_pred", axis=1).reset_index(drop=True)

    def oof_dataset(self):
        df = self.oof_dataset_single_seed(0)
        for i in range(1, len(self.seeds)):
            act_preds = self.oof_dataset_single_seed(i)
            pred_col = act_preds.columns[-1]
            df[pred_col] = act_preds[pred_col]
        return df

    def oof_pred_cols(self):
        return [self._get_oof_pred_col(seed) for seed in self.seeds]

    def get_oof_score(self, scorer):
        oof = self.oof_dataset()
        cv_dict = {col: scorer(oof["label"], oof[col]) for col in self.oof_pred_cols()}
        if len(cv_dict) > 1:
            cv_dict["mean"] = np.mean(list(cv_dict.values()))
            cv_dict["oof_ens_mean"] = scorer(
                oof["label"], oof[self.oof_pred_cols()].mean(1)
            )

        return cv_dict
