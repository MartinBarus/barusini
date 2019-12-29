###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################


from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

TARGET_STR = "y"  # default target name
INDEX_STR = "index"  # default temporary index name


def unique_name(X, name):
    while name in X.columns:
        name += str(np.random.randint(10))
    return name


def make_dataframe(X):
    if type(X) is pd.Series:
        X = pd.DataFrame({X.name: X})
    return X


class CustomLabelEncoder:
    def __init__(self):
        self.enc = LabelEncoder()

    def fit_transform(self, X, *args):
        x = self.enc.fit_transform(X)
        return pd.Series(x, index=X.index, name=X.name)


class MeanEncoder:
    def __init__(self):
        self.mean = None
        self.columns = None
        self.target_name = None
        self.missing_value = None
        self.target_name = None

    def fit(self, X, y, target_name=None):
        if target_name is None:
            target_name = unique_name(X, TARGET_STR)
        y = pd.DataFrame({target_name: y})
        x = pd.concat([X, y], axis=1)
        self.columns = list(X.columns)
        self.mean = x.groupby(self.columns).mean()
        self.target_name = target_name
        self.missing_value = x[target_name].min() - 1
        return self

    def transform(self, X):
        orig_index_name = X.index.name
        index_name = unique_name(X, INDEX_STR)
        X.index.name = index_name
        X = (
            X.reset_index()
            .set_index(self.columns)
            .join(self.mean)
            .reset_index()
            .set_index(index_name)
        )
        X.index.name = orig_index_name
        X[self.target_name] = X[self.target_name].fillna(self.missing_value)
        return X.sort_index()

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)


class TargetEncoder:
    def __init__(self, fold=None, random_seed=42, encoder=MeanEncoder):
        if fold is None:
            fold = KFold(n_splits=5, random_state=random_seed)

        self.fold = fold
        self.splits = None
        self.predictors = None
        self.main_predictor = None
        self.encoder = encoder
        self.train_shape = None
        self.target_name = None

    def fit(self, X, y):
        X = make_dataframe(X)
        splits = []
        predictors = []
        target_name = ", ".join(list(X.columns)) + f" {TARGET_STR}"
        target_name = unique_name(X, target_name)
        self.target_name = target_name
        for train, test in self.fold.split(X):
            splits.append((train, test))
            XX, yy = X.iloc[train], y.iloc[train]
            enc = self.encoder().fit(XX, yy, target_name=self.target_name)
            predictors.append(enc)

        self.splits = splits
        self.predictors = predictors
        self.main_predictor = self.encoder().fit(
            X, y, target_name=self.target_name
        )
        self.train_shape = X.shape
        return self

    def transform(self, X, train_data=False, return_all_cols=False):
        X = make_dataframe(X)
        if not train_data:
            transformed_X = self.main_predictor.transform(X)

        else:
            transformed_X = X.copy()
            transformed_X[self.target_name] = None
            for (train, test), predictor in zip(self.splits, self.predictors):
                partial_transformed_X = predictor.transform(X.iloc[test])
                transformed_X.iloc[test] = partial_transformed_X

        if not return_all_cols:
            return transformed_X[self.target_name]
        return transformed_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        transformed_X = self.transform(X, True)
        return transformed_X
