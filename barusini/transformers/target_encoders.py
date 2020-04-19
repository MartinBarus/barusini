import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from barusini.transformers.encoders import Encoder
from barusini.utils import unique_name, reshape

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


class BaseEncoder(Encoder):
    show_unseen = False

    def __init__(self, encoder=None, target_name="y", **kwargs):
        """BaseEncoder serves as a wrapper of sklearn.Transformer that can
        will be used for Target Encoding. This wrapper allows user to use any
        Transformer to encode regression, binary or multi-class target. In
        case of multi-class target, encoder will encode each target value.

        :param encoder: sklearn.Transformer: Transformer to be used for encoding
        :param target_name: str: optional: target output name
        :param kwargs: omitted
        """
        super().__init__(**kwargs)
        self.encoder_prototype = encoder
        self.encoders = []
        self.target_name_pattern = target_name
        self.target_names = []

    def fit(self, X, y, multi_class=False, **kwargs):
        super().fit(X)
        if not multi_class:
            self.encoder_prototype.fit(X, y, **kwargs)
            self.encoders = [copy.deepcopy(self.encoder_prototype)]
            self.target_names = [self.target_name_pattern]
        else:
            y_vals = sorted(y.unique())
            for y_val in y_vals:
                encoder = copy.deepcopy(self.encoder_prototype)
                y_act = 1 * (y == y_val)
                encoder.fit(X, y_act, **kwargs)
                self.encoders.append(encoder)
                self.target_names.append(
                    "{}{}".format(self.target_name_pattern, y_val)
                )
        return self

    def transform(self, X, return_all_cols=True, **kwargs):
        transformed = np.array([enc.predict(X) for enc in self.encoders])
        return pd.DataFrame(transformed.T, columns=self.target_names)


class TargetEncoder(Encoder):
    target_str = "[TE]"  # default target name
    show_unseen = False
    x_dim = 2

    def __init__(
        self,
        fold=None,
        random_seed=42,
        encoder=None,
        multi_class=False,
        create_single_col=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if fold is None:
            fold = KFold(n_splits=5, random_state=random_seed, shuffle=True)

        self.fold = fold
        self.splits = None
        self.predictors = None
        self.main_predictor = None
        self.predictors = []
        self.encoder = BaseEncoder(encoder=encoder)
        self.train_shape = None
        self.target_names = []
        self.multi_class = multi_class
        self.create_single_col = create_single_col

    def create_single_feature(self, X):
        if not self.create_single_col:
            return X
        X = X[self.used_cols].apply(
            lambda x: "_x_".join([str(val) for val in x]), axis=1
        )
        X.name = "_x_".join(self.used_cols)
        X = reshape(X, self.x_dim)
        return X

    def fit(self, X, y, *args, **kwargs):
        super().fit(X)
        X = self.preprocess(X)
        X = reshape(X[self.used_cols], self.x_dim)
        X = self.create_single_feature(X)
        splits = []
        predictors = []

        columns = [X.name] if len(X.shape) == 1 else X.columns
        target_name = ", ".join(list(columns)) + f" {self.target_str}"
        target_name = unique_name(X, target_name)

        for train, test in self.fold.split(X):
            splits.append((train, test))
            X_tr, y_tr = X.iloc[train], y.iloc[train]
            enc = copy.deepcopy(self.encoder).fit(
                X_tr, y_tr, multi_class=self.multi_class
            )
            predictors.append(enc)

        if self.multi_class:
            self.target_names = [
                f"{target_name}_{i}" for i in range(y.nunique())
            ]
        else:
            self.target_names = [target_name]
        self.splits = splits
        self.predictors = predictors
        self.main_predictor = copy.deepcopy(self.encoder).fit(
            X, y, multi_class=self.multi_class
        )
        self.train_shape = X.shape
        return self

    def replace_unseen(self, X):
        return X

    def transform(
        self,
        X,
        train_data=False,
        return_all_cols=True,
        remove_original=False,
        **kwargs,
    ):
        X = self.preprocess(X)
        if not train_data:
            new_X = X.copy()
            values = reshape(X[self.used_cols], self.x_dim)
            values = self.create_single_feature(values)
            values = self.main_predictor.transform(values).values
            for i, col in enumerate(self.target_names):
                new_X[col] = values[:, i]
        else:
            new_X = X.copy()
            for (train, test), predictor in zip(self.splits, self.predictors):
                act_new_X = reshape(X.iloc[test][self.used_cols], self.x_dim)
                act_new_X = self.create_single_feature(act_new_X)
                act_new_X = predictor.transform(act_new_X).values
                for i, col in enumerate(self.target_names):
                    if col not in new_X.columns:
                        new_X[col] = 0
                    col_idx = new_X.columns.tolist().index(col)
                    new_X.iloc[test, col_idx] = act_new_X[:, i]

        for target_name in self.target_names:
            new_X[target_name] = new_X[target_name].astype(float)

        if not return_all_cols:
            return new_X[self.target_names]

        if remove_original:
            new_X = new_X.drop(self.used_cols, axis=1)

        return new_X

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y)
        transformed_X = self.transform(X, train_data=True, return_all_cols=True)
        return transformed_X

    def output_columns(self):
        return self.target_names

    def __str__(self):
        if self.main_predictor is not None:
            encoder_str = str(self.main_predictor.mean)
        else:
            encoder_str = "Unfitted Transformer"
        return (
            f"Target encoder for feature '{self.used_cols}'" f":\n{encoder_str}"
        )


class MeanTargetEncoder(TargetEncoder):
    def __init__(self, **kwargs):
        super().__init__(
            encoder=Pipeline(
                steps=[
                    ("enc", OneHotEncoder(handle_unknown="ignore")),
                    ("mean", LinearRegression()),
                ]
            ),
            **kwargs,
        )
