import pandas as pd
from sklearn.model_selection import KFold
from barusini.transformers.transformer import Transformer
from barusini.transformers.encoders import Encoder, INDEX_STR
from barusini.utils import unique_name

TARGET_STR = "[TE]"  # default target name


class MeanEncoder(Transformer):
    def __init__(self):
        self.mean = None
        self.columns = None
        self.target_name = None
        self.global_mean = None
        self.target_name = None

    def fit(self, X, y, target_name=None, **kwargs):
        if target_name is None:
            target_name = unique_name(X, TARGET_STR)
        y = pd.DataFrame({target_name: y})
        x = pd.concat([X, y], axis=1)
        self.columns = list(X.columns)
        self.mean = x.groupby(self.columns).mean()
        self.target_name = target_name
        self.global_mean = y.values.mean()
        return self

    def transform(self, X, **kwargs):
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
        unseen_rows = pd.isna(X[self.target_name])
        if any(unseen_rows):
            unseen = X.loc[unseen_rows, self.columns]
            unseen = unseen.apply(
                lambda x: "_".join([str(c) for c in x]), axis=1
            ).value_counts()
            print(
                f"WARNING!: {unseen.shape[0]} unseen values for {self.columns}"
                f", value counts:\n{unseen}"
            )

        X[self.target_name] = X[self.target_name].fillna(self.global_mean)
        return X.sort_index()

    def __str__(self):
        return f"Mean encoder for feature '{self.target_name}':\n\t{self.mean}"


class TargetEncoder(Encoder):
    def __init__(
        self, fold=None, random_seed=42, encoder=MeanEncoder, **kwargs
    ):
        super().__init__(**kwargs)
        if fold is None:
            fold = KFold(n_splits=5, random_state=random_seed, shuffle=True)

        self.fold = fold
        self.splits = None
        self.predictors = None
        self.main_predictor = None
        self.encoder = encoder
        self.train_shape = None
        self.target_name = None

    def fit(self, X, y, *args, **kwargs):
        super().fit(X)
        X = self.preprocess(X)[self.used_cols]
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

    def replace_unseen(self, X):
        return X

    def transform(
        self,
        X,
        train_data=False,
        return_all_cols=True,
        remove_original=True,
        **kwargs,
    ):
        X = self.preprocess(X)
        if not train_data:
            transformed_X = self.main_predictor.transform(X)

        else:
            transformed_X = X.copy()
            transformed_X[self.target_name] = None
            for (train, test), predictor in zip(self.splits, self.predictors):
                partial_transformed_X = predictor.transform(X.iloc[test])
                transformed_X.iloc[test] = partial_transformed_X

        transformed_X[self.target_name] = transformed_X[
            self.target_name
        ].astype(float)

        if not return_all_cols:
            return transformed_X[[self.target_name]]

        if remove_original:
            transformed_X = transformed_X.drop(self.used_cols, axis=1)

        return transformed_X

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y)
        transformed_X = self.transform(X, train_data=True, return_all_cols=True)
        return transformed_X

    def output_columns(self):
        return [self.target_name]

    def __str__(self):
        if self.main_predictor is not None:
            encoder_str = str(self.main_predictor.mean)
        else:
            encoder_str = "Unfitted Transformer"
        return (
            f"Target encoder for feature '{self.used_cols}'" f":\n{encoder_str}"
        )
