from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from barusini.transformers.transformer import Transformer
from barusini.utils import unique_name, sanitize, make_dataframe, unique_value
import pandas as pd

TARGET_STR = "[TE]"  # default target name
INDEX_STR = "index"  # default temporary index name


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


class Encoder(Transformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frequencies = None
        self.unseen_value = None

    def get_frequencies(self, X):
        new_x = make_dataframe(X[self.used_cols]).copy()
        new_x["cnt"] = 1
        return new_x.groupby(self.used_cols).count()["cnt"] / X.shape[0]

    def fit_unseen_values(self, X):
        self.unseen_value = unique_value(X.values[:], "MISS")

    def fit_frequencies(self, X):
        self.frequencies = self.get_frequencies(X)

    def fit(self, X):
        X = make_dataframe(X)
        if self.used_cols is None:
            self.used_cols = list(X.columns)
        self.fit_unseen_values(X)
        self.fit_frequencies(X)

    def preprocess(self, X):
        x = make_dataframe(X)
        x[self.used_cols] = x[self.used_cols].fillna(self.unseen_value)
        x[self.used_cols] = self.replace_unseen(x[self.used_cols])
        return x

    def get_unseen_replace_map(self, X):
        new_frequencies = self.get_frequencies(X)
        unseen_values = [
            x for x in new_frequencies.index if x not in self.frequencies.index
        ]
        n_unseen = len(unseen_values)
        if n_unseen:
            print(
                f"WARNING!: {n_unseen} unseen values for {self.used_cols}: "
                f"{unseen_values}"
            )

        replace_map = {}
        for unseen in unseen_values:
            min_dist = None
            replacement = None
            for seen in self.frequencies.index:
                act_dist = (
                    new_frequencies.loc[unseen] - self.frequencies.loc[seen]
                )
                if min_dist is None or act_dist < min_dist:
                    min_dist = act_dist
                    replacement = seen

            if len(self.used_cols) == 1:
                unseen = (unseen,)
                replacement = (replacement,)
            replace_map[unseen] = replacement
        return replace_map

    def replace_unseen(self, X):
        replace_map = self.get_unseen_replace_map(X)
        for old, new in replace_map.items():
            to_replace = {col: val for col, val in zip(self.used_cols, old)}
            value = {col: val for col, val in zip(self.used_cols, new)}
            X = X.replace(to_replace, value)
        return X


class TargetEncoder(Encoder):
    def __init__(
        self, fold=None, random_seed=42, encoder=MeanEncoder, **kwargs
    ):
        super().__init__(**kwargs)
        if fold is None:
            fold = KFold(n_splits=5, random_state=random_seed)

        self.fold = fold
        self.splits = None
        self.predictors = None
        self.main_predictor = None
        self.encoder = encoder
        self.train_shape = None
        self.target_name = None

    def fit(self, X, y, **kwargs):
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


class GenericEncoder(Encoder):
    encoder_class = None
    defaults = {}

    def __init__(self, used_cols=None, **kwargs):
        super().__init__(used_cols=used_cols)
        self.encoder = self.encoder_class(**self.defaults)

    def fit(self, X, *args, **kwargs):
        super().fit(X)
        processed = self.preprocess(X)
        # print("Now", processed)
        self.encoder.fit(processed)
        assert (
            X.shape[0] == processed.shape[0]
        ), f"Expected to see {X.shape[0]} rows, found {processed.shape[0]}"
        return self

    def preprocess(self, X):
        x = super().preprocess(X)
        return x[self.used_cols]


class CustomLabelEncoder(GenericEncoder):
    encoder_class = LabelEncoder
    # defaults = dict(handle_unknown="ignore")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_name = None

    def preprocess(self, X):
        x = super().preprocess(X)
        return x.values.reshape(-1)

    def fit(self, X, *args, **kwargs):
        super().fit(X, *args, **kwargs)
        self.target_name = sanitize(f"{self.used_cols} [LE]")

    def transform(self, X, **kwargs):
        x = self.preprocess(X)
        x = self.encoder.transform(x)

        x = pd.Series(x, name=self.target_name, index=X.index)
        result = X.join(x).drop(self.used_cols, axis=1)
        return result

    def __str__(self):
        if hasattr(self.encoder, "categories_"):
            encoder_str = str(self.encoder.classes_[0])
        else:
            encoder_str = "Unfitted Transformer"

        return (
            f"Label Encoder for feature '{self.used_cols}':\n\t"
            f"Categories: {encoder_str}"
        )

    def output_columns(self):
        return [self.target_name]


class CustomOneHotEncoder(GenericEncoder):
    encoder_class = OneHotEncoder
    defaults = dict(sparse=False, categories="auto", handle_unknown="ignore")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_names = None

    def preprocess(self, X):
        x = super().preprocess(X)
        return x.values.reshape(-1, 1)

    def fit(self, X, *args, **kwargs):
        super().fit(X, *args, **kwargs)
        self.target_names = [
            sanitize(f"{self.used_cols} [OHE:{val}]")
            for val in self.encoder.categories_[0]
        ]

    def transform(self, X, **kwargs):
        assert len(self.encoder.categories_) == 1, "OHE categories level issue"
        x = self.preprocess(X)
        x = self.encoder.transform(x)
        x = pd.DataFrame(x, columns=self.target_names, index=X.index)
        result = X.join(x).drop(self.used_cols, axis=1)
        return result

    def output_columns(self):
        return self.target_names

    def __str__(self):
        if hasattr(self.encoder, "categories_"):
            encoder_str = str(self.encoder.categories_[0])
        else:
            encoder_str = "Unfitted Transformer"

        return (
            f"One Hot Encoder for feature '{self.used_cols}':\n\t"
            f"Categories: {encoder_str}"
        )
