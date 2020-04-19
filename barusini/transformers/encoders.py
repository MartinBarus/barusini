import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from barusini.transformers.transformer import Transformer
from barusini.utils import sanitize, make_dataframe


INDEX_STR = "index"  # default temporary index name
JOIN_STR = "___XXX___YYY___"


class Encoder(Transformer):
    show_unseen = True
    missing_val = "MISSING"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unique_values = None
        self.top_vals = None

    def preprocess(self, X):
        x = make_dataframe(X)
        x[self.used_cols] = x[self.used_cols].fillna(self.missing_val)
        if self.show_unseen:
            self.replace_unseen(x)
        return x

    @staticmethod
    def join_cols(cols):
        return JOIN_STR.join([str(x) for x in cols])

    def fit_unique(self, X):
        vals = X[self.used_cols].apply(self.join_cols, axis=1)
        self.unique_values = set(vals)
        self.top_vals = vals.value_counts().index.values[0].split(JOIN_STR)

    def fit(self, X, *args, **kwargs):
        X = make_dataframe(X)
        if self.used_cols is None:
            self.used_cols = list(X.columns)
        if self.show_unseen:
            self.fit_unique(X)

    def replace_unseen(self, X):
        vals = X[self.used_cols].apply(self.join_cols, axis=1)
        mask = ~vals.isin(self.unique_values)
        unseen_vals = vals.loc[mask].drop_duplicates()
        unseen_vals = [x.split(JOIN_STR) for x in unseen_vals]
        n_unseen = len(unseen_vals)
        if n_unseen:
            print(
                f"WARNING!: {n_unseen} unseen values for {self.used_cols}"
            )
        X.loc[mask, self.used_cols] = self.top_vals
        return X


class GenericEncoder(Encoder):
    is_sparse = False

    def __init__(self, used_cols=None, **kwargs):
        super().__init__(used_cols=used_cols)
        self.encoder = None
        self.target_names = None

    def fit_names(self):
        pass

    def fit(self, X, *args, **kwargs):
        super().fit(X)
        processed = self.preprocess(X)
        self.encoder.fit(processed,  *args)
        assert (
            X.shape[0] == processed.shape[0]
        ), f"Expected to see {X.shape[0]} rows, found {processed.shape[0]}"
        self.fit_names()
        return self

    def transform(self, X, **kwargs):
        x = self.preprocess(X)
        x = self.encoder.transform(x)
        if self.is_sparse:
            x = x.todense()
        x = pd.DataFrame(x, columns=self.target_names, index=X.index)
        result = X.join(x).drop(self.used_cols, axis=1)
        return result

    def preprocess(self, X):
        x = super().preprocess(X)
        return x[self.used_cols]

    def output_columns(self):
        return self.target_names


class CustomLabelEncoder(GenericEncoder):
    def __init__(self, used_cols=None, **kwargs):
        super().__init__(used_cols=used_cols)
        self.encoder = LabelEncoder()

    def preprocess(self, X):
        x = super().preprocess(X)
        return x.values.reshape(-1)

    def fit_names(self):
        self.target_names = [sanitize(f"{self.used_cols} [LE]")]

    def __str__(self):
        if hasattr(self.encoder, "categories_"):
            encoder_str = str(self.encoder.classes_[0])
        else:
            encoder_str = "Unfitted Transformer"

        return (
            f"Label Encoder for feature '{self.used_cols}':\n\t"
            f"Categories: {encoder_str}"
        )


class CustomOneHotEncoder(GenericEncoder):
    def __init__(self, used_cols=None, **kwargs):
        super().__init__(used_cols=used_cols)
        self.encoder = OneHotEncoder(
            sparse=False, categories="auto", handle_unknown="ignore"
        )

    def preprocess(self, X):
        x = super().preprocess(X)
        return x.values.reshape(-1, 1)

    def fit_names(self):
        self.target_names = [
            sanitize(f"{self.used_cols} [OHE:{val}]")
            for val in self.encoder.categories_[0]
        ]

    def __str__(self):
        if hasattr(self.encoder, "categories_"):
            encoder_str = str(self.encoder.categories_[0])
        else:
            encoder_str = "Unfitted Transformer"

        return (
            f"One Hot Encoder for feature '{self.used_cols}':\n\t"
            f"Categories: {encoder_str}"
        )
