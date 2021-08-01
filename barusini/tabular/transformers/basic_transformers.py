import numpy as np
import pandas as pd

from barusini.tabular.transformers.base_transformer import Transformer
from barusini.utils import sanitize


class Identity(Transformer):
    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return X

    def __str__(self):
        return f"Identity: {self.used_cols}"


class MissingValueImputer(Transformer):
    def __init__(self, agg="median", **kwargs):
        super().__init__(**kwargs)
        self.agg = agg
        self.missing = {}
        self.new_names = {}

    def fit(self, X, *args, **kwargs):
        aggregation = "min" if self.agg == "new" else self.agg
        for col in self.used_cols:
            value = X[col].agg(aggregation)
            if type(value) is pd.Series:
                if len(value) != 1:
                    assert (
                        self.agg == "mode"
                    ), f"Expected single value for agg {self.agg}, got {value}"
                value = value.values[0]
            value = 0 if pd.isna(value) else value
            if self.agg == "new":
                value -= 1
            self.missing[col] = value
        return self

    def transform(self, X, **kwargs):
        X = X.copy()
        for col, value in self.missing.items():
            if col in X:
                x = X[col].fillna(value)
                X[col] = x
            else:
                print(f"Warning!: Column {col} expected but nor found {self}")
        self.new_names = {
            col: sanitize(f"{col} [{self.agg}]") for col in self.missing
        }
        X = X.rename(columns=self.new_names)
        return X

    def output_columns(self):
        return sorted(self.new_names.values())

    def __str__(self):
        base = f"Missing Value Imputer ({self.agg}): ["
        for col, value in self.missing.items():
            base += f"'{col}' imputed by '{value}'"
        return base + "]"


class ReorderColumnsTransformer(Transformer):
    def transform(self, X, **kwargs):
        return X[self.used_cols]

    def fit(self, *args, **kwargs):
        # Nothing to do
        pass

    def __str__(self):
        return f"Column Reorder/Subset transformer: '{self.used_cols}'"


class QuantizationTransformer(Transformer):
    def __init__(
        self,
        used_cols,
        n_bins=10,
        return_quantiles=False,
        hide=False,
        output_name=None,
    ):
        super().__init__(used_cols=used_cols)
        self.n_bins = n_bins
        self.return_quantiles = return_quantiles
        self.bins = None
        self.quantile_mapping = None
        self.hide = hide
        self.output_name = output_name

    def fit(self, X, *args, **kwargs):
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        quantile_values = np.quantile(X[self.used_cols[0]], quantiles)

        # Map bins 1, 2, 3 to original quantile values 0.1, 0.2, 0.3
        if self.return_quantiles:
            mapping = {}
            for quantile, value in zip(quantiles, quantile_values):
                if value not in mapping:
                    mapping[value] = quantile
            self.quantile_mapping = {
                i + 1: quantile for i, quantile in enumerate(mapping.values())
            }

        self.bins = sorted(list(set(quantile_values)))
        self.bins[0] = float("-inf")
        self.bins[-1] = float("inf")
        return self

    def transform(self, X, **kwargs):
        if self.output_name is None:
            col = self.used_cols[0]
        else:
            col = self.output_name
        X = X.copy()
        X[col] = np.digitize(X[self.used_cols[0]], self.bins)
        if self.return_quantiles:
            X[col] = X[col].map(self.quantile_mapping)
        return X

    def output_columns(self):
        # if transformer is used as intermediate computation, do not show output
        if self.hide:
            return []
        return self.used_cols

    def __str__(self):
        return (
            f"Quantization Transformer: [{self.used_cols[0]} "
            f"binned to {len(self.bins)} bins {self.bins}]"
        )


class DateExtractor(Transformer):
    def __init__(
        self,
        periods=["day", "dayofweek", "month", "dayofyear", "year"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(self.used_cols) == 1
        self.time_col = self.used_cols[0]
        self.periods = periods
        self.output_names = []

    def fit(self, X, *args, **kwargs):
        self.output_names = [f"{self.time_col} _{x}_" for x in self.periods]
        return self

    def transform(self, X, **kwargs):
        features = [X]
        print(self.periods)
        for i, period in enumerate(self.periods):
            feature = getattr(X[self.time_col].dt, period)
            feature.name = self.output_names[i]
            features.append(feature)
        return pd.concat(features, axis=1)

    def output_columns(self):
        return self.output_names

    def __str__(self):
        return f"Date Extractor of feature '{self.time_col}': {self.periods}"
