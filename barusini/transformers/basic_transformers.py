import pandas as pd
import numpy as np

from barusini.transformers.transformer import Transformer


class MissingValueImputer(Transformer):
    def __init__(self, column):
        self.missing = {}
        self.used_cols = [column]

    def fit(self, X, *args, **kwargs):
        for col in self.used_cols:
            min_val = X[col].min()
            if pd.isna(min_val):
                min_val = 0
            imputed_value = min_val - 1

            self.missing[col] = imputed_value
        return self

    def transform(self, X, **kwargs):
        X = X.copy()
        for col, value in self.missing.items():
            if col in X:
                x = X[col].fillna(value)
                X[col] = x
            else:
                print(f"Warning!: Column {col} expected but nor found {self}")
        return X

    def __str__(self):
        base = f"Missing Value Imputer: ["
        for col, value in self.missing.items():
            base += f"'{col}' imputed by '{value}'"
        return base + "]"


class ReorderColumnsTransformer(Transformer):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, **kwargs):
        return X[self.columns]

    def fit(self, *args, **kwargs):
        # Nothing to do
        pass

    def __str__(self):
        return f"Column Reorder/Subset transformer: '{self.columns}'"


class QuantizationTransformer(Transformer):
    def __init__(
        self,
        column,
        n_bins=10,
        return_quantiles=False,
        hide=False,
        output_name=None,
    ):
        self.used_cols = [column]
        self.n_bins = n_bins
        self.return_quantiles = return_quantiles
        self.bins = None
        self.quantile_mapping = None
        self.hide = hide
        self.output_name = output_name

    def fit(self, X, *args, **kwargs):
        quantiles = np.linspace(0, 1, self.n_bins+1)
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
