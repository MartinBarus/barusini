import pandas as pd

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