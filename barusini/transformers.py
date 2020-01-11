###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################


from category_encoders import BinaryEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

TARGET_STR = "[TE]"  # default target name
INDEX_STR = "index"  # default temporary index name


def unique_name(X, name):
    while name in X.columns:
        name += str(np.random.randint(10))
    return name


def make_dataframe(X):
    if type(X) is pd.Series:
        X = pd.DataFrame({X.name: X})
    return X


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

    def __str__(self):
        return f"Mean encoder for feature '{self.target_name}':\n\t{self.mean}"


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
        self.used_cols = None

    def fit(self, X, y):
        X = make_dataframe(X)
        splits = []
        predictors = []
        target_name = ", ".join(list(X.columns)) + f" {TARGET_STR}"
        target_name = unique_name(X, target_name)
        self.target_name = target_name
        self.used_cols = list(X.columns)
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

    def transform(
        self, X, train_data=False, return_all_cols=True, remove_original=True
    ):
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
            return transformed_X[[self.target_name]]

        if remove_original:
            transformed_X = transformed_X.drop(self.used_cols, axis=1)

        return transformed_X

    def fit_transform(self, X, y):
        self.fit(X, y)
        transformed_X = self.transform(
            X, train_data=True, return_all_cols=False
        )
        return transformed_X

    def __str__(self):
        return (
            f"Target encoder for feature '{self.used_cols}'"
            f":\n{self.main_predictor.mean}"
        )


class ColumnDropTransformer:
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)

    def __str__(self):
        return f"Column Drop Transformer: {self.columns_to_drop}"


class MissingValueImputer:
    def __init__(self):
        self.missing = {}
        self.col_dropper = None

    def fit(self, X):
        dropped = []
        for col in X:
            min_val = X[col].min()
            if pd.isna(min_val):
                min_val = 0
            imputed_value = min_val - 1
            x = X[col].fillna(imputed_value)
            if x.std() == 0:
                dropped.append(col)
            else:
                self.missing[col] = imputed_value
        self.col_dropper = ColumnDropTransformer(dropped)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = self.col_dropper.transform(X)
        for col, value in self.missing.items():
            X[col] = X[col].fillna(value)
        return X

    def __str__(self):
        base = f"Missing Value Imputer:\n\t{str(self.col_dropper)}\n"
        for col, value in self.missing.items():
            base += f"\tColumn {col} imputed by '{value}'\n"
        return base


class ReorderColumnsTransformer:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        return X[self.columns]

    def __str__(self):
        return f"Column Reorder/Subset transformer: '{self.columns}'"


class GenericEncoder:
    encoder_class = None
    kwargs = {}

    def __init__(self):
        self.encoder = self.encoder_class(**self.kwargs)
        self.used_col = None

    def preprocess(self, X):
        return X

    def fit_transform(self, X, *args):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, *args):
        X = make_dataframe(X)
        processed = self.preprocess(X)
        self.encoder.fit(processed)
        assert X.shape[1] == 1, (
            f"Encoder is expecting single column input, "
            f"found {X.shape[0]} columns"
        )
        self.used_col = X.columns[0]

    def transform(self, X):
        pass


class CustomLabelEncoder(GenericEncoder):
    encoder_class = LabelEncoder

    def preprocess(self, X):
        return X.values.reshape(-1)

    def transform(self, X):
        X = make_dataframe(X)
        new_name = f"{self.used_col} [LE]"
        x = self.encoder.transform(X[self.used_col])
        x = pd.Series(x, name=new_name, index=X.index)
        result = X.join(x).drop(self.used_col, axis=1)
        return result

    def __str__(self):
        return (
            f"Label Encoder for feature '{self.used_col}':\n\t"
            f"Categories: {self.encoder.classes_[0]}"
        )


class CustomOneHotEncoder(GenericEncoder):
    encoder_class = OneHotEncoder
    kwargs = dict(sparse=False, categories="auto")

    def preprocess(self, X):
        return X.values.reshape(-1, 1)

    def transform(self, X):
        assert len(self.encoder.categories_) == 1, "OHE categories level issue"
        cols = [
            f"{self.used_col} [OHE:{val}]"
            for val in self.encoder.categories_[0]
        ]
        X = make_dataframe(X)
        x = X[self.used_col]
        x = self.preprocess(x)
        x = self.encoder.transform(x)
        x = pd.DataFrame(x, columns=cols, index=X.index)
        result = X.join(x).drop(self.used_col, axis=1)
        return result

    def __str__(self):
        return (
            f"One Hot Encoder for feature '{self.used_col}':\n\t"
            f"Categories: {self.encoder.categories_[0]}"
        )
