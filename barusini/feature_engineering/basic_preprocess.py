import pandas as pd

from barusini.feature_engineering.generic_stage import subset_numeric_features
from barusini.transformers import (
    Identity,
    MissingValueImputer,
    Pipeline,
)
from barusini.utils import duration


def drop_uniques(X, thr=0.99):
    nunique = X.nunique()
    dropped_cols = []
    for x in nunique.index:
        # only drop unique ints/strings, not floats
        if "float" not in str(X.dtypes[x]) and (nunique[x] / X.shape[0]) >= thr:
            dropped_cols.append(x)

    return dropped_cols


@duration("Basic Preprocessing")
def basic_preprocess(X, y, estimator, impute_all=False):
    X = subset_numeric_features(X)
    dropped = drop_uniques(X)
    missing_columns = X.apply(lambda x: any(pd.isna(x)))
    missing_columns = missing_columns[missing_columns].index
    transformers = []
    for column in X:
        if column not in dropped:
            if column in missing_columns or impute_all:
                transformers.append(MissingValueImputer(used_cols=[column]))
            else:
                transformers.append(Identity(used_cols=[column]))

    pipeline = Pipeline(transformers, estimator)
    pipeline.fit(X, y)
    return pipeline
