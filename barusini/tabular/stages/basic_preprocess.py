import pandas as pd

from barusini.tabular.stages.base_stage import subset_numeric_features
from barusini.tabular.transformers import (
    Identity,
    MeanTargetEncoder,
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


def drop_constants(X):
    nunique = X.nunique()
    return nunique[nunique == 1].index.tolist()


def get_pipeline(
    X,
    estimator,
    impute_all=False,
    encoding_dict={},
    default_encoder=MeanTargetEncoder,
):
    dropped = drop_uniques(X)
    dropped = dropped + drop_constants(X)
    X_numeric = subset_numeric_features(X)
    missing_columns = X_numeric.apply(lambda x: any(pd.isna(x)))
    missing_columns = missing_columns[missing_columns].index
    transformers = []
    for column in X_numeric:
        if column not in dropped:
            if column in missing_columns or impute_all:
                transformers.append(MissingValueImputer(used_cols=[column]))
            else:
                transformers.append(Identity(used_cols=[column]))
    if encoding_dict or default_encoder:
        skipped = dropped + list(X_numeric.columns)
        cat_cols = [col for col in X.columns if col not in skipped]
        for column in cat_cols:
            enc = encoding_dict.get(column, default_encoder)
            transformers.append(enc(used_cols=[column]))

    pipeline = Pipeline(transformers, estimator)
    return pipeline


@duration("Basic Preprocessing")
def basic_preprocess(X, y, estimator, impute_all=False):
    pipeline = get_pipeline(
        X, estimator, impute_all=impute_all, default_encoder=None
    )
    pipeline.fit(X, y)
    return pipeline
