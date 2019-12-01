###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################

import pandas as pd
from category_encoders import BinaryEncoder, CatBoostEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm as tqdm
import os


class CustomLabelEncoder:
    def __init__(self):
        self.enc = LabelEncoder()

    def fit_transform(self, X, *args):
        x = self.enc.fit_transform(X)
        return pd.Series(x, index=X.index, name=X.name)


ESTIMATOR = RandomForestClassifier(n_estimators=100, n_jobs=-1)
CV = StratifiedKFold(n_splits=3)
METRIC = "roc_auc"
MAXIMIZE = True
STAGE_NAME = "STAGE"
_, TERMINAL_COLS = os.popen("stty size", "r").read().split()
TERMINAL_COLS = int(TERMINAL_COLS)


def format_str(x, total_len=TERMINAL_COLS):
    middle = total_len // 2
    num_paddings = middle - len(x) // 2 - 1
    padding = "-" * num_paddings
    result = "{} {} {}".format(padding, x, padding)
    return result


def trange(x):
    if type(x) is int:
        return tqdm(range(x), leave=False)
    return tqdm(x, leave=False)


def is_new_better(old, new, maximize):
    if maximize:
        return old <= new
    return new <= old


def clean_missing(X):
    missing = {}
    for col in X:
        min_val = X[col].min()
        if pd.isna(min_val):
            min_val = 0
        X[col] = X[col].fillna(min_val - 1)
        if X[col].std() == 0:
            X = X.drop(col, axis=1)
        else:
            missing[col] = min_val - 1
    return X, missing


def drop_categoricals(X):
    return X.drop(X.select_dtypes(object).columns, axis=1)


def drop_uniques(X, thr=0.99):
    nunique = X.nunique()
    for x in nunique.index:
        if (nunique[x] / X.shape[0]) >= thr:
            X = X.drop(x, axis=1)
    return X


def basic_preprocess(X):
    X = drop_uniques(X)
    X = drop_categoricals(X)
    X, missing = clean_missing(X)
    return X


def feature_reduction_generator(X):
    for i in trange(X.shape[1]):
        yield X.drop(X.columns[i], axis=1)


def dummy_base_line(x):
    return x


def generic_change(
    X_old,
    y,
    stage_name=STAGE_NAME,
    cv=CV,
    estimator=ESTIMATOR,
    metric=METRIC,
    maximize=MAXIMIZE,
    get_baseline_X=dummy_base_line,
    generator=None,
    recursive=False,
):
    print(format_str("Starting stage {}".format(stage_name)))
    X = get_baseline_X(X_old).copy()
    base_score = cross_val_score(
        estimator, X, y, cv=cv, n_jobs=-1, scoring=metric
    ).mean()
    print("BASE", base_score)
    original_best = base_score
    while True:
        act_best = None
        for X_act in generator(X):
            act_score = cross_val_score(
                estimator, X_act, y, cv=cv, n_jobs=-1, scoring=metric
            ).mean()
            if is_new_better(base_score, act_score, maximize):
                base_score = act_score
                act_best = X_act

        if act_best is not None:
            X = act_best
            print("CURRENT BEST", base_score)
        else:
            break
        if not recursive:
            break
    print("ORIGINAL BEST", original_best)
    print("NEW BEST", base_score)
    print("DIFF", abs(base_score - original_best))
    print("Dropped", [x for x in X_old.columns if x not in X.columns])
    print("Left", [x for x in X.columns])
    print("New", [x for x in X.columns if x not in X_old.columns])
    print(format_str("Stage {} finished".format(stage_name)))
    return X


def find_best_subset(X, y, **kwargs):
    return generic_change(
        X,
        y,
        stage_name="Finding best subset",
        generator=feature_reduction_generator,
        recursive=True,
        **kwargs
    )


def get_encoding_generator(feature, target, drop=False):
    def categorical_encoding_generator(X):
        encoders = [CustomLabelEncoder, BinaryEncoder, CatBoostEncoder]
        for enc_class in trange(encoders):
            enc = enc_class()
            enc_str = enc.__class__.__name__
            new = enc.fit_transform(feature, target)
            if type(new) is pd.Series:
                new.name = new.name + " " + enc_str
            else:
                name_map = {x: "{} {}".format(x, enc_str) for x in new.columns}
                new = new.rename(columns=name_map)

            if drop:
                X_ = X.drop(feature.name, axis=1)
            else:
                X_ = X
            X_ = X_.join(new)
            yield X_

    return categorical_encoding_generator


def encode_categoricals(X_all, y, features_to_use, categoricals=None, **kwargs):
    X = X_all[features_to_use]
    if categoricals is None:
        categoricals = X_all.select_dtypes(object).columns

    for feature in trange(categoricals):
        X = generic_change(
            X,
            y,
            stage_name="Encoding categoricals {}".format(feature),
            generator=get_encoding_generator(X_all[feature], y),
            **kwargs
        )
    return X


def recode_categoricals(X, y, max_unique=10, **kwargs):
    nunique = X.nunique()
    categoricals = [f for f in nunique.index if nunique[f] <= max_unique]
    print("Trying to recode following categorical values:", categoricals)
    for feature in trange(categoricals):
        X = generic_change(
            X,
            y,
            stage_name="Recoding {}".format(feature),
            generator=get_encoding_generator(X[feature], y, drop=True),
            **kwargs
        )
    return X


def auto_ml(X, y, **kwargs):
    X_ = basic_preprocess(X)
    X_ = find_best_subset(X_, y, **kwargs)
    X_ = encode_categoricals(X, y, X_.columns, **kwargs)
    X_ = recode_categoricals(X_, y, **kwargs)
    cols = sorted([x for x in X_.columns])
    print("Final features:", cols)
    return X_


if __name__ == "__main__":
    import sys

    file = sys.argv[1]
    target = sys.argv[2]
    X = pd.read_csv(file)
    y = X[target]
    X = X.drop(target, axis=1)
    X_ = auto_ml(X, y)
