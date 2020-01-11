###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm as tqdm

from barusini.transformers import (
    ColumnDropTransformer,
    CustomOneHotEncoder,
    CustomLabelEncoder,
    MissingValueImputer,
    ReorderColumnsTransformer,
    TargetEncoder,
)
from barusini.utils import get_terminal_size, load_object, save_object


ESTIMATOR = RandomForestClassifier(n_estimators=100, n_jobs=-1)
CV = StratifiedKFold(n_splits=3)
METRIC = "roc_auc"
MAXIMIZE = True
STAGE_NAME = "STAGE"
TERMINAL_COLS = get_terminal_size()
ALLOWED_CAT_ENCODERS = [CustomOneHotEncoder, CustomLabelEncoder, TargetEncoder]


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
    imputer = MissingValueImputer()
    X = imputer.fit_transform(X)

    return X, imputer


def drop_categoricals(X):
    dropped_cols = X.select_dtypes(object).columns
    dropper = ColumnDropTransformer(dropped_cols)
    X = dropper.transform(X)

    return X, dropper


def drop_uniques(X, thr=0.99):
    nunique = X.nunique()
    dropped_cols = []
    for x in nunique.index:
        if (nunique[x] / X.shape[0]) >= thr:
            dropped_cols.append(x)
    dropper = ColumnDropTransformer(dropped_cols)
    X = dropper.transform(X)

    return X, dropper


def basic_preprocess(X):
    X, unique_dropper = drop_uniques(X)
    X, categorical_dropper = drop_categoricals(X)
    X, imputer = clean_missing(X)
    transformers = [unique_dropper, categorical_dropper, imputer]

    return X, transformers


def feature_reduction_generator(X):
    for i in trange(X.shape[1]):
        col = X.columns[i]
        transformer = ColumnDropTransformer([col])
        yield transformer.transform(X), [transformer]


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
    generator=None,
    recursive=False,
):
    print(format_str("Starting stage {}".format(stage_name)))
    X = X_old.copy()
    base_score = cross_val_score(
        estimator, X, y, cv=cv, n_jobs=-1, scoring=metric
    ).mean()
    print("BASE", base_score)
    original_best = base_score
    transformers = []
    while True:
        act_best = None
        transformers_best = None
        for X_act, transformers_act in generator(X):
            act_score = cross_val_score(
                estimator, X_act, y, cv=cv, n_jobs=-1, scoring=metric
            ).mean()
            if is_new_better(base_score, act_score, maximize):
                base_score = act_score
                act_best = X_act
                transformers_best = transformers_act

        if act_best is not None:
            X = act_best
            transformers.extend(transformers_best)
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
    return X, transformers


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
        encoders = ALLOWED_CAT_ENCODERS
        for enc_class in trange(encoders):
            enc = enc_class()
            new = enc.fit_transform(feature, target)

            if drop:
                X_ = X.drop(feature.name, axis=1)
            else:
                X_ = X
            X_ = X_.join(new)
            yield X_, [enc]

    return categorical_encoding_generator


def encode_categoricals(X_all, y, features_to_use, categoricals=None, **kwargs):
    X = X_all[features_to_use]
    if categoricals is None:
        categoricals = X_all.select_dtypes(object).columns

    transformers = []
    for feature in trange(categoricals):
        X, act_transformers = generic_change(
            X,
            y,
            stage_name="Encoding categoricals {}".format(feature),
            generator=get_encoding_generator(X_all[feature], y),
            **kwargs
        )
        transformers.extend(act_transformers)
    return X, transformers


def recode_categoricals(X, y, max_unique=10, **kwargs):
    nunique = X.nunique()
    categoricals = [f for f in nunique.index if nunique[f] <= max_unique]
    print("Trying to recode following categorical values:", categoricals)
    transformers = []
    for feature in trange(categoricals):
        X, act_transformers = generic_change(
            X,
            y,
            stage_name="Recoding {}".format(feature),
            generator=get_encoding_generator(X[feature], y, drop=True),
            **kwargs
        )
        transformers.extend(act_transformers)
    return X, transformers


def auto_ml(X, y, model_path=None, **kwargs):
    X_, trans_basic = basic_preprocess(X.copy())
    X_, trans_subset = find_best_subset(X_, y, **kwargs)
    X_, trans_encode = encode_categoricals(X, y, X_.columns, **kwargs)
    X_, trans_decode = recode_categoricals(X_, y, **kwargs)
    trans_reodred = [ReorderColumnsTransformer(list(X_.columns))]
    cols = sorted([x for x in X_.columns])
    print("Final features:", cols)
    transformers = (
        trans_basic + trans_subset + trans_encode + trans_decode + trans_reodred
    )
    if model_path:
        save_object(transformers, model_path)

    return X_


def predict(X, transformers):
    for tr in transformers:
        print(tr, "\n")
        X = tr.transform(X)

    print("Final Cols:", list(X.columns))
    return X


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI AutoML Tool")
    parser.add_argument("input", type=str, help="input csv path")
    parser.add_argument("target", type=str, help="target name str")
    parser.add_argument(
        "--model",
        type=str,
        help="model pickle path",
        default="transformers.pickle",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="make prediction (default: train a model)",
    )
    args = parser.parse_args()

    file = args.input
    target = args.target
    model_file = args.model

    X = pd.read_csv(file)
    if args.predict:
        transformers = load_object(model_file)
        predict(X, transformers)
    else:
        y = X[target]
        X = X.drop(target, axis=1)
        X_ = auto_ml(X, y, model_path=model_file)
