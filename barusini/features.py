###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################

import copy
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed

from barusini.transformers import (
    CustomOneHotEncoder,
    CustomLabelEncoder,
    MissingValueImputer,
    Pipeline,
    TargetEncoder,
)
from barusini.utils import (
    format_time,
    get_terminal_size,
    load_object,
    save_object,
)


ESTIMATOR = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
CV = StratifiedKFold(n_splits=3, random_state=42)
SCORER = roc_auc_score
MAXIMIZE = True
STAGE_NAME = "STAGE"
TERMINAL_COLS = get_terminal_size()
ALLOWED_CAT_ENCODERS = [CustomOneHotEncoder, CustomLabelEncoder, TargetEncoder]
MAX_RELATIVE_CARDINALITY = 0.9


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


def drop_uniques(X, thr=0.99):
    nunique = X.nunique()
    dropped_cols = []
    for x in nunique.index:
        if (nunique[x] / X.shape[0]) >= thr:
            dropped_cols.append(x)

    return dropped_cols


def subset_numeric_features(X):
    ignored_columns = X.select_dtypes(object).columns
    numeric_columns = [col for col in X if col not in ignored_columns]
    X = X[numeric_columns]
    return X


def basic_preprocess(X):
    X = subset_numeric_features(X)
    dropped = drop_uniques(X)
    transformers = []
    for column in X:
        if column not in dropped:
            transformers.append(MissingValueImputer(column))

    pipeline = Pipeline(transformers, ESTIMATOR)
    pipeline.fit(X, y)
    return pipeline


def feature_reduction_generator(model):
    for idx in trange(len(model.transformers)):
        new_model = copy.deepcopy(model)
        # print("Deleting", new_model.transformers[idx])
        del new_model.transformers[idx]
        yield new_model


def dummy_base_line(x):
    return x


def validation(model, X, y, train, test, scoring):
    trn_X = X.loc[train]
    trn_y = y.loc[train]
    model.fit(trn_X, trn_y)

    tst_X = X.loc[test]
    tst_y = y.loc[test]
    predictions = model.predict(tst_X)
    score = scoring(tst_y, predictions)
    return score


def cross_val_score(model, X, y, cv, scoring, n_jobs):
    parallel = Parallel(n_jobs=n_jobs, verbose=False, pre_dispatch="2*n_jobs")
    scores = parallel(
        delayed(validation)(copy.deepcopy(model), X, y, train, test, scoring)
        for train, test in cv.split(X, y)
    )

    return np.mean(scores), model


def best_alternative_model(
    alternative_pipelines,
    base_score,
    maximize,
    cv,
    metric,
    cv_n_jobs,
    alternative_n_jobs,
):
    parallel = Parallel(
        n_jobs=alternative_n_jobs, verbose=False, pre_dispatch="2*n_jobs"
    )
    result = parallel(
        delayed(cross_val_score)(
            pipeline, X, y, cv=cv, scoring=metric, n_jobs=cv_n_jobs
        )
        for pipeline in alternative_pipelines
    )

    best_pipeline = None
    for act_score, act_pipeline in result:
        if is_new_better(base_score, act_score, maximize):
            base_score = act_score
            best_pipeline = act_pipeline

    return best_pipeline, base_score


def generic_change(
    X,
    y,
    model_pipeline,
    stage_name=STAGE_NAME,
    cv=CV,
    metric=SCORER,
    maximize=MAXIMIZE,
    generator=None,
    recursive=False,
    cv_n_jobs=-1,
    alternative_n_jobs=1,
):
    print(format_str("Starting stage {}".format(stage_name)))
    base_score, _ = cross_val_score(
        model_pipeline, X, y, cv=cv, n_jobs=-1, scoring=metric
    )
    print("BASE", base_score)
    original_best = base_score
    old_cols = list(model_pipeline.transform(X).columns)
    while True:
        best_pipeline, base_score = best_alternative_model(
            generator(model_pipeline),
            base_score,
            maximize,
            cv,
            metric,
            cv_n_jobs,
            alternative_n_jobs,
        )
        if best_pipeline is not None:
            model_pipeline = best_pipeline
            model_pipeline.fit(X, y)
            print("CURRENT BEST", base_score)
        else:
            break
        if not recursive:
            break

    new_cols = list(model_pipeline.transform(X).columns)
    print("ORIGINAL BEST", original_best)
    print("NEW BEST", base_score)
    print("DIFF", abs(base_score - original_best))
    print("Dropped", [x for x in old_cols if x not in new_cols])
    # print("Left", [x for x in old_cols if x in new_cols])
    print("New", [x for x in new_cols if x not in old_cols])
    print(format_str("Stage {} finished".format(stage_name)))
    return model_pipeline


def find_best_subset(X, y, model, **kwargs):
    return generic_change(
        X,
        y,
        model,
        stage_name="Finding best subset",
        generator=feature_reduction_generator,
        recursive=True,
        cv_n_jobs=1,
        alternative_n_jobs=-1,
        **kwargs,
    )


def get_encoding_generator(feature, drop=False):
    def categorical_encoding_generator(model):
        encoders = ALLOWED_CAT_ENCODERS
        for enc_class in trange(encoders):
            enc = enc_class(used_cols=[feature])
            new_model = copy.deepcopy(model)
            if drop:
                new_model.remove_transformers([feature], partial_match=False)
            new_model = new_model.add_transformators([enc])
            yield new_model

    return categorical_encoding_generator


def encode_categoricals(X, y, model, **kwargs):
    X_ = model.transform(X)
    categoricals = X_.select_dtypes(object).columns
    del X_
    print("Encoding stage for ", categoricals)
    for feature in trange(categoricals):
        if X[feature].nunique() / X.shape[0] > MAX_RELATIVE_CARDINALITY:
            continue  # Skip
        model = generic_change(
            X,
            y,
            model,
            stage_name="Encoding categoricals {}".format(feature),
            generator=get_encoding_generator(feature),
            **kwargs,
        )
    return model


def recode_categoricals(X, y, model, max_unique=50, **kwargs):

    transformed_X = model.transform(X)
    transformed_X = subset_numeric_features(transformed_X)
    used = [c for c in transformed_X if c in model.used_cols]
    transformed_X = transformed_X[used]
    nunique = transformed_X.nunique()
    categoricals = [f for f in nunique.index if nunique[f] <= max_unique]
    categoricals = [c for c in categoricals if "[" not in c]
    print("Trying to recode following categorical values:", categoricals)
    for feature in trange(categoricals):
        model = generic_change(
            X,
            y,
            model,
            stage_name="Recoding {}".format(feature),
            generator=get_encoding_generator(feature, drop=True),
            **kwargs,
        )
    return model


def auto_ml(X, y, model_path=None, **kwargs):
    model = basic_preprocess(X.copy())
    model = find_best_subset(X, y, model, **kwargs)
    model = encode_categoricals(X, y, model, **kwargs)
    model = recode_categoricals(X, y, model, **kwargs)
    if model_path:
        save_object(model, model_path)

    cols = sorted(list(model.transform(X).columns))
    # trans_reodred = [ReorderColumnsTransformer(list(X_.columns))]
    print("Final features:", cols)

    return model


def predict(X, model, included_columns, output_path, probability):
    print(model)
    included = None
    if included_columns:
        included = X[included_columns].copy()
    if probability:
        predicted = model.predict_proba(X)
    else:
        predicted = model.predict(X)

    if len(predicted.shape) == 1:
        cols = [model.target]
    else:
        cols = [f"{model.target}.{i}" for i in range(predicted.shape[1])]

    predicted = pd.DataFrame(predicted, columns=cols)
    if included is not None:
        predicted = included.reset_index(drop=True).join(
            predicted.reset_index(drop=True)
        )

    predicted.to_csv(output_path, index=False)
    return predicted


if __name__ == "__main__":
    import argparse

    start = time.time()
    parser = argparse.ArgumentParser(description="CLI AutoML Tool")
    parser.add_argument("input", type=str, help="input csv path")
    parser.add_argument("--target", type=str, help="target name str")
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

    parser.add_argument(
        "--predict-proba",
        action="store_true",
        default=False,
        help="predict probability (default: train a model)",
    )

    parser.add_argument(
        "--drop", metavar="N", type=str, nargs="+", help="drop original columns"
    )

    parser.add_argument(
        "--include",
        metavar="N",
        type=str,
        nargs="+",
        help="include original columns",
    )

    parser.add_argument(
        "--output", type=str, help="output csv path", default="predictions.csv"
    )
    args = parser.parse_args()

    file = args.input
    target = args.target
    model_file = args.model

    X = pd.read_csv(file)
    if args.predict or args.predict_proba:
        transformers = load_object(model_file)
        predict(X, transformers, args.include, args.output, args.predict_proba)
    else:
        y = X[target]
        dropped = args.drop
        dropped = dropped if dropped else []
        dropped = dropped + [target]
        X = X.drop(dropped, axis=1)
        X_ = auto_ml(X, y, model_path=model_file)

    duration = format_time(time.time() - start)
    print(f"Duration: {duration}")
