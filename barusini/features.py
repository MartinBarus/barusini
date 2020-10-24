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
from joblib import Parallel, delayed

from tqdm import tqdm as tqdm
from barusini.constants import (
    ESTIMATOR,
    CV,
    STAGE_NAME,
    TERMINAL_COLS,
    MAX_RELATIVE_CARDINALITY,
    MAX_ABSOLUTE_CARDINALITY,
)
from barusini.transformers import (
    CustomLabelEncoder,
    CustomOneHotEncoder,
    LinearTextEncoder,
    MeanTargetEncoder,
    MissingValueImputer,
    Pipeline,
    TfIdfEncoder,
    TfIdfPCAEncoder,
)
from barusini.utils import (
    deepcopy,
    duration,
    get_default_settings,
    get_probability,
    load_object,
    save_object,
)


ALLOWED_TRANSFORMERS = (
    CustomLabelEncoder,
    CustomLabelEncoder,
    MeanTargetEncoder,
    TfIdfPCAEncoder,
    TfIdfEncoder,
    LinearTextEncoder,
)


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


@duration("Basic Preprocessing")
def basic_preprocess(X, y, estimator):
    X = subset_numeric_features(X)
    dropped = drop_uniques(X)
    transformers = []
    for column in X:
        if column not in dropped:
            transformers.append(MissingValueImputer(used_cols=[column]))

    pipeline = Pipeline(transformers, estimator)
    pipeline.fit(X, y)
    return pipeline


def feature_reduction_generator(model):
    for idx in trange(len(model.transformers)):
        new_model = deepcopy(model)
        del new_model.transformers[idx]
        yield new_model


def dummy_base_line(x):
    return x


def validation(model, X, y, train, test, scoring, proba=None):
    proba = get_probability(scoring) if proba is None else proba
    trn_X = X.loc[train]
    trn_y = y.loc[train]
    model.fit(trn_X, trn_y)

    tst_X = X.loc[test]
    tst_y = y.loc[test]
    if proba:
        predictions = model.predict_proba(tst_X)
        if predictions.shape[1] == 2:
            predictions = predictions[:, -1]
    else:
        predictions = model.predict(tst_X)
    score = scoring(tst_y, predictions)
    return score


def cross_val_score_parallel(model, X, y, cv, scoring, n_jobs, proba=None):
    parallel = Parallel(n_jobs=n_jobs, verbose=False, pre_dispatch="2*n_jobs")
    scores = parallel(
        delayed(validation)(deepcopy(model), X, y, train, test, scoring, proba)
        for train, test in cv.split(X, y)
    )

    return np.mean(scores), model


def cross_val_score_sequential(model, X, y, cv, scoring, proba=None):
    scores = [
        validation(deepcopy(model), X, y, train, test, scoring, proba)
        for train, test in cv.split(X, y)
    ]
    return np.mean(scores), model


def cross_val_score(model, X, y, cv, scoring, n_jobs, proba=None):
    if n_jobs < 2:
        return cross_val_score_sequential(model, X, y, cv, scoring, proba=proba)
    return cross_val_score_parallel(model, X, y, cv, scoring, n_jobs, proba)


def best_alternative_model(
    alternative_pipelines,
    base_score,
    maximize,
    cv,
    metric,
    cv_n_jobs,
    alternative_n_jobs,
    proba,
    X,
    y,
):

    kwargs = dict(cv=cv, scoring=metric, n_jobs=cv_n_jobs, proba=proba)
    if alternative_n_jobs > 1:
        parallel = Parallel(
            n_jobs=alternative_n_jobs, verbose=False, pre_dispatch="2*n_jobs"
        )
        result = parallel(
            delayed(cross_val_score)(pipeline, X, y, **kwargs)
            for pipeline in alternative_pipelines
        )
    else:
        result = [
            cross_val_score(pipeline, X, y, **kwargs)
            for pipeline in alternative_pipelines
        ]

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
    cv=CV,
    metric=None,
    maximize=None,
    proba=None,
    stage_name=STAGE_NAME,
    generator=None,
    recursive=False,
    cv_n_jobs=-1,
    alternative_n_jobs=1,
    **kwargs,
):
    print(format_str("Starting stage {}".format(stage_name)))
    base_score, _ = cross_val_score(
        model_pipeline, X, y, cv=cv, n_jobs=-1, scoring=metric, proba=proba,
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
            proba,
            X,
            y,
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


@duration("Find Best Subset")
def find_best_subset(
    X, y, model, cv=CV, metric=None, maximize=None, proba=None, **kwargs
):
    return generic_change(
        X,
        y,
        model,
        stage_name="Finding best subset",
        generator=feature_reduction_generator,
        recursive=True,
        cv_n_jobs=1,
        alternative_n_jobs=-1,
        cv=cv,
        metric=metric,
        maximize=maximize,
        proba=proba,
        **kwargs,
    )


def get_encoding_generator(feature, encoders, drop=False):
    def categorical_encoding_generator(model):
        for encoder in trange(encoders):
            new_model = deepcopy(model)
            if drop:
                new_model.remove_transformers([feature], partial_match=False)
            new_model = new_model.add_transformators([encoder])
            yield new_model

    return categorical_encoding_generator


def subset_allowed_encoders(encoders, allowed_encoders):
    return [x for x in encoders if x.__class__ in allowed_encoders]


def get_valid_encoders(column, y, classification, allowed_encoders):
    n_unique = column.nunique()
    too_many = ((n_unique / column.size) > MAX_RELATIVE_CARDINALITY) or (
        n_unique > MAX_ABSOLUTE_CARDINALITY
    )
    multiclass = classification and len(set(y)) > 2
    if too_many:
        if str(column.dtypes) == "object":
            encoders = [
                LinearTextEncoder(
                    used_cols=[column.name], multi_class=multiclass
                ),
                TfIdfPCAEncoder(used_cols=[column.name], n_components=20),
                TfIdfEncoder(used_cols=[column.name], vocab_size=20),
            ]
            return subset_allowed_encoders(encoders, allowed_encoders)
        else:
            return []

    if n_unique < 3:
        if column.apply(type).eq(str).any():
            encoders = [CustomLabelEncoder(used_cols=[column.name])]
            return subset_allowed_encoders(encoders, allowed_encoders)
        return []

    encoders = [
        MeanTargetEncoder(used_cols=[column.name], multi_class=multiclass)
    ]
    if n_unique < 10:
        encoders.extend(
            [
                CustomOneHotEncoder(used_cols=[column.name]),
                CustomLabelEncoder(used_cols=[column.name]),
            ]
        )
    return subset_allowed_encoders(encoders, allowed_encoders)


@duration("Encode categoricals")
def encode_categoricals(
    X,
    y,
    model,
    classification,
    allowed_encoders,
    cv=CV,
    metric=None,
    maximize=None,
    proba=None,
    **kwargs,
):
    X_ = model.transform(X)
    categoricals = X_.select_dtypes(object).columns
    del X_
    print("Encoding stage for ", categoricals)
    for feature in trange(categoricals):
        encoders = get_valid_encoders(
            X[feature], y, classification, allowed_encoders
        )
        print(
            f"Encoders for {feature}:", [x.__class__.__name__ for x in encoders]
        )
        if encoders:
            model = generic_change(
                X,
                y,
                model,
                stage_name="Encoding categoricals {}".format(feature),
                generator=get_encoding_generator(feature, encoders),
                cv=cv,
                metric=metric,
                maximize=maximize,
                proba=proba,
                **kwargs,
            )
    return model


@duration("Recode categoricals")
def recode_categoricals(
    X,
    y,
    model,
    classification,
    allowed_encoders,
    max_unique=50,
    cv=CV,
    metric=None,
    maximize=None,
    proba=None,
    **kwargs,
):
    transformed_X = model.transform(X)
    transformed_X = subset_numeric_features(transformed_X)
    used = [c for c in transformed_X if c in model.used_cols]
    transformed_X = transformed_X[used]
    original_used = list(set(X.columns).intersection(set(used)))
    nunique = transformed_X[original_used].nunique()
    categoricals = [f for f in nunique.index if nunique[f] <= max_unique]
    print("Trying to recode following categorical values:", categoricals)
    for feature in trange(categoricals):
        encoders = get_valid_encoders(
            X[feature], y, classification, allowed_encoders
        )
        print(
            f"Encoders for {feature}:", [x.__class__.__name__ for x in encoders]
        )
        if encoders:
            model = generic_change(
                X,
                y,
                model,
                stage_name="Recoding {}".format(feature),
                generator=get_encoding_generator(feature, encoders, drop=True),
                cv=cv,
                metric=metric,
                maximize=maximize,
                proba=proba,
                **kwargs,
            )
    return model


@duration("Feature engineering")
def feature_engineering(
    X,
    y,
    model_path=None,
    subset_stage=True,
    encode_stage=True,
    recode_stage=True,
    allowed_transformers=ALLOWED_TRANSFORMERS,
    max_unique=50,
    estimator=ESTIMATOR,
    cv=CV,
    metric=None,
    classification=None,
    maximize=None,
    proba=None,
    **kwargs,
):
    model = basic_preprocess(X.copy(), y, estimator)
    proba, maximize, metric, classification = get_default_settings(
        proba, maximize, metric, classification, model
    )
    if subset_stage:
        model = find_best_subset(
            X,
            y,
            model,
            cv=cv,
            metric=metric,
            maximize=maximize,
            proba=proba,
            **kwargs,
        )

    if encode_stage:
        model = encode_categoricals(
            X,
            y,
            model,
            classification=classification,
            allowed_encoders=allowed_transformers,
            cv=cv,
            metric=metric,
            maximize=maximize,
            proba=proba,
            **kwargs,
        )
    if recode_stage:
        model = recode_categoricals(
            X,
            y,
            model,
            classification=classification,
            allowed_encoders=allowed_transformers,
            cv=cv,
            metric=metric,
            maximize=maximize,
            proba=proba,
            max_unique=max_unique,
            **kwargs,
        )
    if model_path:
        save_object(model, model_path)

    cols = sorted(list(model.transform(X).columns))
    print("Final features:", cols)

    return model


def model_search(
    X_train,
    y_train,
    model,
    X_test=None,
    y_test=None,
    model_path=None,
    cv=CV,
    scorer=None,
    maximize=None,
    proba=None,
    classification=None,
    **kwargs,
):
    proba, maximize, scorer, classification = get_default_settings(
        proba, maximize, scorer, classification, model
    )
    best = model.find_hyper_params(
        model, X_train, y_train, None, cv, scorer, maximize, proba
    )

    new_model = deepcopy(model)
    print("BEST PARAMS", best.params)
    new_model.model = model.trial.get_model_class(new_model)(**best.params)
    new_model.fit(X_train, y_train)
    if X_test is not None:
        if proba:
            test_preds = new_model.predict_proba(X_test)
        else:
            test_preds = new_model.predict(X_test)
        test_score = scorer(y_test, test_preds)
        print(f"TEST SCORE FOR {scorer.__name__} SCORER is {test_score}")
    if model_path:
        print("Saving model to", model_path)
        save_object(model, model_path)
    return new_model


def auto_ml(
    X,
    y,
    estimator=ESTIMATOR,
    X_test=None,
    y_test=None,
    model_path=None,
    classification=True,
    subset_stage=True,
    encode_stage=True,
    recode_stage=True,
    allowed_transformers=ALLOWED_TRANSFORMERS,
    max_unique=50,
    cv=CV,
    scorer=None,
    maximize=None,
    proba=None,
):
    model = feature_engineering(
        X,
        y,
        model_path=model_path,
        classification=classification,
        subset_stage=subset_stage,
        encode_stage=encode_stage,
        recode_stage=recode_stage,
        allowed_transformers=allowed_transformers,
        max_unique=max_unique,
        estimator=estimator,
        cv=cv,
        metric=scorer,
        maximize=maximize,
        proba=proba,
        # **kwargs,
    )
    model = model_search(
        X,
        y,
        model,
        X_test=X_test,
        y_test=y_test,
        model_path=None,
        cv=cv,
        scorer=scorer,
        maximize=maximize,
        proba=proba,
        classification=classification,
    )
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
        "--features",
        action="store_true",
        default=False,
        help="Find best features",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Optimize final model",
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
    if args.features:
        y = X[target]
        dropped = args.drop
        dropped = dropped if dropped else []
        dropped = dropped + [target]
        X = X.drop(dropped, axis=1)
        model = feature_engineering(X, y, model_path=model_file)
    if args.tune:
        if not args.features:
            y = X[target]
            model = load_object(model_file)
        best_model_file = model_file.replace(".", "_best.")
        model = model_search(X, y, model, X, y, best_model_file)
