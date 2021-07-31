###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################
import pandas as pd

from barusini.constants import CV, ESTIMATOR
from barusini.tabular import (
    ALLOWED_TRANSFORMERS,
    basic_preprocess,
    encode_categoricals,
    find_best_imputation,
    find_best_subset,
    recode_categoricals,
)
from barusini.utils import (
    deepcopy,
    duration,
    get_default_settings,
    load_object,
    save_object,
)


@duration("Feature engineering")
def feature_engineering(
    X,
    y,
    model_path=None,
    imputation_stage=False,
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
    if imputation_stage:
        model = find_best_imputation(
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
    n_trials=20,
    **kwargs,
):
    proba, maximize, scorer, classification = get_default_settings(
        proba, maximize, scorer, classification, model
    )
    best = model.tune(
        X_train,
        y_train,
        cv,
        scorer,
        maximize=maximize,
        probability=proba,
        n_trials=n_trials,
        **kwargs,
    )

    print("BEST PARAMS", best.params)
    if X_test is not None:
        if proba:
            test_preds = model.predict_proba(X_test)
        else:
            test_preds = model.predict(X_test)
        test_score = scorer(y_test, test_preds)
        print(f"TEST SCORE FOR {scorer.__name__} SCORER is {test_score}")
    if model_path:
        print("Saving model to", model_path)
        save_object(model, model_path)
    return model


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
