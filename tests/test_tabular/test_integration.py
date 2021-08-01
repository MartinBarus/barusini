import copy
import os

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from barusini.tabular import feature_engineering, model_search
from barusini.tabular.stages.hyper_parameter_tuning import LOG, LOGINT, UNIFORM
from barusini.tabular.transformers import (
    CustomLabelEncoder,
    CustomOneHotEncoder,
    Identity,
    MeanTargetEncoder,
    Pipeline,
)
from barusini.tabular.transformers.ensemble import Ensemble
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

N_ESTIMATORS = 2

XGB_PARAMS = {
    "n_estimators": N_ESTIMATORS,
    "use_label_encoder": False,
}

LGB_PARAMS = {
    "n_estimators": N_ESTIMATORS,
}


@pytest.fixture(scope="session")
def salary_data():
    def transform_target(value):
        if ">" in value:
            return 1
        return 0

    train_path = "integration_train.csv"
    test_path = "integration_test.csv"

    if os.path.exists(train_path) and os.path.exists(test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(train_path)
        return train, test

    train_data = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/adult/adult.data"
    )
    test_data = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/adult/adult.test"
    )

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "target",
    ]

    train = pd.read_csv(train_data, header=None, names=columns)
    test = pd.read_csv(test_data, header=None, names=columns, skiprows=1)

    train["target"] = train["target"].apply(transform_target)
    test["target"] = test["target"].apply(transform_target)

    encoder = LabelEncoder()
    train["relationship"] = encoder.fit_transform(train["relationship"])
    test["relationship"] = encoder.transform(test["relationship"])

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train, test


def run_integration(
    data, target, models_list, classification, metric, cv, proba
):
    train, test = data
    models_dict = {
        model: feature_engineering(
            train.drop(columns=[target]),
            train[target],
            estimator=copy.deepcopy(model),
            cv=cv,
            classification=classification,
            metric=metric,
        )
        for model in models_list
    }

    ensemble = Ensemble(list(models_dict.values()))
    ensemble.fit(train, train[target])
    models_dict["Ensemble"] = ensemble
    if proba:
        test_scores = {
            key: metric(test[target], model.predict_proba(test))
            for key, model in models_dict.items()
        }
    else:
        test_scores = {
            key: metric(test[target], model.predict(test))
            for key, model in models_dict.items()
        }
    print(ensemble)
    print(metric.__name__)
    print(pd.Series(test_scores).sort_values())


def test_binary_classification(salary_data):
    run_integration(
        salary_data,
        "target",
        [
            LGBMClassifier(**LGB_PARAMS),
            XGBClassifier(eval_metric="logloss", **XGB_PARAMS),
        ],
        True,
        log_loss,
        StratifiedKFold(n_splits=3, random_state=42, shuffle=True),
        True,
    )


def test_multiclass_classification(salary_data):
    run_integration(
        salary_data,
        "relationship",
        [
            LGBMClassifier(**LGB_PARAMS),
            XGBClassifier(eval_metric="mlogloss", **XGB_PARAMS),
        ],
        True,
        accuracy_score,
        StratifiedKFold(n_splits=3, random_state=42, shuffle=True),
        False,
    )


def test_regression(salary_data):
    run_integration(
        salary_data,
        "age",
        [LGBMRegressor(**LGB_PARAMS), XGBRegressor(**XGB_PARAMS)],
        False,
        mean_absolute_error,
        KFold(n_splits=3, random_state=42, shuffle=True),
        False,
    )


def test_hyper_parameter_tuning(salary_data):
    train, test = salary_data
    target = "target"

    model = Pipeline(
        transformers=[
            Identity(used_cols=["fnlwgt"]),
            Identity(used_cols=["education-num"]),
            Identity(used_cols=["relationship"]),
            MeanTargetEncoder(used_cols=["occupation"]),
            CustomLabelEncoder(used_cols=["sex"]),
            CustomOneHotEncoder(used_cols=["race"]),
        ],
        model=LGBMClassifier(),
    )

    best_model = model_search(
        train.drop(columns=[target]),
        train[target],
        model,
        X_test=test.drop(columns=[target]),
        y_test=test[target],
        cv=StratifiedKFold(n_splits=3, random_state=42, shuffle=True),
        classification=True,
        scorer=log_loss,
        proba=True,
        maximize=False,
        n_trials=8,
        # fixed parameters
        static_params={"n_estimators": 100, "seed": 42, "n_jobs": 1},
        # hyper parameters and tuning space
        params={
            "min_child_samples": (LOGINT, (1, 1000)),
            "num_leaves": (LOGINT, (2 ** 3, 2 ** 12)),
            "learning_rate": (LOG, (1e-4, 1e1)),
            "subsample": (UNIFORM, (0.6, 1)),
            "colsample_bytree": (UNIFORM, (0.6, 1)),
        },
        # additional parameters passed to fit (f.e. "eval_set=True")
        additional_fit_params={"verbose": 0},
        # used only with early stopping to monitor number of iterations
        attributes_to_monitor={},
    )
    print(best_model)
