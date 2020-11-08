import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, KFold

from barusini.features import feature_engineering
from barusini.transformers.transformer import Ensemble
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


@pytest.fixture(scope="session")
def salary_data():
    def transform_target(value):
        if ">" in value:
            return 1
        return 0

    train_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

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

    train = pd.read_csv(train_path, header=None, names=columns)
    test = pd.read_csv(test_path, header=None, names=columns, skiprows=1)
    train["target"] = train["target"].apply(transform_target)
    test["target"] = test["target"].apply(transform_target)
    return train, test


def run_integration(
    data, target, models_list, classification, metric, cv, proba
):
    train, test = data
    models_dict = {
        Class.__name__: feature_engineering(
            train.drop(columns=[target]),
            train[target],
            estimator=Class(),
            cv=cv,
            classification=classification,
            metric=metric,
            n_estimators=1,
        )
        for Class in models_list
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
    print(pd.Series(test_scores).sort_values())
    print(ensemble)


def test_binary_classification(salary_data):
    run_integration(
        salary_data,
        "target",
        [LGBMClassifier, XGBClassifier],
        True,
        log_loss,
        StratifiedKFold(n_splits=3, random_state=42, shuffle=True),
        True,
    )


def test_multiclass_classification(salary_data):
    run_integration(
        salary_data,
        "relationship",
        [LGBMClassifier, XGBClassifier],
        True,
        accuracy_score,
        StratifiedKFold(n_splits=3, random_state=42, shuffle=True),
        False,
    )


def test_regression(salary_data):
    run_integration(
        salary_data,
        "age",
        [LGBMRegressor, XGBRegressor],
        False,
        mean_absolute_error,
        KFold(n_splits=3, random_state=42, shuffle=True),
        False,
    )
