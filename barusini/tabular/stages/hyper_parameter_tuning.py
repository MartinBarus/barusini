from functools import partial

import numpy as np
import pandas as pd

import optuna
from barusini.utils import deepcopy
from lightgbm import LGBMModel
from xgboost import XGBModel

LOG = "log"
LOGINT = "logint"
INT = "int"
UNIFORM = "uniform"
CATEGORY = "categoty"
ALLOWED_DISTRIBUTIONS = [LOG, LOGINT, INT, UNIFORM, CATEGORY]
CONDITIONAL = "CONDITIONAL_"

ERR = f"Parameter type has to be one of {ALLOWED_DISTRIBUTIONS}, found: "
ERR += "'{}'"


def fn_attr_rec(obj, attributes, fn):
    idx = attributes.find(".")
    if idx >= 0:
        first = attributes[:idx]
        rest = attributes[idx + 1 :]
        if hasattr(obj, first):
            return fn_attr_rec(getattr(obj, first), rest, fn)
        return False
    return fn(obj, attributes)


def hasattr_rec(obj, attributes):
    return fn_attr_rec(obj, attributes, hasattr)


def getattr_rec(obj, attributes):
    return fn_attr_rec(obj, attributes, getattr)


def cv_predictions(
    model,
    X_train,
    y_train,
    X_test,
    cv,
    probability,
    attributes_to_monitor=[],
    print_split_info=False,
    eval_set=False,  # used for xgb/lgbm
    **additional_fit_params,
):
    """

    :param model:
    :param X_train:
    :param y_train:
    :param X_test:
    :param cv:
    :param probability:
    :param attributes_to_monitor: list[str]: list of attributes to retrieve
    from fitted model
    :return:
    """
    if probability:
        size = len(set(y_train))
        oof = np.zeros((len(X_train), size))
    else:
        oof = np.zeros(len(X_train))

    preds = None
    if X_test is not None:
        test_shape = list(deepcopy(oof.shape))
        test_shape[0] = len(X_test)
        preds = np.zeros(test_shape)

    monitored_attributes = {attr: [] for attr in attributes_to_monitor}
    for i, (idxT, idxV) in enumerate(cv.split(X_train, y_train)):
        if print_split_info:
            print(" rows of train =", len(idxT), "rows of holdout =", len(idxV))
        X_act_train = X_train.iloc[idxT]
        y_act_train = y_train.iloc[idxT]

        X_act_val = X_train.iloc[idxV]
        y_act_val = y_train.iloc[idxV]

        if eval_set:
            fit_params = {
                "eval_set": [(X_act_val, y_act_val)],
                **additional_fit_params,
            }
        else:
            fit_params = additional_fit_params

        model.fit(
            X_act_train, y_act_train, **fit_params,
        )

        if probability:
            oof_preds = model.predict_proba(X_act_val)
        else:
            oof_preds = model.predict(X_act_val)
        oof[idxV] += oof_preds
        if X_test is not None:
            if probability:
                preds += model.predict_proba(X_test) / cv.n_splits
            else:
                preds += model.predict(X_test) / cv.n_splits

        for attr in attributes_to_monitor:
            monitored_attributes[attr].append(getattr_rec(model, attr))

    return oof, preds, monitored_attributes


class Parameter:
    def __init__(self, name, param_type, param_args):
        assert param_type in ALLOWED_DISTRIBUTIONS, ERR.format(param_type)
        self.name = name
        self.param_type = param_type
        self.param_args = param_args

    def suggest(self, trial):
        assert self.param_type in ALLOWED_DISTRIBUTIONS, ERR.format(
            self.param_type
        )
        if self.param_type == LOG:
            suggest_function = trial.suggest_loguniform
        elif self.param_type == LOGINT:
            suggest_function = lambda name, low, high: trial.suggest_int(
                name, low, high, log=True
            )
        elif self.param_type == INT:
            suggest_function = trial.suggest_int
        elif self.param_type == UNIFORM:
            suggest_function = trial.suggest_uniform
        elif self.param_type == CATEGORY:
            suggest_function = trial.suggest_categorical
        else:
            raise ValueError(f"Unsupported parameter type {self.param_type}")
        return suggest_function(self.name, *self.param_args)


class Trial:

    default_params = {}
    static_params = {}
    attributes_to_monitor = {}
    additional_fit_params = {}

    def __init__(self):
        self.study = optuna.create_study()
        self.maximize = None

    @staticmethod
    def objective(
        pipeline,
        X_train,
        y_train,
        X_test,
        cv,
        scoring,
        maximize,
        probability,
        original_params,
        static_params,
        additional_fit_params,
        attributes_to_monitor,
        csv_path,
        print_intermediate_results,
        trial,
    ):
        """Perform Hyper-Parameter Search

        :param pipeline: barusini.transformer.Pipeline: pipeline object
        :param X_train: pd.DataFrame: train data
        :param y_train: pd.Series: train label
        :param X_test: optional: pd.DataFrame: test data
        :param cv: sklearn.model_selection._split class
        :param scoring: callable: validation metric
        :param maximize: bool: whether to maximize the metric or minimize
        :param probability: bool: whether to predict probability or class
        :param original_params: dict: parameters to tune
        :param static_params: dict: fixed parameters
        :param additional_fit_params: dict: additional fit parameters
        :param attributes_to_monitor: dict: attributes to report back
        :param csv_path: optional: str: path prefix for storing predictions
        :param print_intermediate_results: bool: more verbose output
        :param trial: optuna.Trial: hyper-parameter tuning trial object
        :return:
        """
        params = {
            name: Parameter(name, param_type, param_args).suggest(trial)
            for name, (param_type, param_args) in original_params.items()
        }
        params = {**params, **static_params}
        new_model = deepcopy(pipeline)
        new_model.model = pipeline.model.__class__(**params)
        if print_intermediate_results:
            print(params)

        monitored_attrs = [
            attr["param"] for attr in attributes_to_monitor.values()
        ]
        oof, preds, monitored_attributes = cv_predictions(
            new_model,
            X_train,
            y_train,
            X_test,
            cv,
            probability,
            monitored_attrs,
            **additional_fit_params,
        )

        # In trees, monitor the best number of trees with early stopping
        for attr, info in attributes_to_monitor.items():
            monitored_vals = monitored_attributes[info["param"]]
            if print_intermediate_results:
                print(monitored_attributes, info["param"])
            default = getattr(new_model.model, attr)
            monitored_vals = [
                default if x is None else x for x in monitored_vals
            ]
            attr_mean = np.mean(monitored_vals)
            attr_std = np.std(monitored_vals)
            attr_value = info["type"](attr_mean)
            trial.suggest_int(attr, attr_value, attr_value)
            if print_intermediate_results:
                print(attr, "mean", attr_mean, "std", attr_std)

        try:
            if len(oof.shape) == 2 and oof.shape[1] == 2:
                oof = oof[:, 1]
            score = scoring(y_train, oof)
        except Exception as e:
            score = float("inf")
            print("Error occured:", e)
        print_score = score
        score = -score if maximize else score
        if print_intermediate_results:
            print("XGB OOF CV =", print_score)
        if csv_path:
            oof_path = csv_path.format(score, "OOF")
            np.savetxt(oof_path, oof)
            test_path = csv_path.format(score, "TEST")
            np.savetxt(test_path, preds)

        return score

    def find_hyper_params(
        self,
        pipeline,
        X_train,
        y_train,
        X_test,
        cv,
        scoring,
        maximize,
        proba,
        params={},
        static_params={},
        additional_fit_params={},
        attributes_to_monitor={},
        csv_path=None,
        n_trials=20,
        n_jobs=1,
        print_intermediate_results=False,
    ):

        objective = partial(
            self.objective,
            pipeline,
            X_train,
            y_train,
            X_test,
            cv,
            scoring,
            maximize,
            proba,
            params,
            static_params,
            additional_fit_params,
            attributes_to_monitor,
            csv_path,
            print_intermediate_results,
        )

        self.maximize = maximize
        self.study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        print(
            f"Out of {n_trials} trials the best score is "
            f"{self.study.best_value} with params {self.study.best_params}"
        )
        return self.study.best_trial

    def table(self):
        results = pd.DataFrame(
            [
                {
                    **x.params,
                    "score": x.value,
                    "duration": str(x.datetime_complete - x.datetime_start),
                }
                for x in self.study.trials
            ]
        )
        if self.maximize:
            results["score"] *= -1
        return results.sort_values("score", ascending=not self.maximize)

    def print_table(self):
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(self.table())


class TreeTrial(Trial):
    additional_fit_params = {
        "eval_set": True,
        "early_stopping_rounds": 20,
    }


class XGBoostTrial(TreeTrial):
    additional_fit_params = {
        "verbose": 0,
    }

    attributes_to_monitor = {
        "n_estimators": {
            "type": round,
            "param": "model._Booster.best_iteration",
            "default": "model.n_estimators",
        }
    }
    static_params = {
        "n_estimators": 1000,
        "tree_method": "hist",
        "seed": 42,
        "n_jobs": 1,
    }
    default_params = {
        "min_child_weight": (LOG, (1e-2, 1e2)),
        "max_depth": (INT, (3, 12)),
        "learning_rate": (LOG, (1e-4, 1e1)),
        "subsample": (UNIFORM, (0.6, 1)),
        "colsample_bytree": (UNIFORM, (0.6, 1)),
    }


class LightGBMTrial(TreeTrial):
    additional_fit_params = {
        "verbose": 0,
    }
    attributes_to_monitor = {
        "n_estimators": {
            "type": round,
            "param": "model.best_iteration_",
            "default": "model.n_estimators",
        }
    }
    static_params = {"n_estimators": 1000, "seed": 42, "n_jobs": 1}
    default_params = {
        "min_child_samples": (LOGINT, (1, 1000)),
        "num_leaves": (LOGINT, (2 ** 3, 2 ** 12)),
        "learning_rate": (LOG, (1e-4, 1e1)),
        "subsample": (UNIFORM, (0.6, 1)),
        "colsample_bytree": (UNIFORM, (0.6, 1)),
    }


def get_trial_for_model(model, **kwargs):
    if isinstance(model, LGBMModel):
        return LightGBMTrial(**kwargs)

    if isinstance(model, XGBModel):
        return XGBoostTrial(**kwargs)

    return Trial(**kwargs)
