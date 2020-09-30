import copy
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

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
    model, X_train, y_train, X_test, cv, probability, attributes_to_monitor=[]
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
        test_shape = copy.deepcopy(oof.shape)
        test_shape[0] = len(X_test)
        preds = np.zeros(test_shape)

    monitored_attributes = {attr: [] for attr in attributes_to_monitor}
    for i, (idxT, idxV) in enumerate(cv.split(X_train, y_train)):
        print(" rows of train =", len(idxT), "rows of holdout =", len(idxV))
        X_act_train = X_train.iloc[idxT]
        y_act_train = y_train.iloc[idxT]

        X_act_val = X_train.iloc[idxV]
        y_act_val = y_train.iloc[idxV]

        model.fit(
            X_act_train,
            y_act_train,
            eval_set=[(X_act_val, y_act_val)],
            verbose=0,
            early_stopping_rounds=20,
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

    if X_test is not None and not probability:
        preds = preds.round().astype(int)

    return oof, preds, monitored_attributes


class Parameter:
    def __init__(self, name, param_type, param_args):
        assert param_type in ALLOWED_DISTRIBUTIONS, ERR.format(param_type)
        self.name = name
        self.param_type = param_type
        self.param_args = param_args

    def suggest(self, trial):
        # print(self.name, self.param_type, self.param_args)
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

    def __init__(self):
        self.study = optuna.create_study()
        self.maximize = None

    @staticmethod
    def objective(
        pipeline,
        model_class,
        X_train,
        y_train,
        X_test,
        cv,
        scoring,
        maximize,
        probability,
        original_params,
        static_params,
        attributes_to_monitor,
        csv_path,
        trial,
    ):
        params = {
            name: Parameter(name, param_type, param_args).suggest(trial)
            for name, (param_type, param_args) in original_params.items()
        }
        params = {**params, **static_params}

        new_model = copy.deepcopy(pipeline)
        new_model.model = model_class(**params)
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
        )

        # In trees, monitor the best number of trees with early stopping
        for attr, info in attributes_to_monitor.items():
            attr_mean = np.mean(monitored_attributes[info["param"]])
            attr_std = np.std(monitored_attributes[info["param"]])
            attr_value = info["type"](attr_mean)
            trial.suggest_int(attr, attr_value, attr_value)
            print(attr, "mean", attr_mean, "std", attr_std)

        score = scoring(y_train, oof)
        print_score = score
        score = -score if maximize else score
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
        model_class=None,
        params=None,
        static_params=None,
        attributes_to_monitor=None,
        csv_path=None,
        n_trials=20,
        n_jobs=1,
    ):
        if not params:
            params = self.default_params

        if not static_params:
            static_params = self.static_params

        if not attributes_to_monitor:
            attributes_to_monitor = self.attributes_to_monitor

        if not model_class:
            model_class = self.get_model_class(pipeline)

        objective = partial(
            self.objective,
            pipeline,
            model_class,
            X_train,
            y_train,
            X_test,
            cv,
            scoring,
            maximize,
            proba,
            params,
            static_params,
            attributes_to_monitor,
            csv_path,
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

    @staticmethod
    def is_classification(model):
        if issubclass(model.__class__, bool):
            return model
        if issubclass(model.__class__, ClassifierMixin):
            return True
        if issubclass(model.__class__, RegressorMixin):
            return False
        if hasattr(model, "model"):
            return Trial.is_classification(model.model)
        else:
            raise ValueError(
                "Model is not subclass of neither "
                "ClassifierMixin nor RegressorMixin"
            )

    @staticmethod
    def get_model_class(pipeline):
        pass


class TreeTrial(Trial):
    pass


class XGBoostTrial(TreeTrial):
    attributes_to_monitor = {
        "n_estimators": {"type": round, "param": "model.best_iteration"}
    }
    static_params = {"n_estimators": 10000, "tree_method": "hist", "seed": 42}
    default_params = {
        "min_child_weight": (LOG, (1e-2, 1e2)),
        "max_depth": (INT, (3, 12)),
        "learning_rate": (LOG, (1e-4, 1e1)),
        "subsample": (UNIFORM, (0.6, 1)),
        "colsample_bytree": (UNIFORM, (0.6, 1)),
    }

    @staticmethod
    def get_model_class(pipeline):
        if Trial.is_classification(pipeline):
            return XGBClassifier
        return XGBRegressor


class LightGBMTrial(TreeTrial):
    attributes_to_monitor = {
        "n_estimators": {"type": round, "param": "model.best_iteration_"}
    }
    static_params = {"n_estimators": 10000, "seed": 42}
    default_params = {
        "min_child_samples": (LOGINT, (1, 1000)),
        "num_leaves": (LOGINT, (2 ** 3, 2 ** 12)),
        "learning_rate": (LOG, (1e-4, 1e1)),
        "subsample": (UNIFORM, (0.6, 1)),
        "colsample_bytree": (UNIFORM, (0.6, 1)),
    }

    @staticmethod
    def get_model_class(pipeline):
        if Trial.is_classification(pipeline):
            return LGBMClassifier
        return LGBMRegressor
