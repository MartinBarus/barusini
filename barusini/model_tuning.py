import copy
from functools import partial

import numpy as np

import optuna
from xgboost import XGBClassifier, XGBRegressor

LOG = "log"
INT = "int"
UNIFORM = "uniform"
CATEGORY = "categoty"
ALLOWED_DISTRIBUTIONS = [LOG, INT, UNIFORM, CATEGORY]

ERR = f"Parameter type has to be one of {ALLOWED_DISTRIBUTIONS}, found: "
ERR += "'{}'"


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
    def __init__(
        self,
        # cv,
        # scoring,
        # maximize,
        # proba,
        # parameters,
        # csv_path=None,
        # n_trials=20,
        # n_jobs=1
    ):
        # self.cv = cv
        # self.scoring = scoring
        # self.proba = proba
        # self.parameters = parameters
        # self.csv_path= csv_path
        self.study = optuna.create_study()

    default_params = {}
    default_model_class = None

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
        params,
        csv_path,
        trial,
    ):
        # print("pipeline", pipeline)
        # print("model_class", model_class)
        # print("X_train", X_train)
        # print("y_train", y_train)
        # print("X_test", X_test)
        # print("cv", cv)
        # print("scoring", scoring)
        # print("maximize", maximize)
        # print("probability", probability)
        # print("params", params)
        # print("csv_path", csv_path)
        # print("trial", trial)

        params = {
            name: Parameter(name, param_type, param_args).suggest(trial)
            for name, (param_type, param_args) in params.items()
        }

        new_model = copy.deepcopy(pipeline)
        new_model.model = model_class(**params)
        print(params)

        if probability:
            size = len(set(y_train))
            oof = np.zeros((len(X_train), size))
        else:
            oof = np.zeros(len(X_train))

        if X_test is not None:
            test_shape = copy.deepcopy(oof.shape)
            test_shape[0] = len(X_test)
            preds = np.zeros(test_shape)

        n_iterations = []
        for i, (idxT, idxV) in enumerate(cv.split(X_train, y_train)):
            print(" rows of train =", len(idxT), "rows of holdout =", len(idxV))
            X_act_train = X_train.iloc[idxT]
            y_act_train = y_train.iloc[idxT]

            X_act_val = X_train.iloc[idxV]
            y_act_val = y_train.iloc[idxV]

            new_model.fit(
                X_act_train,
                y_act_train,
                eval_set=[(X_act_val, y_act_val)],
                verbose=0,
                early_stopping_rounds=20,
            )

            if probability:
                oof_preds = new_model.predict_proba(X_act_val)
            else:
                oof_preds = new_model.predict(X_act_val)
            oof[idxV] += oof_preds
            if X_test is not None:
                if probability:
                    preds += new_model.predict_proba(X_test) / cv.n_splits
                else:
                    preds += new_model.predict(X_test) / cv.n_splits

            n_iterations.append(new_model.model.best_iteration)

        if X_test is not None and not probability:
            preds = preds.round().astype(int)

        iter_mean = np.mean(n_iterations)
        iter_std = np.std(n_iterations)
        n_estimators = int(np.round(iter_mean))
        trial.suggest_int("n_estimators", n_estimators, n_estimators)
        score = scoring(y_train, oof)
        score = -score if maximize else score
        print("XGB OOF CV=", score, "mean", iter_mean, "std", iter_std)
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
        csv_path=None,
        n_trials=20,
        n_jobs=1,
    ):
        if not params:
            params = self.default_params

        if not model_class:
            model_class = self.default_model_class

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
            csv_path,
        )

        self.study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        print(
            f"Out of {n_trials} trials the best score is "
            f"{self.study.best_value} with params {self.study.best_params}"
        )
        return self.study.best_trial


class XGBoostTrial(Trial):
    default_params = {
        "min_child_weight": (LOG, (1e-1, 1e3)),
        "max_depth": (INT, (3, 12)),
        "learning_rate": (CATEGORY, ([0.01],)),
        "subsample": (UNIFORM, (0.6, 1)),
        "colsample_bytree": (UNIFORM, (0.6, 1)),
        "tree_method": (CATEGORY, (["hist"],)),
        "seed": (CATEGORY, ([42],)),
    }


class XGBoostRegression(XGBoostTrial):
    default_model_class = XGBRegressor


class XGBoostClassification(XGBoostTrial):
    default_model_class = XGBClassifier
