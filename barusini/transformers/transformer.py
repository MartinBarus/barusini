import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from tqdm import tqdm as tqdm

from barusini.constants import CV, STR_BULLET, STR_SPACE
from barusini.model_tuning import cv_predictions, get_trial_for_model
from barusini.utils import deepcopy, get_maximize, get_probability


class Transformer:
    def __init__(self, used_cols=None):
        self.used_cols = used_cols

    def fit(self, X, *args, **kwargs):
        raise ValueError("Fit method is not implemented")

    def transform(self, X, **kwargs):
        raise ValueError("Transform method is not implemented")

    def fit_transform(self, X, *args, **kwargs):
        self.fit(X, *args, **kwargs)
        return self.transform(X, **kwargs)

    def output_columns(self):
        if self.used_cols is not None:
            return self.used_cols
        return []


class Pipeline(Transformer):
    def __init__(self, transformers, model):
        super().__init__()
        self.transformers = transformers
        self.model = model
        self.target = None
        self.trial = None

    def fit(self, X, y, **kwargs):
        X_transformed = self.fit_transform(X, y)
        X_transformed = X_transformed[self.used_cols]
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            eval_set = [
                (self.transform(x)[self.used_cols], y) for x, y in eval_set
            ]
            kwargs["eval_set"] = eval_set

        self.model.fit(X_transformed, y, **kwargs)
        self.target = y.name
        return self

    def transform(self, X, **kwargs):
        for transformer in self.transformers:
            X = transformer.transform(X, **kwargs)
        return X

    def fit_transform(self, X, y, **kwargs):
        used_cols = []
        for transformer in self.transformers:
            X = transformer.fit_transform(X, y, **kwargs)
            used_cols.extend(transformer.output_columns())

        act_cols = set(X.columns)
        used_cols = set(used_cols).intersection(act_cols)  # remove intermediate

        self.used_cols = sorted(list(used_cols))
        return X

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.model.predict(X_transformed[self.used_cols])

    def predict_proba(self, X):
        X_transformed = self.transform(X)
        return self.model.predict_proba(X_transformed[self.used_cols])

    def add_transformators(self, transformers):
        orig_transformers = [deepcopy(x) for x in self.transformers]
        transformers = orig_transformers + transformers
        return Pipeline(transformers, deepcopy(self.model))

    @staticmethod
    def _match_name(transformer, columns, partial_match):
        match = all([c in transformer.used_cols for c in columns])
        if partial_match:
            return match
        return match and len(columns) == len(transformer.used_cols)

    def remove_transformers(self, columns, partial_match=False):

        self.transformers = [
            x
            for x in self.transformers
            if not self._match_name(x, columns, partial_match)
        ]

    def varimp(self):
        importance = None
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = self.model.coef_
        else:
            raise ValueError(f"varimp not implemented for {type(self.model)}")

        df = pd.DataFrame({"Feature": self.used_cols, "Importance": importance})
        df = df.set_index("Feature")
        df["Importance"] /= df["Importance"].max()
        return df.sort_values("Importance", ascending=False)

    def __str__(self):
        str_representation = (
            f"Pipeline ({len(self.transformers)} Transformers):\n"
        )
        for transformer in self.transformers:
            transformer_str = str(transformer).replace("\n", f"\n{STR_SPACE}")
            str_representation += f"{STR_BULLET}{transformer_str}\n"

        str_representation += f"{STR_BULLET}{str(self.model)}\n"
        return str_representation

    def tune(
        self,
        X,
        y,
        cv,
        score,
        probability=None,
        maximize=None,
        params=None,
        static_params=None,
        additional_fit_params=None,
        attributes_to_monitor=None,
        n_trials=20,
        n_jobs=4,
    ):
        if probability is None:
            probability = get_probability(score)

        if maximize is None:
            maximize = get_maximize(score)

        if self.trial is None:
            self.trial = get_trial_for_model(self.model)

        if params is None:
            params = self.trial.default_params

        if static_params is None:
            static_params = self.trial.static_params

        if additional_fit_params is None:
            additional_fit_params = self.trial.additional_fit_params

        if attributes_to_monitor is None:
            attributes_to_monitor = self.trial.attributes_to_monitor

        best = self.trial.find_hyper_params(
            self,
            X_train=X,
            y_train=y,
            X_test=None,
            cv=cv,
            scoring=score,
            maximize=maximize,
            proba=probability,
            params=params,
            static_params=static_params,
            additional_fit_params=additional_fit_params,
            attributes_to_monitor=attributes_to_monitor,
            csv_path=None,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
        used_static_params = {
            k: v
            for k, v in static_params.items()
            if k not in attributes_to_monitor
        }
        self.model = self.model.__class__(**best.params, **used_static_params)
        self.fit(X, y)
        return best


class Ensemble(Transformer):
    def __init__(self, pipelines, meta, cv=CV):
        self.pipelines = pipelines
        self.meta = meta
        self.cv = cv

    def fit(self, X, y, *args, **kwargs):
        oof_predictions = []
        for pipeline in tqdm(self.pipelines):
            proba = hasattr(pipeline.model, "predict_proba")
            oof, _, _ = cv_predictions(pipeline, X, y, None, self.cv, proba)
            oof_predictions.append(oof)
            pipeline.fit(X, y, *args, **kwargs)

        train_X = pd.DataFrame(oof_predictions).T
        self.meta.fit(train_X, y)

    def _get_base_predictions(self, X):
        predictions = []
        for pipeline in self.pipelines:
            proba = hasattr(pipeline.model, "predict_proba")
            if proba:
                preds = pipeline.predict_proba(X)
            else:
                preds = pipeline.predict(X)
            predictions.append(preds)
        return pd.DataFrame(predictions).T

    def predict(self, X):
        X = self._get_base_predictions(X)
        return self.meta.predict(X)

    def predict_proba(self, X):
        X = self._get_base_predictions(X)
        if hasattr(self.meta, "predict_proba"):
            return self.meta.predict_proba(X)
        return self.meta.predict(X)

    def __str__(self):
        str_representation = (
            f"Ensemble,  ({str(self.meta.__class__.__name__)}, "
            f"{len(self.pipelines)} Pipelines):\n"
        )
        for pipeline in self.pipelines:
            pipeline_str = str(pipeline).replace("\n", f"\n{STR_SPACE}")
            str_representation += f"{STR_BULLET}{pipeline_str}\n"

        str_representation += f"{STR_BULLET}{str(self.meta)}\n"
        return str_representation


class WeightedAverage:
    def __init__(self, min_weight=0.01, method="SLSQP", tol=1e-6):
        self.weights = None
        self.res = None
        self.min_weight = min_weight
        self.method = method
        self.tol = tol

    @staticmethod
    def _get_optimization_fn(X, y):
        def fn_to_minimize(x):
            return ((X.dot(x) - y) ** 2).mean()

        n_models = X.shape[1]
        start = np.ones(n_models) / n_models
        constraints = (
            LinearConstraint(np.identity(n_models), 0, 1),
            LinearConstraint(np.ones(n_models), 1, 1),
        )

        return fn_to_minimize, start, constraints

    def _solve(self, X, y, zero_weight_idxs):
        significant_models = [
            i for i in range(X.shape[1]) if i not in zero_weight_idxs
        ]
        fn, start, constraints = self._get_optimization_fn(
            X.iloc[:, significant_models], y
        )
        self.res = minimize(
            fn, start, method=self.method, tol=self.tol, constraints=constraints
        )
        weights = np.zeros(X.shape[1])
        for i, weight in enumerate(self.res.x):
            weights[significant_models[i]] = weight

        return weights

    def _get_zero_weight_idxs(self, weights):
        idx = -1
        min_val = 1
        for i, w in enumerate(weights):
            if w < self.min_weight and w < min_val and w != 0:
                min_val, idx = w, i

        if idx > -1:
            return [idx]
        return []

    def fit(self, X, y):
        assert self.min_weight < 1
        zero_weight_idxs = []
        weights = self._solve(X, y, zero_weight_idxs)
        progress = tqdm(total=X.shape[1])
        while self.min_weight and len(np.nonzero(weights)[0]) > 1:
            new_zero_weights = self._get_zero_weight_idxs(weights)
            if not len(new_zero_weights):
                break

            zero_weight_idxs.extend(new_zero_weights)
            weights = self._solve(X, y, zero_weight_idxs)
            progress.update(1)
        progress.update(progress.total - progress.n)
        progress.close()
        self.weights = weights

    def predict(self, X):
        return X.dot(self.weights)
