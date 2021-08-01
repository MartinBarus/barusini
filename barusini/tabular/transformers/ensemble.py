import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize
from tqdm import tqdm as tqdm

from barusini.constants import STR_BULLET, STR_SPACE, UNIVERSAL_CV
from barusini.tabular.stages.hyper_parameter_tuning import cv_predictions
from barusini.tabular.transformers.base_transformer import Transformer


class Ensemble(Transformer):
    def __init__(self, pipelines, meta=None, cv=UNIVERSAL_CV):
        self.pipelines = pipelines
        self.meta = meta
        self.cv = cv

    def fit(self, X, y, *args, **kwargs):
        shapes = set()
        oof_predictions = []
        for pipeline in tqdm(self.pipelines):
            proba = hasattr(pipeline.model, "predict_proba")
            oof, _, _ = cv_predictions(pipeline, X, y, None, self.cv, proba)
            if len(oof.shape) == 1:
                oof = oof.reshape(-1, 1)
            shapes.add(oof.shape[1])
            oof_predictions.append(oof)
            pipeline.fit(X, y, *args, **kwargs)

        assert len(shapes) == 1, "All models must have same output shape!"
        train_X = pd.DataFrame(np.hstack(oof_predictions))
        if self.meta is None:
            self.meta = WeightedAverage(num_classes=shapes.pop())
        self.meta.fit(train_X, y)

    def _get_base_predictions(self, X):
        predictions = []
        for pipeline in self.pipelines:
            proba = hasattr(pipeline.model, "predict_proba")
            if proba:
                preds = pipeline.predict_proba(X)
            else:
                preds = pipeline.predict(X)
            if len(preds.shape) == 1:
                preds = preds.reshape(-1, 1)
            predictions.append(preds)
        return pd.DataFrame(np.hstack(predictions))

    def predict(self, X):
        X = self._get_base_predictions(X)
        predictions = self.meta.predict(X)

        if len(predictions.shape) == 2:
            return predictions.argmax(axis=1)

        return predictions

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
    def __init__(
        self, num_classes=1, min_weight=0.01, method="SLSQP", tol=1e-6
    ):
        self.weights = None
        self.res = None
        self.min_weight = min_weight
        self.method = method
        self.tol = tol
        self.num_classes = num_classes

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

    def process_target(self, y):
        return np.hstack([1 * (y.values == i) for i in range(self.num_classes)])

    def process_data(self, X):
        new_X = []
        num_models = X.shape[1] // self.num_classes
        for i_class in range(self.num_classes):
            act_class_cols = [
                i_model * self.num_classes + i_class
                for i_model in range(num_models)
            ]
            new_X.extend(X.iloc[:, act_class_cols].values)

        return pd.DataFrame(new_X)

    def inverse_predictions(self, y):
        size = y.shape[0] // self.num_classes
        return np.vstack(
            [y[i * size : (i + 1) * size] for i in range(self.num_classes)]
        ).T

    def fit(self, X, y):
        assert self.min_weight < 1
        if self.num_classes > 1:
            X = self.process_data(X)
            y = self.process_target(y)

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
        if self.num_classes > 1:
            X = self.process_data(X)

        y = X.dot(self.weights)

        if self.num_classes > 1:
            y = self.inverse_predictions(y)

        return y

    def __str__(self):
        return f"Weighted average: {self.weights}"
