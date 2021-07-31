import numpy as np
import pandas as pd
import inspect
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from tqdm import tqdm as tqdm

from barusini.constants import UNIVERSAL_CV, STR_BULLET, STR_SPACE
from barusini.tabular.stages.hyper_parameter_tuning import (
    cv_predictions,
    get_trial_for_model,
)
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
        self.contribs_ = None

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

    def varimp(self, X=None):
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
        df = df.sort_values("Importance", ascending=False)

        if X is not None and self.contribs_ is None:
            self.contrib_imp(X)

        if self.contribs_ is not None:
            df = df.join(self.contribs_)
            df = df.sort_values("Contribs", ascending=False)
        return df

    @staticmethod
    def _supports_pred_cotribs(predic_fn):
        s = inspect.signature(predic_fn)
        return "pred_contrib" in s.parameters

    def get_predict_contrib_fn(self):
        if hasattr(self.model, "predict_proba"):
            fn = self.model.predict_proba
        else:
            fn = self.model.predict
        if self._supports_pred_cotribs(fn):
            return fn

    def contrib_imp(self, X=None):
        if X is None:
            return self.contribs_
        contrib_fn = self.get_predict_contrib_fn()
        if contrib_fn is not None:
            if len(set(self.used_cols) - set(X.columns)):
                X = self.transform(X)
            X = X[self.used_cols]
            contribs = contrib_fn(X, pred_contrib=True)
            if contribs.shape[1] != len(self.used_cols) + 1:
                return None  # Multiclass not implemented yet
            contribs = pd.DataFrame(contribs, columns=self.used_cols + ["bias"])
            fimp = contribs.abs().sum(0).sort_values(ascending=False)
            fimp = pd.DataFrame({"Contribs": fimp})
            fimp["Relative Contribs"] = fimp["Contribs"]
            fimp["Relative Contribs"] /= fimp["Contribs"].drop("bias").max()

            if self.contribs_ is None:
                self.contribs_ = fimp
            return fimp

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
