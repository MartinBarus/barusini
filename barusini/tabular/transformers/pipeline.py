import inspect

import pandas as pd

from barusini.constants import STR_BULLET, STR_SPACE
from barusini.tabular.stages.hyper_parameter_tuning import get_trial_for_model
from barusini.tabular.transformers.base_transformer import Transformer
from barusini.utils import deepcopy, get_maximize, get_probability


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
