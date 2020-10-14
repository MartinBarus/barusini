import copy
import pandas as pd

from barusini.constants import CV, STR_BULLET, STR_SPACE
from barusini.model_tuning import cv_predictions


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
        orig_transformers = [copy.deepcopy(x) for x in self.transformers]
        transformers = orig_transformers + transformers
        return Pipeline(transformers, copy.deepcopy(self.model))

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


class Ensemble(Transformer):
    def __init__(self, pipelines, meta, cv=CV):
        self.pipelines = pipelines
        self.meta = meta
        self.cv = cv

    def fit(self, X, y, *args, **kwargs):
        oof_predictions = []
        for pipeline in self.pipelines:
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
        return self.meta.predict_proba(X)

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
