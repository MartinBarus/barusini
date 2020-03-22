import copy


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

    def __str__(self):
        str_representation = ""
        for transformer in self.transformers:
            str_representation += f"{str(transformer)}\n"

        str_representation += f"{str(self.model)}"
        return str_representation
