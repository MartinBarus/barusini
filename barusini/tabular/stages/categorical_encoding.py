from copy import deepcopy

from barusini.constants import CV
from barusini.tabular.stages.base_stage import (
    generic_change,
    get_valid_encoders,
    subset_numeric_features,
)
from barusini.utils import duration, trange


def get_encoding_generator(feature, encoders, drop=False):
    def categorical_encoding_generator(model):
        for encoder in trange(encoders):
            new_model = deepcopy(model)
            if drop:
                new_model.remove_transformers([feature], partial_match=False)
            new_model = new_model.add_transformators([encoder])
            yield new_model

    return categorical_encoding_generator


@duration("Encode categoricals")
def encode_categoricals(
    X,
    y,
    model,
    classification,
    allowed_encoders,
    cv=CV,
    metric=None,
    maximize=None,
    proba=None,
    **kwargs,
):
    X_ = model.transform(X)
    categoricals = X_.select_dtypes(object).columns
    del X_
    print("Encoding stage for ", categoricals)
    for feature in trange(categoricals):
        encoders = get_valid_encoders(
            X[feature], y, classification, allowed_encoders
        )
        print(
            f"Encoders for {feature}:", [x.__class__.__name__ for x in encoders]
        )
        if encoders:
            model = generic_change(
                X,
                y,
                model,
                stage_name="Encoding categoricals {}".format(feature),
                generator=get_encoding_generator(feature, encoders),
                cv=cv,
                metric=metric,
                maximize=maximize,
                proba=proba,
                **kwargs,
            )
    return model


@duration("Recode categoricals")
def recode_categoricals(
    X,
    y,
    model,
    classification,
    allowed_encoders,
    max_unique=50,
    cv=CV,
    metric=None,
    maximize=None,
    proba=None,
    **kwargs,
):
    transformed_X = model.transform(X)
    transformed_X = subset_numeric_features(transformed_X)
    used = [c for c in transformed_X if c in model.used_cols]
    transformed_X = transformed_X[used]
    original_used = list(set(X.columns).intersection(set(used)))
    nunique = transformed_X[original_used].nunique()
    categoricals = [f for f in nunique.index if nunique[f] <= max_unique]
    print("Trying to recode following categorical values:", categoricals)
    for feature in trange(categoricals):
        encoders = get_valid_encoders(
            X[feature], y, classification, allowed_encoders
        )
        print(
            f"Encoders for {feature}:", [x.__class__.__name__ for x in encoders]
        )
        if encoders:
            model = generic_change(
                X,
                y,
                model,
                stage_name="Recoding {}".format(feature),
                generator=get_encoding_generator(feature, encoders, drop=True),
                cv=cv,
                metric=metric,
                maximize=maximize,
                proba=proba,
                **kwargs,
            )
    return model
