from joblib import Parallel, delayed

from barusini.constants import (
    CV,
    MAX_ABSOLUTE_CARDINALITY,
    MAX_RELATIVE_CARDINALITY,
    STAGE_NAME,
    TERMINAL_COLS,
)
from barusini.model_selection import cross_val_score
from barusini.transformers import (
    CustomLabelEncoder,
    CustomOneHotEncoder,
    LinearTextEncoder,
    MeanTargetEncoder,
    TfIdfEncoder,
    TfIdfPCAEncoder,
)

ALLOWED_TRANSFORMERS = (
    CustomLabelEncoder,
    CustomLabelEncoder,
    MeanTargetEncoder,
    TfIdfPCAEncoder,
    TfIdfEncoder,
    LinearTextEncoder,
)


def format_str(x, total_len=TERMINAL_COLS):
    middle = total_len // 2
    num_paddings = middle - len(x) // 2 - 1
    padding = "-" * num_paddings
    result = "{} {} {}".format(padding, x, padding)
    return result


def subset_allowed_encoders(encoders, allowed_encoders):
    return [x for x in encoders if x.__class__ in allowed_encoders]


def get_valid_encoders(column, y, classification, allowed_encoders):
    n_unique = column.nunique()
    too_many = ((n_unique / column.size) > MAX_RELATIVE_CARDINALITY) or (
        n_unique > MAX_ABSOLUTE_CARDINALITY
    )
    multiclass = classification and len(set(y)) > 2
    if too_many:
        if str(column.dtypes) == "object":
            encoders = [
                LinearTextEncoder(
                    used_cols=[column.name], multi_class=multiclass
                ),
                TfIdfPCAEncoder(used_cols=[column.name], n_components=20),
                TfIdfEncoder(used_cols=[column.name], vocab_size=20),
            ]
            return subset_allowed_encoders(encoders, allowed_encoders)
        else:
            return []

    if n_unique < 3:
        if column.apply(type).eq(str).any():
            encoders = [CustomLabelEncoder(used_cols=[column.name])]
            return subset_allowed_encoders(encoders, allowed_encoders)
        return []

    encoders = [
        MeanTargetEncoder(used_cols=[column.name], multi_class=multiclass)
    ]
    if n_unique < 10:
        encoders.extend(
            [
                CustomOneHotEncoder(used_cols=[column.name]),
                CustomLabelEncoder(used_cols=[column.name]),
            ]
        )
    return subset_allowed_encoders(encoders, allowed_encoders)


def subset_numeric_features(X):
    ignored_columns = X.select_dtypes(object).columns
    numeric_columns = [col for col in X if col not in ignored_columns]
    X = X[numeric_columns]
    return X


def is_new_better(old, new, maximize):
    if maximize:
        return old <= new
    return new <= old


def best_alternative_model(
    alternative_pipelines,
    base_score,
    maximize,
    cv,
    metric,
    cv_n_jobs,
    alternative_n_jobs,
    proba,
    X,
    y,
):

    kwargs = dict(cv=cv, scoring=metric, n_jobs=cv_n_jobs, proba=proba)
    if alternative_n_jobs not in [0, 1]:
        parallel = Parallel(
            n_jobs=alternative_n_jobs, verbose=False, pre_dispatch="2*n_jobs"
        )
        result = parallel(
            delayed(cross_val_score)(pipeline, X, y, **kwargs)
            for pipeline in alternative_pipelines
        )
    else:
        result = [
            cross_val_score(pipeline, X, y, **kwargs)
            for pipeline in alternative_pipelines
        ]

    best_pipeline = None
    for act_score, act_pipeline in result:
        if is_new_better(base_score, act_score, maximize):
            base_score = act_score
            best_pipeline = act_pipeline

    return best_pipeline, base_score


def generic_change(
    X,
    y,
    model_pipeline,
    cv=CV,
    metric=None,
    maximize=None,
    proba=None,
    stage_name=STAGE_NAME,
    generator=None,
    recursive=False,
    cv_n_jobs=-1,
    alternative_n_jobs=1,
    **kwargs,
):
    print(format_str("Starting stage {}".format(stage_name)))
    base_score, _ = cross_val_score(
        model_pipeline, X, y, cv=cv, n_jobs=-1, scoring=metric, proba=proba,
    )
    print("BASE", base_score)
    original_best = base_score
    old_cols = model_pipeline.transform(X)[model_pipeline.used_cols].columns
    while True:
        best_pipeline, base_score = best_alternative_model(
            generator(model_pipeline),
            base_score,
            maximize,
            cv,
            metric,
            cv_n_jobs,
            alternative_n_jobs,
            proba,
            X,
            y,
        )
        if best_pipeline is not None:
            model_pipeline = best_pipeline
            model_pipeline.fit(X, y)
            print("CURRENT BEST", base_score)
        else:
            break
        if not recursive:
            break

    new_cols = model_pipeline.transform(X)[model_pipeline.used_cols].columns
    print("ORIGINAL BEST", original_best)
    print("NEW BEST", base_score)
    print("DIFF", abs(base_score - original_best))
    print("Dropped", [x for x in old_cols if x not in new_cols])
    # print("Left", [x for x in old_cols if x in new_cols])
    print("New", [x for x in new_cols if x not in old_cols])
    print(format_str("Stage {} finished".format(stage_name)))
    return model_pipeline
