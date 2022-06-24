import copy
import functools
import pickle
import re
import time
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from tqdm import tqdm as tqdm

from barusini.constants import (
    DEFAULT_CASSIFICATION_METRIC,
    DEFAULT_REGRESSION_METRIC,
    JOIN_STR,
    METRIC_DICT,
)


def save_object(o, path):
    with open(path, "wb") as file:
        pickle.dump(o, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def deepcopy(obj):
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    return copy.deepcopy(obj)


def _format_time_helper(elapsed_time, unit, next_unit, max_value):
    unit_changed = False
    if elapsed_time > max_value:
        elapsed_time /= max_value
        unit = next_unit
        unit_changed = True
    return elapsed_time, unit, unit_changed


def format_time(elapsed_time):
    units = ["seconds", "minutes", "hours", "days", "years"]
    max_values = [60, 60, 24, 365]
    for i in range(len(max_values)):
        elapsed_time, unit, unit_changed = _format_time_helper(
            elapsed_time, units[i], units[i + 1], max_values[i]
        )
        if not unit_changed:
            break

    return f"{elapsed_time} {unit}"


def sanitize(x):
    problematic = """";.,{}'[]:"""
    for char in problematic:
        x = x.replace(char, "_")
    return re.sub(r"[^\x00-\x7F]+", " ", x)


def unique_value(x, name):
    while name in x:
        name += str(np.random.randint(10))
    return name


def unique_name(X, name):
    if len(X.shape) == 2:
        columns = X.columns
    else:
        columns = X.name
    return unique_value(columns, sanitize(name))


def make_dataframe(X):
    if type(X) is pd.Series:
        X = pd.DataFrame({X.name: X.copy()})
    else:
        X = X.copy()
    return X


def subset(X, columns):
    if len(X.shape) == 2:
        return X[columns]
    assert len(columns) == 1
    return X


def reshape(X, shape_len):
    if shape_len == 2:
        return make_dataframe(X)
    if len(X.shape) == 2:
        assert X.shape[1] == 1
        return X[X.columns[0]]
    return X


def duration(label):
    def duration_decorator(func):
        @functools.wraps(func)
        def measure_duration(*args, **kw):
            start = time.time()
            try:
                res = func(*args, **kw)
                return res
            finally:
                time_elapsed = format_time(time.time() - start)
                print(f"Duration of stage {label}: {time_elapsed}")

        return measure_duration

    return duration_decorator


def update_kwargs(kwargs, force=False, **additional_kwargs):
    for keyword, value in additional_kwargs.items():
        if force or keyword not in kwargs:
            kwargs[keyword] = value
    return kwargs


def kwargs_subset(kwargs, prefix, remove_prefix=True):
    new_kwargs = {}
    for key, val in kwargs.items():
        if key.startswith(prefix):
            if remove_prefix:
                key = key[len(prefix) :]
            new_kwargs[key] = val
    return new_kwargs


def kwargs_subset_except(kwargs, prefixes):
    return {
        key: val
        for key, val in kwargs.items()
        if not any([key.startswith(prefix) for prefix in prefixes])
    }


def get_metric_str(metric):
    if type(metric) is not str:
        return metric.__name__
    return metric


def is_classification(model):
    if issubclass(model.__class__, ClassifierMixin):
        return True
    if issubclass(model.__class__, RegressorMixin):
        return False

    raise ValueError(
        "Model is not subclass of neither " "ClassifierMixin nor RegressorMixin"
    )


def is_classification_metric(metric):
    if metric in METRIC_DICT:
        return METRIC_DICT[metric]["clf"]

    raise ValueError(f"Unknown whether {metric} is a classification metric.")


def get_probability(metric):
    metric = get_metric_str(metric)
    if metric not in METRIC_DICT:
        raise ValueError(
            "Can not infer if probability is required " f"for {metric} score"
        )
    return METRIC_DICT[metric]["proba"]


def get_maximize(metric):
    metric = get_metric_str(metric)
    if metric not in METRIC_DICT:
        raise ValueError(f"Can not infer if score {metric} should be maximized.")
    return METRIC_DICT[metric]["maximize"]


def get_metric(classification):
    if classification:
        return DEFAULT_CASSIFICATION_METRIC
    return DEFAULT_REGRESSION_METRIC


def get_default_settings(proba, maximize, metric, classification, model):
    if classification is None:
        classification = is_classification(model)

    if metric is None:
        metric = get_metric(classification)

    if proba is None:
        proba = get_probability(metric)

    if maximize is None:
        maximize = get_maximize(metric)

    return proba, maximize, metric, classification


def join_cols(cols):
    return JOIN_STR.join([str(x) for x in cols])


def create_single_column(X, used_cols):
    if len(used_cols) == 1:
        return X[used_cols[0]].astype(str)
    else:
        return X[used_cols].apply(join_cols, axis=1)


def trange(x):
    if type(x) is int:
        return tqdm(range(x), leave=False)
    return tqdm(x, leave=False)


def copy_signature(source_fct):
    def copy(target_fct):
        target_fct.__signature__ = signature(source_fct)
        target_fct.__doc__ = source_fct.__doc__
        return target_fct

    return copy
