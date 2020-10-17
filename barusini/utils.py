###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################
import os
import pickle
import pandas as pd
import numpy as np
import time


def get_terminal_size():
    try:
        _, size = os.popen("stty size", "r").read().split()
        return int(size)
    except ValueError:  # Running from Pycharm causes ValueError
        return 101


def save_object(o, path):
    with open(path, "wb") as file:
        pickle.dump(o, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)


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
    problematic = '''";.,{}'[]:'''
    for char in problematic:
        x = x.replace(char, "_")
    return x


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
