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
    return (
        x.replace("[", "_").replace("]", "_").replace(":", "_").replace("'", "")
    )


def unique_value(x, name):
    while name in x:
        name += str(np.random.randint(10))
    return name


def unique_name(X, name):
    return unique_value(X.columns, sanitize(name))


def make_dataframe(X):
    if type(X) is pd.Series:
        X = pd.DataFrame({X.name: X})
    return X

