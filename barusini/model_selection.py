from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from barusini.utils import get_probability


def validation(model, X, y, train, test, scoring, proba=None):
    proba = get_probability(scoring) if proba is None else proba
    trn_X = X.loc[train]
    trn_y = y.loc[train]
    model.fit(trn_X, trn_y)

    tst_X = X.loc[test]
    tst_y = y.loc[test]
    if proba:
        predictions = model.predict_proba(tst_X)
        if predictions.shape[1] == 2:
            predictions = predictions[:, -1]
    else:
        predictions = model.predict(tst_X)
    score = scoring(tst_y, predictions)
    return score


def cross_val_score_parallel(model, X, y, cv, scoring, n_jobs, proba=None):
    parallel = Parallel(n_jobs=n_jobs, verbose=False, pre_dispatch="2*n_jobs")
    scores = parallel(
        delayed(validation)(deepcopy(model), X, y, train, test, scoring, proba)
        for train, test in cv.split(X, y)
    )

    return np.mean(scores), model


def cross_val_score_sequential(model, X, y, cv, scoring, proba=None):
    scores = [
        validation(deepcopy(model), X, y, train, test, scoring, proba)
        for train, test in cv.split(X, y)
    ]
    return np.mean(scores), model


def cross_val_score(model, X, y, cv, scoring, n_jobs, proba=None):
    if n_jobs < 2 and n_jobs >= 0:
        return cross_val_score_sequential(model, X, y, cv, scoring, proba=proba)
    return cross_val_score_parallel(model, X, y, cv, scoring, n_jobs, proba)


class TimeSplit(object):
    """Abstract class for time based validation split
    """

    def __init__(self, time_col: pd.Series):
        self.time_col = time_col

    def _get_dates(self):
        raise ValueError("Not implemented")

    def split(self, X, *args, verbose=False, **kwargs):
        assert X.shape[0] == self.time_col.size
        assert all(X.index.values == self.time_col.index.values)
        for i, (start, stop) in enumerate(self._get_dates()):
            valid = self.time_col[
                (self.time_col <= stop) & (self.time_col >= start)
            ]
            train = self.time_col[self.time_col < start]

            if verbose:
                print(f"step {i} ({start}, {stop}):")
                print(f"trn\t{train.min()} - {train.max()}")
                print(f"val\t{valid.min()} - {valid.max()}\n")

            train = train.index.values
            valid = valid.index.values
            yield train, valid

    def print_periods(self):
        for _, _ in self.split(self.time_col, verbose=True):
            pass


class TimeRangeSplit(TimeSplit):
    def __init__(self, time_col: pd.Series, start, stop, shift, n_splits):
        """This class creates time-base validation splits using fixed time shift
        together with first time of validation start and stop and number of
        splits.

        :param time_col: pd.Series: pandas Series with time value (int/datetime)
        :param start: int/datetime: start of the most recent validation period
        :param stop: int/datetime: stop of the most recent validation period
        :param shift: int/timedelta: amount by which both start and stop will be
            shifted for the next validation split (positive)
        :param n_splits: int: number of splits

        Example - data X contains column "year" with values 2005 - 2020
            The following example shows time based validation of 1 year:
            ```
            import pandas as pd


            X = pd.DataFrame({"year": range(2005, 2021, 1)})
            trs = TimeRangeSplit(
                X["year"],
                start=2020,
                stop=2020,
                shift=1,
                n_splits=3,
            )
            trs.print_periods()
            ```
            produces following result:
            ```
            step 0 (2020, 2020):
            trn	2005 - 2019
            val	2020 - 2020

            step 1 (2019, 2019):
            trn	2005 - 2018
            val	2019 - 2019

            step 2 (2018, 2018):
            trn	2005 - 2017
            val	2018 - 2018
            ```
            This class is compatible with sklearn.model_selection.KFold
        """
        super().__init__(time_col)
        self.val_start = start
        self.val_stop = stop
        self.shift = shift
        self.n_splits = n_splits

    def _get_dates(self):
        for i in range(self.n_splits):
            offset = i * self.shift
            start = self.val_start - offset
            stop = self.val_stop - offset
            yield start, stop


class TimeSplitList(TimeSplit):
    def __init__(self, time_col: pd.Series, start_stop_list):
        """This class creates time-base validation splits using list of pairs of
        validation start and stop times.

        :param time_col: pd.Series: pandas Series with time value (int/datetime)
        :param start_stop_list: List[Tuple[int/datetime, int/datetime]]: list of
            pairs of validation start and stop times.

        Example - data X contains column "year" with values 2005 - 2020
            The following example shows custom time based validation:
            ```
            import pandas as pd


            X = pd.DataFrame({"year": range(2005, 2021, 1)})
            splits = [(2020, 2020), (2018, 2019), (2010, 2012)]
            tsl = TimeSplitList(X["year"], splits)
            tsl.print_periods()
            ```
            produces following result:
            ```
            step 0 (2020, 2020):
            trn	2005 - 2019
            val	2020 - 2020

            step 1 (2018, 2019):
            trn	2005 - 2017
            val	2018 - 2019

            step 2 (2010, 2012):
            trn	2005 - 2009
            val	2010 - 2012
            ```
            This class is compatible with sklearn.model_selection.KFold
        """
        super().__init__(time_col)
        self.start_stop_list = start_stop_list

    def _get_dates(self):
        return self.start_stop_list
