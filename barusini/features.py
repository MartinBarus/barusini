###################################################################
# Copyright (C) Martin Barus <martin.barus@gmail.com>
#
# This file is part of barusini.
#
# barusini can not be copied and/or distributed without the express
# permission of Martin Barus or Miroslav Barus
####################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm as tqdm
import pandas as pd


def trange(x):
    return tqdm(range(x), leave=False)


def is_new_better(old, new, maximize):
    if maximize:
        return old <= new
    return new <= old


def find_best_subset(
    X_old,
    y,
    cv=StratifiedKFold(n_splits=3),
    estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
    metric="roc_auc",  # roc_auc_score,
    maximize=True,
):
    X = X_old.drop(X_old.select_dtypes(object).columns, axis=1)
    for col in X:
        min_val = X[col].min()
        if pd.isna(min_val):
            min_val = 0
        X[col] = X[col].fillna(min_val - 1)
        if X[col].std() == 0:
            X = X.drop(col, axis=1)

    base_score = cross_val_score(
        estimator, X, y, cv=cv, n_jobs=-1, scoring=metric
    ).mean()
    print("BASE", base_score)
    original_best = base_score
    for j in trange(X.shape[1] - 1):
        act_best = None
        for i in trange(X.shape[1]):
            X_act = X.drop(X.columns[i], axis=1)
            act_score = cross_val_score(
                estimator, X, y, cv=cv, n_jobs=-1, scoring=metric
            ).mean()
            if is_new_better(base_score, act_score, maximize):
                base_score = act_score
                act_best = i

        if act_best:
            X = X.drop(X.columns[i], axis=1)
        else:
            break
    print("ORIGINAL BEST", original_best)
    print("NEW BEST", base_score)
    print("DIFF", abs(base_score - original_best))
    print("DROPPED", [x for x in X_old.columns if x not in X.columns])
    print("Left", [x for x in X.columns])
    return X
