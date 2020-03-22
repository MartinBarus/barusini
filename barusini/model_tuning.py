import optuna
import copy
import numpy as np
from xgboost import XGBClassifier
from functools import partial


def xgboost_objective(
    model, X_train, y_train, X_test, cv, scoring, maximize, probability, trial,
):
    min_child_weight = trial.suggest_loguniform("min_child_weight", 1e-1, 1e3)
    # gamma = [0.5, 1, 1.5, 2, 5]
    subsample = trial.suggest_uniform("subsample", 0.6, 1)
    colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.6, 1)
    max_depth = trial.suggest_int("max_depth", 3, 12)

    params = dict(
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        learning_rate=trial.suggest_categorical('learning_rate', [0.01]),
        seed=trial.suggest_categorical('seed', [42]),
        n_estimators=1000,
    )

    new_model = copy.deepcopy(model)
    new_model.model = XGBClassifier(**params)

    oof = np.zeros(len(X_train))
    preds = None
    if X_test is not None:
        preds = np.zeros(len(X_test))

    n_iterations = []
    for i, (idxT, idxV) in enumerate(cv.split(X_train, y_train)):

        print(" rows of train =", len(idxT), "rows of holdout =", len(idxV))
        X_act_train = X_train.iloc[idxT]
        y_act_train = y_train.iloc[idxT]

        X_act_val = X_train.iloc[idxV]
        y_act_val = y_train.iloc[idxV]

        new_model.fit(
            X_act_train,
            y_act_train,
            eval_set=[(X_act_val, y_act_val)],
            verbose=0,
            early_stopping_rounds=20,
        )

        if probability:
            oof_preds = new_model.predict_proba(X_act_val)[:, 1]
        else:
            oof_preds = new_model.predict(X_act_val)
        oof[idxV] += oof_preds
        if X_test is not None:
            if probability:
                preds += new_model.predict_proba(X_test)[:, 1] / cv.n_splits
            else:
                preds += new_model.predict(X_test) / cv.n_splits

        n_iterations.append(new_model.model.best_iteration)

    if X_test is not None and not probability:
        preds = preds.round().astype(int)


    iter_mean = np.mean(n_iterations)
    iter_std = np.std(n_iterations)
    n_estimators = int(np.round(iter_mean))
    trial.suggest_int('n_estimators', n_estimators, n_estimators)
    score = scoring(y_train, oof)
    score = -score if maximize else maximize
    print("XGB OOF CV=", score, "mean", iter_mean, "std", iter_std)
    return score


def optimize_xboost(
    model, X_train, y_train, X_test, cv, scoring, maximize, proba, n_trials=20
):
    objective = partial(
        xgboost_objective,
        model,
        X_train,
        y_train,
        X_test,
        cv,
        scoring,
        maximize,
        proba
    )

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    print(
        f"Out of {n_trials} trials the best score is {study.best_value}"
        f" with params {study.best_params}"
    )
    return study.best_trial
