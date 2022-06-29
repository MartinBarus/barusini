import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold


def get_terminal_size():
    try:
        _, size = os.popen("stty size", "r").read().split()
        return int(size)
    except ValueError:  # Running from Pycharm causes ValueError
        return 101


ESTIMATOR = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
CV = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
UNIVERSAL_CV = KFold(n_splits=3, random_state=42, shuffle=True)
STAGE_NAME = "STAGE"
JOIN_STR = "_X_"
TERMINAL_COLS = get_terminal_size()
MAX_RELATIVE_CARDINALITY = 0.9
MAX_ABSOLUTE_CARDINALITY = 10000
STR_SUBSPACE = "   "
STR_SPACE = f"{STR_SUBSPACE}   "
STR_BULLET = f"{STR_SUBSPACE} * "
METRIC_STR_MAPPING = {
    "accuracy": "accuracy_score",
    "balanced_accuracy": "balanced_accuracy_score",
    "average_precision": "average_precision_score",
    "neg_brier_score": "brier_score_loss",
    "f1": "f1_score",
    "f1_micro": "f1_score",
    "f1_macro": "f1_score",
    "f1_weighted": "f1_score",
    "f1_samples": "f1_score",
    "neg_log_loss": "log_loss",
    "precision": "precision_score",
    "recall": "recall_score",
    "jaccard": "jaccard_score",
    "roc_auc": "roc_auc_score",
    "roc_auc_ovr": "roc_auc_score",
    "roc_auc_ovo": "roc_auc_score",
    "roc_auc_ovr_weighted": "roc_auc_score",
    "roc_auc_ovo_weighted": "roc_auc_score",
    "explained_variance": "explained_variance_score",
    "max_error": "max_error",
    "neg_mean_absolute_error": "mean_absolute_error",
    "neg_mean_squared_error": "mean_squared_error",
    "neg_root_mean_squared_error": "mean_squared_error",
    "neg_mean_squared_log_error": "mean_squared_log_error",
    "neg_median_absolute_error": "median_absolute_error",
    "r2": "r2_score",
    "neg_mean_poisson_deviance": "mean_poisson_deviance",
    "neg_mean_gamma_deviance": "mean_gamma_deviance",
    "rmse": "rmse",
}
METRIC_DICT = {
    "accuracy": {"proba": False, "maximize": True, "clf": True},
    "balanced_accuracy": {"proba": False, "maximize": True, "clf": True},
    "average_precision": {"proba": False, "maximize": True, "clf": True},
    "neg_brier_score": {"proba": True, "maximize": False, "clf": True},
    "f1": {"proba": False, "maximize": True, "clf": True},
    "f1_micro": {"proba": False, "maximize": True, "clf": True},
    "f1_macro": {"proba": False, "maximize": True, "clf": True},
    "f1_weighted": {"proba": False, "maximize": True, "clf": True},
    "f1_samples": {"proba": False, "maximize": True, "clf": True},
    "neg_log_loss": {"proba": True, "maximize": False, "clf": True},
    "precision": {"proba": False, "maximize": True, "clf": True},
    "recall": {"proba": False, "maximize": True, "clf": True},
    "jaccard": {"proba": False, "maximize": True, "clf": True},
    "roc_auc": {"proba": True, "maximize": True, "clf": True},
    "roc_auc_ovr": {"proba": True, "maximize": True, "clf": True},
    "roc_auc_ovo": {"proba": True, "maximize": True, "clf": True},
    "roc_auc_ovr_weighted": {"proba": True, "maximize": True, "clf": True},
    "roc_auc_ovo_weighted": {"proba": True, "maximize": True, "clf": True},
    "explained_variance": {"proba": False, "maximize": True, "clf": True},
    "max_error": {"proba": False, "maximize": False, "clf": False},
    "neg_mean_absolute_error": {"proba": False, "maximize": False, "clf": False},
    "neg_mean_squared_error": {"proba": False, "maximize": False, "clf": False},
    "neg_root_mean_squared_error": {"proba": False, "maximize": False, "clf": False},
    "neg_mean_squared_log_error": {"proba": False, "maximize": False, "clf": False},
    "neg_median_absolute_error": {"proba": False, "maximize": False, "clf": False},
    "r2": {"proba": False, "maximize": True, "clf": False},
    "neg_mean_poisson_deviance": {"proba": False, "maximize": False, "clf": False},
    "neg_mean_gamma_deviance": {"proba": False, "maximize": False, "clf": False},
    "rmse": {"proba": False, "maximize": False, "clf": False},
}
METRIC_DICT = {
    **METRIC_DICT,
    **{value: METRIC_DICT[key] for key, value in METRIC_STR_MAPPING.items()},
}


def rmse(y_true, y_pred, sample_weight=None):
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    return np.sqrt(mse)


DEFAULT_CASSIFICATION_METRIC = log_loss
DEFAULT_REGRESSION_METRIC = rmse

TRAIN_MODE = "train"
TEST_MODE = "test"
VALID_MODE = "validation"
