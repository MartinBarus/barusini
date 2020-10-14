from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from barusini.model_tuning import LightGBMTrial, XGBoostTrial
from barusini.utils import get_terminal_size


ESTIMATOR = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
CV = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
CLASSIFICATION = True
SCORER = log_loss
PROBA = True
MAXIMIZE = False
STAGE_NAME = "STAGE"
TERMINAL_COLS = get_terminal_size()
MAX_RELATIVE_CARDINALITY = 0.9
MAX_ABSOLUTE_CARDINALITY = 10000
TRIAL = XGBoostTrial()
STR_SUBSPACE = "   "
STR_SPACE = f"{STR_SUBSPACE}   "
STR_BULLET = f"{STR_SUBSPACE} * "
