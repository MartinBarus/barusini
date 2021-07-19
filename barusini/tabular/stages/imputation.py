from copy import deepcopy

import pandas as pd

from barusini.constants import CV
from barusini.tabular.stages.generic_stage import generic_change
from barusini.tabular.transformers import MissingValueImputer
from barusini.utils import duration


def get_imputation_generator(column):
    def find_imputation_generator(model):
        for aggregation in ["mode", "mean", "min", "max", "new"]:
            new_model = deepcopy(model)
            new_model.remove_transformers([column])
            new_model.transformers.append(
                MissingValueImputer(used_cols=[column], agg=aggregation)
            )
            yield new_model

    return find_imputation_generator


@duration("Find Best Imputation")
def find_best_imputation(
    X, y, model, cv=CV, metric=None, maximize=None, proba=None, **kwargs
):
    imputed_columns = [
        enc.used_cols[0]
        for enc in model.transformers
        if isinstance(enc, MissingValueImputer)
    ]
    imputed_columns = X[imputed_columns].apply(lambda x: any(pd.isna(x)))
    imputed_columns = imputed_columns[imputed_columns].index
    for column in imputed_columns:
        model = generic_change(
            X,
            y,
            model,
            stage_name=f"Finding imputation for {column}",
            generator=get_imputation_generator(column),
            cv=cv,
            metric=metric,
            maximize=maximize,
            proba=proba,
            **kwargs,
        )
    return model
