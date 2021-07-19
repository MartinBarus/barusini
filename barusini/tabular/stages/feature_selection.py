from copy import deepcopy

from barusini.constants import CV
from barusini.tabular.stages.generic_stage import generic_change
from barusini.utils import duration, trange


def feature_reduction_generator(model):
    for idx in trange(len(model.transformers)):
        new_model = deepcopy(model)
        del new_model.transformers[idx]
        yield new_model


@duration("Find Best Subset")
def find_best_subset(
    X, y, model, cv=CV, metric=None, maximize=None, proba=None, **kwargs
):
    return generic_change(
        X,
        y,
        model,
        stage_name="Finding best subset",
        generator=feature_reduction_generator,
        recursive=True,
        cv_n_jobs=1,
        alternative_n_jobs=-1,
        cv=cv,
        metric=metric,
        maximize=maximize,
        proba=proba,
        **kwargs,
    )
