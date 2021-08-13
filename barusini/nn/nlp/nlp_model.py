import os

from barusini.nn.generic.high_level_model import HighLevelModel
from barusini.nn.generic.ensemble import Ensemble
from barusini.nn.generic.scorer import Scorer
from barusini.nn.nlp.data import NLPDataset
from barusini.nn.nlp.low_level_model import NlpNet


class NlpModel(HighLevelModel):
    model_class = NlpNet
    dataset_class = NLPDataset

    def fit(self, train, val, num_workers=8, gpus=("0",), verbose=True):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        super().fit(train, val, num_workers, gpus, verbose)


class NlpScorer(Scorer):
    model_class = NlpNet
    dataset_class = NLPDataset


class NlpEnsemble(Ensemble):
    high_level_model_class = NlpModel
