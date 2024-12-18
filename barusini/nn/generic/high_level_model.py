import base64
import filecmp
import glob
import hashlib
import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd

from barusini.constants import TEST_MODE, TRAIN_MODE
from barusini.nn.generic.loading import Serializable
from barusini.nn.generic.mid_level_model import Model
from barusini.nn.generic.utils import get_data, set_seed
from barusini.utils import is_classification_metric
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_attributes(self, **kwargs):
    attrs = [
        a
        for a in sorted(dir(self))
        if not a.startswith("_") and not callable(getattr(self, a))
    ]
    attrs = sorted(attrs)
    dct = OrderedDict()
    for attr in attrs:
        val = getattr(self, attr)
        if val is not None:  # exclude uninitialized params
            dct[attr] = val

    for key, val in kwargs.items():
        dct[key] = val

    return dct


class HighLeveMetalModel(Serializable):
    _ckpt_folder_name = "/ckpt/"
    _ckpt_file_ext = "*.ckpt"

    def __init__(
        self,
        backbone="timm_or_transformers_backbone",
        artifact_path="barusini_nn/",
        model_id="",
        **kwargs,
    ):
        self.backbone = backbone
        self.artifact_path = artifact_path
        self.model_id = model_id
        self.val_split = None
        self._dct_str = None
        self._hash = None
        self.experiment_name = None
        self.experiment_path = None
        self.ckpt_save_path = None
        self.logger_path = None
        self.oof_preds_file = None

    def set_hash(self, **kwargs):
        # create hash of important settings
        self._dct_str = json.dumps(get_attributes(self, **kwargs))
        self._hash = hashlib.md5(self._dct_str.encode("utf-8")).digest()
        self._hash = base64.b64encode(self._hash).decode("utf-8")
        self._hash = self._hash.replace("/", "@")
        self.experiment_name = f"{self.backbone.replace('.pt', '')}_{self._hash}"
        if self.model_id:
            self.experiment_name = f"{self.model_id}_{self.experiment_name}"
        self.experiment_path = self.artifact_path + self.experiment_name
        self.experiment_path = self.experiment_path + "/{val}/seed_{seed}"
        self.ckpt_save_path = self.experiment_path + self._ckpt_folder_name
        self.logger_path = self.experiment_path
        self.oof_preds_file = self.experiment_path + "oof_preds.pickle"

    def checkpoint_folder(self):
        return self.ckpt_save_path.format(val=self.val_split, seed=self._seed)

    def config_file(self):
        return os.path.join(self.checkpoint_folder(), "high_level_config.json")

    def status_file(self):
        path = self.ckpt_save_path.format(val=self.val_split, seed=self._seed)
        return os.path.join(path, "status.txt")

    def is_trained(self):
        return os.path.exists(self.status_file())

    def remove_duplicated_checkpoint(self):
        ckpt_save_path = self.ckpt_save_path.format(val=self.val_split, seed=self._seed)
        checkpoints = glob.glob(ckpt_save_path + self._ckpt_file_ext)
        last_checkpoint = [x for x in checkpoints if "last" in x]
        best_checkpoint = [x for x in checkpoints if "last" not in x]
        assert len(last_checkpoint) == 1, len(last_checkpoint)
        assert len(best_checkpoint) == 1, len(best_checkpoint)
        last_checkpoint = last_checkpoint[0]
        best_checkpoint = best_checkpoint[0]
        if filecmp.cmp(last_checkpoint, best_checkpoint):
            print("Removing duplicated checkpoint")
            os.remove(best_checkpoint)

    def create_config(self):
        exp_path = self.experiment_path.format(val=self.val_split, seed=self._seed)
        os.makedirs(exp_path, exist_ok=True)
        ckpt_conf = self.config_file()
        os.makedirs(os.path.dirname(ckpt_conf), exist_ok=True)
        with open(ckpt_conf, "w") as file:
            file.write(self._dct_str)
        return ckpt_conf

    def write_status_done(self):
        with open(self.status_file(), "w") as file:
            file.write("Done.")


class HighLevelModel(HighLeveMetalModel):
    model_class = None
    train_dataset_class = None
    val_dataset_class = None

    def __init__(
        self,
        n_classes,
        metric,
        label,
        backbone="timm_or_transformers_backbone",
        batch_size=16,
        artifact_path="barusini_nn/",
        lr=1e-4,
        weight_decay=1e-4,
        gradient_accumulation_steps=1,
        optimizer="adamw",
        scheduler={"method": "cosine", "warmup_epochs": 1},
        max_epochs=10,
        precision=16,
        model_id="",
        seed=1234,
        val_check_interval=1.0,
        log_every_n_steps=50,
        metric_threshold=None,
        classification=None,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone, artifact_path=artifact_path, model_id=model_id
        )
        if seed is None:
            seed = np.random.randint(1, 1e4)

        assert type(seed) not in [
            list,
            tuple,
        ], "seed has to be a single number, use nlp_ensemble instead!"

        self.n_classes = n_classes
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.label = label
        self.batch_size = batch_size
        self.artifact_path = artifact_path
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.precision = precision
        self.weight_decay = weight_decay
        self._seed = seed
        self.val_check_interval = val_check_interval
        self.log_every_n_steps = log_every_n_steps
        self.set_hash(**kwargs)
        self.val_data_path = None
        self.classification = classification

    def fit(
        self,
        train,
        val,
        val_split=None,
        num_workers=8,
        gpus=("0",),
        verbose=True,
        custom_metric=None,
        custom_loss=None,
    ):
        classification = self.classification
        if custom_metric or custom_loss:
            err = "In case of custom {} you must also provide custom {}."
            assert self.classification is not None, "specify if this is classification"
            assert custom_metric is not None, err.format("loss", "metric")
            assert custom_loss is not None, err.format("metric", "loss")

        elif self.classification is None:
            classification = is_classification_metric(self.metric)

        print(
            "Params",
            "val_split",
            val_split,
            "num_workers",
            num_workers,
        )
        param_err = "{} should be path to {} file or a dataframe"
        if type(train) not in [str, pd.DataFrame]:
            raise ValueError(param_err.format("train", "training"))
        if type(val) not in [str, pd.DataFrame, type(None)]:
            raise ValueError(param_err.format("val", "validation"))

        self.val_data_path = val
        if val_split is not None:
            # if fold provided
            self.val_split = val_split
        else:
            self.val_split = os.path.basename(val)

        if self.is_trained():
            if verbose:
                print("Model already trained")
            return

        ckpt_conf = self.create_config()
        set_seed(self._seed)
        train = get_data(train)
        val = get_data(val)
        if val is None:
            val = train.head(self.batch_size * 10)

        tr_ds = self.train_dataset_class.from_config(
            config_path=ckpt_conf, df=train, mode=TRAIN_MODE
        )
        val_ds = self.val_dataset_class.from_config(
            config_path=ckpt_conf, df=val, mode=TEST_MODE
        )

        tr_dl = DataLoader(
            dataset=tr_ds,
            batch_size=self.batch_size,
            sampler=RandomSampler(tr_ds),
            num_workers=num_workers,
            pin_memory=False,
            multiprocessing_context="fork",
        )
        print("ds len", len(tr_ds))

        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=self.batch_size,
            sampler=SequentialSampler(val_ds),
            num_workers=num_workers,
            pin_memory=False,
            multiprocessing_context="fork",
        )
        trainer = self.get_trainer(gpus)
        model = Model(
            lr=self.lr,
            len_tr_dl=len(tr_dl),
            metric=self.metric,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            model_path=self.oof_preds_file.format(val=self.val_split, seed=self._seed),
            experiment_name=self.experiment_name,
            n_classes=self.n_classes,
            weight_decay=self.weight_decay,
            val_check_interval=self.val_check_interval,
            model=self.model_class.from_config(config_path=ckpt_conf),
            metric_threshold=self.metric_threshold,
            classification=classification,
            custom_metric=custom_metric,
            custom_loss=custom_loss,
        )

        trainer.fit(model, tr_dl, val_dl)
        self.remove_duplicated_checkpoint()
        model.model.to_folder(self.checkpoint_folder())
        tr_ds.to_folder(self.checkpoint_folder())
        self.write_status_done()

    def get_trainer(self, gpus):
        logger_path = self.logger_path.format(val=self.val_split, seed=self._seed)
        logger = TensorBoardLogger(save_dir=logger_path)

        ckpt_save_path = self.ckpt_save_path.format(val=self.val_split, seed=self._seed)
        ckpt = ModelCheckpoint(
            ckpt_save_path,
            monitor="val_loss",
            verbose=False,
            mode="min",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
        )

        experiment_path = self.experiment_path.format(
            val=self.val_split, seed=self._seed
        )
        return Trainer(
            accelerator="gpu",
            devices=gpus,
            logger=logger,
            max_epochs=self.max_epochs,
            accumulate_grad_batches=self.gradient_accumulation_steps,
            default_root_dir=experiment_path,
            callbacks=[ckpt],
            precision=self.precision,
            num_sanity_val_steps=0,
            val_check_interval=self.val_check_interval,
            log_every_n_steps=self.log_every_n_steps,
        )

    def get_oof_dict(self):
        oof_file_path = self.oof_preds_file.format(val=self.val_split, seed=self._seed)
        with open(oof_file_path, "rb") as file:
            return pickle.load(file)

    def oof_dataset(self):
        oof_dict = self.get_oof_dict()
        if type(self.val_data_path) is str:
            val = pd.read_csv(self.val_data_path)
        else:
            # if val provided as pd.DataFrame
            val = self.val_data_path.copy()
        target = oof_dict["target"]
        # Assert the preds are in correct order
        assert all(np.isclose(val[self.label].values, target))
        val["oof_pred"] = oof_dict["preds"]
        return val
