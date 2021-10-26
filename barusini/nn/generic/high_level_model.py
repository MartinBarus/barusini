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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_attributes(self, **kwargs):
    attrs = [
        a
        for a in sorted(dir(self))
        if not a.startswith("__") and not callable(getattr(self, a))
    ]
    attrs = sorted(attrs)
    dct = OrderedDict()
    for attr in attrs:
        dct[attr] = getattr(self, attr)

    for key, val in kwargs.items():
        dct[key] = val

    return dct


class HighLevelModel(Serializable):
    model_class = None
    dataset_class = None

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
        **kwargs,
    ):
        # settings that affect model's predictions
        self.backbone = backbone
        self.n_classes = n_classes
        self.metric = metric
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
        self.seed = seed
        assert type(seed) not in [
            list,
            tuple,
        ], "seed has to be a single number, use nlp_ensemble instead!"
        self.val_check_interval = val_check_interval
        self.log_every_n_steps = log_every_n_steps

        # create hash of important settings
        self._dct_str = json.dumps(get_attributes(self, **kwargs))
        self._hash = hashlib.md5(self._dct_str.encode("utf-8")).digest()
        self._hash = base64.b64encode(self._hash).decode("utf-8")
        self._hash = self._hash.replace("/", "@")

        # settings that do NOT affect model's prediction
        self.model_id = model_id

        self.experiment_name = f"{self.backbone}_{self._hash}"
        if model_id:
            self.experiment_name = f"{self.model_id}_{self.experiment_name}"
        self.experiment_path = self.artifact_path + self.experiment_name
        self.experiment_path = self.experiment_path + "/{val}/"
        self.ckpt_save_path = self.experiment_path + "/ckpt/"
        self.logger_path = self.experiment_path
        self.oof_preds_file = self.experiment_path + "oof_preds.pickle"
        self.val_data_path = None
        self.val_split = None

    def checkpoint_folder(self):
        return self.ckpt_save_path.format(val=self.val_split)

    def config_file(self):
        return os.path.join(self.checkpoint_folder(), "high_level_config.json")

    def status_file(self):
        path = self.ckpt_save_path.format(val=self.val_split)
        return os.path.join(path, "status.txt")

    def is_trained(self):
        return os.path.exists(self.status_file())

    def fit(self, train, val, num_workers=8, gpus=("0",), verbose=True):
        param_err = "{} should be path to {} file"
        if type(train) is not str or not len(train):
            raise ValueError(param_err.format("train", "training"))
        if type(val) is not str or not len(val):
            raise ValueError(param_err.format("val", "validation"))

        self.val_data_path = val
        self.val_split = os.path.basename(val)

        if self.is_trained():
            if verbose:
                print("Model already trained")
            return

        exp_path = self.experiment_path.format(val=self.val_split)
        os.makedirs(exp_path, exist_ok=True)
        ckpt_conf = self.config_file()
        os.makedirs(os.path.dirname(ckpt_conf), exist_ok=True)
        with open(ckpt_conf, "w") as file:
            file.write(self._dct_str)

        set_seed(self.seed)
        train = get_data(train)
        val = get_data(val)
        if val is None:
            val = train

        tr_ds = self.dataset_class.from_config(
            config_path=ckpt_conf, df=train, mode=TRAIN_MODE
        )
        val_ds = self.dataset_class.from_config(
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
            model_path=self.oof_preds_file.format(val=self.val_split),
            experiment_name=self.experiment_name,
            n_classes=self.n_classes,
            weight_decay=self.weight_decay,
            val_check_interval=self.val_check_interval,
            model=self.model_class.from_config(config_path=ckpt_conf),
        )

        trainer.fit(model, tr_dl, val_dl)
        self.remove_duplicated_checkpoint()
        model.model.to_folder(self.checkpoint_folder())
        tr_ds.to_folder(self.checkpoint_folder())
        with open(self.status_file(), "w") as file:
            file.write("Done.")

    def get_trainer(self, gpus):
        logger_path = self.logger_path.format(val=self.val_split)
        logger = TensorBoardLogger(save_dir=logger_path)

        ckpt_save_path = self.ckpt_save_path.format(val=self.val_split)
        ckpt = ModelCheckpoint(
            ckpt_save_path,
            monitor="val_loss",
            verbose=False,
            mode="min",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
        )

        experiment_path = self.experiment_path.format(val=self.val_split)
        return Trainer(
            gpus=gpus,
            logger=logger,
            resume_from_checkpoint=None,
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
        oof_file_path = self.oof_preds_file.format(val=self.val_split)
        with open(oof_file_path, "rb") as file:
            return pickle.load(file)

    def oof_dataset(self):
        oof_dict = self.get_oof_dict()
        val = pd.read_csv(self.val_data_path)
        target = oof_dict["target"]
        # Assert the preds are in correct order
        assert all(np.isclose(val[self.label].values, target))
        val["oof_pred"] = oof_dict["preds"]
        return val

    def remove_duplicated_checkpoint(self):
        ckpt_save_path = self.ckpt_save_path.format(val=self.val_split)
        checkpoints = glob.glob(ckpt_save_path + "*.ckpt")
        last_checkpoint = [x for x in checkpoints if "last" in x]
        best_checkpoint = [x for x in checkpoints if "last" not in x]
        assert len(last_checkpoint) == 1, len(last_checkpoint)
        assert len(best_checkpoint) == 1, len(best_checkpoint)
        last_checkpoint = last_checkpoint[0]
        best_checkpoint = best_checkpoint[0]
        if filecmp.cmp(last_checkpoint, best_checkpoint):
            print("Removing duplicated checkpoint")
            os.remove(best_checkpoint)
