import math
import pickle
from collections import OrderedDict

import numpy as np
import sklearn.metrics
from scipy.special import expit as sigmoid
from scipy.special import softmax

import pytorch_lightning as pl
import torch
from barusini.constants import rmse, TRAIN_MODE, VALID_MODE
from barusini.nn.generic.utils import expand_classification_label
from torch.optim import Adam
from transformers import AdamW, get_cosine_schedule_with_warmup


class Model(pl.LightningModule):
    def __init__(
        self,
        lr,
        len_tr_dl,
        metric,
        gradient_accumulation_steps,
        optimizer,
        scheduler,
        max_epochs,
        batch_size,
        model_path,
        experiment_name,
        n_classes,
        weight_decay,
        val_check_interval=1.0,
        metric_threshold=None,
        model=None,
        classification=None,
    ):
        super(Model, self).__init__()

        self.lr = lr
        self.len_tr_dl = len_tr_dl
        self.metric = metric
        self.metric_threshold = metric_threshold

        self.weight_decay = weight_decay
        self.optimizer_str = optimizer
        self.optimizer = None
        self.scheduler_dict = scheduler
        self.scheduler = None
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.experiment_name = experiment_name
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_classes = n_classes
        self.loss_fn = self.get_loss_fn()  # used for computing gradient
        self.sklearn_metric = self.get_sklearn_metric()  # used as val loss
        self.model = model
        self.val_check_interval = val_check_interval
        self.num_train_steps = math.ceil(len_tr_dl / gradient_accumulation_steps)
        self.classification = classification

    def get_loss_fn(self):
        if self.metric.lower() in ["rmse", "mse"]:
            return torch.nn.MSELoss()

        if self.metric.lower() in ["mean_absolute_error"]:
            return torch.nn.L1Loss()

        if self.metric.lower() in [
            "roc_auc_score",
            "log_loss",
            "accuracy",
            "accuracy_score",
        ]:
            if self.n_classes < 3:
                # allows to implement label smoothing, link is sigmoid
                return torch.nn.BCEWithLogitsLoss()
            # link is softmax
            return torch.nn.CrossEntropyLoss()
        raise ValueError(f"metric {self.metric} not supported")

    def get_sklearn_metric(self):
        metric_name = self.metric.lower()
        if metric_name == "rmse":
            return rmse

        if hasattr(sklearn.metrics, metric_name):
            return getattr(sklearn.metrics, metric_name)

        err = f"metric {self.metric} is not supported, use metric from sklearn.metrics"
        raise ValueError(err)

    def forward(self, x, mode):
        return self.model(x, mode)

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        print("trainable_parameters", len(trainable_parameters))

        if self.optimizer_str == "adamw":
            self.optimizer = AdamW(
                [{"params": trainable_parameters}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_str == "adam":
            self.optimizer = Adam(
                [{"params": trainable_parameters}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            print("USING MY ADAM", self.weight_decay, self.lr)
        elif self.optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(
                [{"params": trainable_parameters}],
                lr=self.lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay,
            )

        if self.scheduler_dict["method"] == "cosine":
            num_warmup_steps = (
                self.num_train_steps * self.scheduler_dict["warmup_epochs"]
            )
            num_training_steps = int(self.num_train_steps * (self.max_epochs))
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            print("USING MY OPT AND SCHED")

            return (
                [self.optimizer],
                [{"scheduler": self.scheduler, "interval": "step"}],
            )

        elif self.scheduler_dict["method"] == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_dict["step_size"],
                gamma=self.scheduler_dict["gamma"],
                last_epoch=-1,
            )
            return (
                [self.optimizer],
                [{"scheduler": self.scheduler, "interval": "epoch"}],
            )
        elif self.scheduler_dict["method"] == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.1, mode="max", patience=1, verbose=True
            )
            return (
                [self.optimizer],
                [
                    {
                        "scheduler": self.scheduler,
                        "interval": "epoch",
                        "reduce_on_plateau": True,
                        "monitor": "val_loss",
                    }
                ],
            )
        else:
            self.scheduler = None
            return [self.optimizer]

    def loss(self, preds, target):
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            preds = preds.squeeze()

        if len(target.shape) == 2 and target.shape[1] == 1:
            target = target.squeeze()

        if self.classification:
            target = target.long()

        loss = self.loss_fn(preds, target)
        return loss

    def get_loss(self, batch, mode):
        input_dict = batch["input"]
        target = batch["target"]
        output_dict = self.forward(input_dict, mode)
        preds = output_dict["logits"]
        loss = self.loss(preds, target)
        return loss, preds, target

    def training_step(self, batch, batch_num):
        loss, preds, target = self.get_loss(batch, TRAIN_MODE)
        step = self.global_step * self.batch_size * self.gradient_accumulation_steps
        tb_dict = {"train_loss": loss, "step": step}

        for i, param_group in enumerate(self.optimizer.param_groups):
            tb_dict[f"lr/lr{i}"] = param_group["lr"]

        self.log_all(tb_dict)
        output = OrderedDict({"loss": loss})
        return output

    def validation_step(self, batch, batch_idx):
        loss, preds, target = self.get_loss(batch, VALID_MODE)
        output = {
            "val_loss": loss.view(1),
            "preds": preds,
            "target": target,
        }
        return output

    def log_all(self, dct):
        for k, v in dct.items():
            self.log(k, v, prog_bar=True)

    def validation_epoch_end(self, outputs):
        out_val = {}
        for key in outputs[0].keys():
            out_val[key] = torch.cat([o[key] for o in outputs])

        val_loss = self.loss(out_val["preds"], out_val["target"])

        for key in out_val.keys():
            out_val[key] = out_val[key].detach().cpu().numpy().astype(np.float32)

        with open(self.model_path, "wb") as handle:
            pickle.dump(out_val, handle)

        val_loss_mean = np.mean(out_val["val_loss"])
        if self.n_classes > 1 and self.classification:
            # turn logits into probabilities for sklearn metrics
            out_val["target"] = out_val["target"].round().astype(int)
            if self.n_classes == 2:
                out_val["preds"] = sigmoid(out_val["preds"])
            else:
                out_val["preds"] = softmax(out_val["preds"], axis=1)
                out_val["target"] = expand_classification_label(out_val["target"])

        if self.metric_threshold is not None:
            preds = (out_val["preds"] >= self.metric_threshold).astype(int)
            val_metric = self.sklearn_metric(out_val["target"], preds)
        else:
            val_metric = self.sklearn_metric(out_val["target"], out_val["preds"])

        tqdm_dict = {
            "val_metric": val_metric,
            "val_loss": val_loss,
            "val_loss_mean": val_loss_mean,
        }
        if int(self.val_check_interval) == self.val_check_interval:
            tqdm_dict["step"] = self.current_epoch

        self.log_all(tqdm_dict)
