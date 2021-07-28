import math
import pickle
from collections import OrderedDict

import numpy as np

import pytorch_lightning as pl
import torch
from barusini.constants import rmse
from barusini.nlp.low_level_model import NlpNet
from transformers import (
    AdamW,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


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
        backbone,
        n_classes,
        weight_decay,
        pretrained_weights,
        val_check_interval=1.0,
        net_class=NlpNet,
    ):
        super(Model, self).__init__()

        self.lr = lr
        self.len_tr_dl = len_tr_dl
        self.metric = metric

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
        self.loss_fn = self.get_loss_fn()  # used for computing gradient
        self.sklearn_metric = self.get_sklearn_metric()  # used as val loss
        self.model = net_class(backbone, n_classes, pretrained_weights)
        self.val_check_interval = val_check_interval
        self.num_train_steps = math.ceil(
            len_tr_dl / gradient_accumulation_steps
        )

    def get_loss_fn(self):
        if self.metric.lower() in ["rmse", "mse"]:
            return torch.nn.MSELoss()
        raise ValueError(f"metric {self.metric} not supported")

    def get_sklearn_metric(self):
        if self.metric.lower() == "rmse":
            return rmse

        raise ValueError(f"metric {self.metric} not supported")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        parameters = list(self.model.parameters())
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, parameters)
        )
        print("trainable_parameters", len(trainable_parameters))

        if self.optimizer_str == "adamw":
            self.optimizer = AdamW(
                [{"params": trainable_parameters}],
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            print("USING MY ADAM")
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

    def get_loss(self, batch):
        input_dict = batch["input"]
        target = batch["target"]
        output_dict = self.forward(input_dict)
        preds = output_dict["logits"]
        loss = self.loss_fn(preds.view(-1), target)
        return loss, preds, target

    def training_step(self, batch, batcn_num):
        loss, preds, target = self.get_loss(batch)
        step = (
            self.global_step
            * self.batch_size
            * self.gradient_accumulation_steps
        )
        tb_dict = {"train_loss": loss, "step": step}

        for i, param_group in enumerate(self.optimizer.param_groups):
            tb_dict[f"lr/lr{i}"] = param_group["lr"]

        self.log_all(tb_dict)
        output = OrderedDict({"loss": loss})
        return output

    def validation_step(self, batch, batch_idx):
        loss, preds, target = self.get_loss(batch)
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

        for key in out_val.keys():
            out_val[key] = (
                out_val[key].detach().cpu().numpy().astype(np.float32)
            )

        with open(self.model_path, "wb") as handle:
            pickle.dump(out_val, handle)

        val_loss_mean = np.mean(out_val["val_loss"])
        val_loss = self.sklearn_metric(out_val["target"], out_val["preds"])

        tqdm_dict = {
            "val_loss": val_loss,
            "val_loss_mean": val_loss_mean,
        }
        if int(self.val_check_interval) == self.val_check_interval:
            tqdm_dict["step"] = self.current_epoch

        self.log_all(tqdm_dict)
