import os
import subprocess
import sys

from barusini.nn.generic.high_level_model import HighLeveMetalModel, get_attributes


class YoloModel(HighLeveMetalModel):
    _ckpt_folder_name = "/weights/"
    _ckpt_file_ext = "*.pt"

    _hyp_params = [
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "box",
        "cls",
        "cls_pw",
        "obj",
        "obj_pw",
        "iou_t",
        "anchor_t",
        "fl_gamma",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
    ]

    _bool_args = [
        "quad",
        "linear_lr",
        "multi_scale",
        "sync_bn",
        "rect",
    ]

    _value_args = [
        "epochs",
        "batch_size",
        "imgsz",
        "optimizer",
        "label_smoothing",
        "patience",
    ]

    def __init__(
        self,
        yolo_lib_path=".",
        lr0=0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        lrf=0.1,  # final OneCycleLR learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay 5e-4
        warmup_epochs=3.0,  # warmup epochs (fractions ok)
        warmup_momentum=0.8,  # warmup initial momentum
        warmup_bias_lr=0.1,  # warmup initial bias lr
        weights="yolov5s6.pt",
        box=0.05,  # box loss gain
        cls=0.5,  # cls loss gain
        cls_pw=1.0,  # cls BCELoss positive_weight
        obj=1.0,  # obj loss gain (scale with pixels)
        obj_pw=1.0,  # obj BCELoss positive_weight
        iou_t=0.20,  # IoU training threshold
        anchor_t=4.0,  # anchor-multiple threshold
        fl_gamma=0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # image HSV-Value augmentation (fraction)
        degrees=0.0,  # image rotation (+/- deg)
        translate=0.1,  # image translation (+/- fraction)
        scale=0.5,  # image scale (+/- gain)
        shear=0.0,  # image shear (+/- deg)
        perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # image flip up-down (probability)
        fliplr=0.5,  # image flip left-right (probability)
        mosaic=1.0,  # image mosaic (probability)
        mixup=0.0,  # image mixup (probability)
        copy_paste=0.0,  # segment copy-paste (probability)
        artifact_path="barusini_nn/",
        model_id="",
        epochs=15,
        batch_size=16,
        imgsz=640,
        optimizer="SGD",  # choices=['SGD', 'Adam', 'AdamW']
        label_smoothing=0.0,
        patience=100,
        quad=False,
        linear_lr=False,
        multi_scale=False,
        sync_bn=True,
        rect=False,
    ):
        super().__init__(
            backbone=weights, artifact_path=artifact_path, model_id=model_id
        )

        self.yolo_lib_path = yolo_lib_path
        self._yolo_train_path = os.path.join(self.yolo_lib_path, "train.py")
        assert os.path.exists(self._yolo_train_path), (
            f"Training Script {self._yolo_train_path} does not exist, provide proper "
            f"yolo_lib_path argument"
        )

        self.lr0 = lr0
        self.lrf = lrf
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.box = box
        self.cls = cls
        self.cls_pw = cls_pw
        self.obj = obj
        self.obj_pw = obj_pw
        self.iou_t = iou_t
        self.anchor_t = anchor_t
        self.fl_gamma = fl_gamma
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic = mosaic
        self.mixup = mixup
        self.copy_paste = copy_paste
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.optimizer = optimizer
        self.artifact_path = artifact_path
        self.label_smoothing = label_smoothing
        self.patience = patience
        self.quad = quad
        self.linear_lr = linear_lr
        self.multi_scale = multi_scale
        self.sync_bn = sync_bn
        self.rect = rect
        self.set_hash()
        self.data_yaml = None
        self.hyp_yaml = None

    def fit(
        self, train=None, num_workers=8, device="", verbose=True,
    ):
        assert type(device) is str, "device must be str"
        if train.endswith(".yaml"):
            self.val_split = os.path.basename(train).replace(".yaml", "")
            self.data_yaml = train
        else:
            raise ValueError("train has to by YOLOv5 style data yaml file")

        if self.is_trained():
            if verbose:
                print("Model already trained")
            return

        ckpt_conf = self.create_config()
        self.hyp_yaml = os.path.join(os.path.dirname(ckpt_conf), "hyp.yaml")
        self.create_hyp_yaml()
        train_args = self.compile_train_command(num_workers=num_workers, device=device)
        print(train_args)
        subprocess.run(
            train_args.split(),
            stderr=sys.stderr,
            stdout=sys.stdout,
            text=True,
            check=True,
        )
        self.write_status_done()

    def create_hyp_yaml(self):
        with open(self.hyp_yaml, "w") as file:
            for param in self._hyp_params:
                param_val = getattr(self, param)
                file.write(f"{param}: {param_val}\n")

    def compile_train_command(self, num_workers, device):
        all_params = get_attributes(self)

        value_args = [
            f"--{param.replace('_','-')} {value}"
            for param, value in all_params.items()
            if param in self._value_args
        ]

        bool_args = [
            f"--{param.replace('_','-')}"
            for param, value in all_params.items()
            if param in self._bool_args and value
        ]

        train_cmd = (
            f"python {self.yolo_lib_path}/train.py --data {self.data_yaml} --hyp "
            f"{self.hyp_yaml} --workers {num_workers}  --exist-ok --project "
            f"{self.artifact_path} --name {self.experiment_name}/{self.val_split} "
        )
        if len(device):
            train_cmd += f" --device {device}"

        train_cmd = train_cmd + " ".join(value_args + bool_args)
        return train_cmd


class YoloScorer:
    pass
