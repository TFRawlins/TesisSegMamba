import os
import re
import sys
import json
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    EnsureTyped,
)
from monai.data import Dataset, CacheDataset
from monai.inferers import SlidingWindowInferer
mport monai.transforms.compose as _monai_comp
_monai_comp.get_seed = lambda: 123
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last

# =====================
# Args
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="colorectal")
parser.add_argument("--data_dir", required=True, help="/home/.../data/colorectal/fullres/colorectal")
parser.add_argument("--save_dir", default=None)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--sw_batch_size", type=int, default=1)
# Folds
parser.add_argument("--fold", type=int, default=0, help="Fold index [0-4]")
parser.add_argument(
    "--fold_lists_dir",
    default="/home/trawlins/tesis/data/colorectal/fold_lists",
    help="Folder with fold{n}_train.txt and fold{n}_val.txt",
)
parser.add_argument("--val_ratio_internal", type=float, default=0.1, help="Fraction of train IDs for internal val")
# ROI / preprocessing (keep defaults conservative; adjust if needed)
parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

# =====================
# Setup
# =====================
set_determinism(123)
os.environ["TQDM_DISABLE"] = os.getenv("TQDM_DISABLE", "1")

EXP_NAME = args.exp_name
DATA_DIR = args.data_dir
LOG_DIR = f"./logs/{EXP_NAME}"
BASE_SAVE = args.save_dir or os.path.join("./ckpts_seg", EXP_NAME)
FOLD_SAVE = os.path.join(BASE_SAVE, f"fold{args.fold}")
ROI_SIZE = list(map(int, args.roi))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FOLD_SAVE, exist_ok=True)

LOGFILE = os.path.join(LOG_DIR, f"trainer_log_fold{args.fold}.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

num_gpus = torch.cuda.device_count()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
max_epoch = args.epochs
batch_size = args.batch_size

logging.info(f"EXP_NAME={EXP_NAME}")
logging.info(f"data_dir={DATA_DIR}")
logging.info(f"fold={args.fold} | fold_lists_dir={args.fold_lists_dir}")
logging.info(f"save_dir={FOLD_SAVE}")
logging.info(f"device={device} | num_gpus={num_gpus} | epochs={max_epoch} | batch_size={batch_size} | sw_batch_size={args.sw_batch_size}")
logging.info(f"ROI_SIZE={ROI_SIZE}")

# =====================
# Helpers for folds & IO
# =====================

def _clean_id(x: str) -> str:
    x = os.path.basename(x)
    x = re.sub(r"_0000$", "", x)
    x = re.sub(r"\.nii(\.gz)?$", "", x)
    return x


def load_fold_ids(fold_lists_dir: str, fold: int) -> Tuple[List[str], List[str]]:
    p_tr = os.path.join(fold_lists_dir, f"fold{fold}_train.txt")
    p_va = os.path.join(fold_lists_dir, f"fold{fold}_val.txt")
    assert os.path.isfile(p_tr), f"No existe: {p_tr}"
    assert os.path.isfile(p_va), f"No existe: {p_va}"
    with open(p_tr) as f:
        train_ids = [l.strip() for l in f if l.strip()]
    with open(p_va) as f:
        holdout_ids = [l.strip() for l in f if l.strip()]
    return train_ids, holdout_ids


def split_internal(train_ids_all: List[str], val_ratio=0.1, seed=123) -> Tuple[List[str], List[str]]:
    import random

    random.seed(seed)
    idx = list(range(len(train_ids_all)))
    random.shuffle(idx)
    cut = int(len(idx) * (1 - val_ratio))
    train_ids = [train_ids_all[j] for j in idx[:cut]]
    val_ids = [train_ids_all[j] for j in idx[cut:]]
    return train_ids, val_ids


def build_sample_list_from_ids(data_dir: str, ids: List[str]) -> List[dict]:
    """
    Assumes layout like:
      data_dir/
        imagesTr/<ID>_0000.nii.gz
        labelsTr/<ID>.nii.gz   (or <ID>_seg.nii.gz)
    """
    im_dir = os.path.join(data_dir, "imagesTr")
    gt_dir = os.path.join(data_dir, "labelsTr")

    samples = []
    for cid in ids:
        img_p = os.path.join(im_dir, f"{cid}_0000.nii.gz")
        # try common variants for label
        cand_labels = [
            os.path.join(gt_dir, f"{cid}.nii.gz"),
            os.path.join(gt_dir, f"{cid}_gt.nii.gz"),
            os.path.join(gt_dir, f"{cid}_seg.nii.gz"),
        ]
        lab_p = next((p for p in cand_labels if os.path.isfile(p)), None)
        if not (os.path.isfile(img_p) and lab_p and os.path.isfile(lab_p)):
            logging.warning(f"Saltando ID {cid}: no se encontró imagen o label. img={img_p} label={lab_p}")
            continue
        samples.append({"image": img_p, "label": lab_p, "case_id": cid})
    return samples


# =====================
# Datasets & Transforms
# =====================
# (Ajusta normalización/spacing/orientation si tu prepro ya lo dejó homogéneo)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    # Orientationd(keys=["image", "label"], axcodes="RAS"),  # activa si lo necesitas
    # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    RandGaussianNoised(keys=["image"], prob=0.1),
    RandSpatialCropd(keys=["image", "label"], roi_size=ROI_SIZE, random_center=True, random_size=False),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    # Centro (o crops determinísticos si es necesario)
    RandSpatialCropd(keys=["image", "label"], roi_size=ROI_SIZE, random_center=False, random_size=False),
    EnsureTyped(keys=["image", "label"]),
])


# =====================
# Model & Trainer
# =====================
class ColorectalVesselsTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=5, num_gpus=1,
                 logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)

        self.window_infer = SlidingWindowInferer(roi_size=ROI_SIZE, sw_batch_size=args.sw_batch_size, overlap=0.5)
        self.augmentation = True

        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384]
        )
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpus)))

        self.patch_size = ROI_SIZE
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True
        )
        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        label = label[:, 0].long()  # canal 0 como clase
        return image, label

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)
        loss = self.cross(pred, label)
        return loss

    def cal_metric(self, gt, pred):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        else:
            return np.array([0.0, 50])

    def train_step(self, batch):
        self.model.train()
        image, label = batch["image"].to(self.device), batch["label"][:, 0].long().to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(image)
            loss = self.cross(logits, label)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        logging.info(f"[train] epoch={self.epoch + 1} step={self.global_step + 1} loss={loss.item():.5f}")
        return loss.detach()

    def validation_step(self, batch):
        self.model.eval()
        image, label = self.get_input(batch)
        image, label = image.to(self.device), label.to(self.device)
        with torch.no_grad():
            logits = self.model(image)
            preds = torch.argmax(logits, dim=1)
        preds_np = preds.detach().cpu().numpy().astype(np.uint8)
        labels_np = label.detach().cpu().numpy().astype(np.uint8)
        dices = []
        for p, g in zip(preds_np, labels_np):
            d = dice(p, g) if (p.sum() > 0 or g.sum() > 0) else 1.0
            if isinstance(d, np.ndarray):
                d = float(d)
            elif torch.is_tensor(d):
                d = float(d.detach().cpu().item())
            else:
                d = float(d)
            dices.append(d)
        return float(np.mean(dices)) if len(dices) > 0 else 0.0

    def validation_end(self, val_outputs):
        vals = []
        for v in val_outputs:
            if torch.is_tensor(v):
                vals.append(float(v.detach().cpu().item()))
            elif isinstance(v, np.ndarray):
                vals.append(float(v))
            elif isinstance(v, dict) and 'dice' in v:
                vals.append(float(v['dice']))
            else:
                vals.append(float(v))

        mean_dice = float(np.mean(vals)) if len(vals) > 0 else 0.0
        self.log("dice_vena", mean_dice, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)

        # Guardados
        best_path = os.path.join(FOLD_SAVE, "best_model.pt")
        final_path = os.path.join(FOLD_SAVE, f"final_model_{mean_dice:.4f}.pt")
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, best_path, delete_symbol="best_model")
        save_new_model_and_delete_last(self.model, final_path, delete_symbol="final_model")
        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(FOLD_SAVE, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))
        logging.info(f"[val] dice_vena={mean_dice:.4f} | mean_dice={mean_dice:.4f}")


# =====================
# Main
# =====================
if __name__ == "__main__":
    # 1) Leer IDs de fold y hacer split interno (no tocar hold-out aquí)
    train_ids_all, holdout_ids = load_fold_ids(args.fold_lists_dir, args.fold)
    train_ids, val_ids = split_internal(train_ids_all, args.val_ratio_internal, seed=123)
    logging.info(f"[Fold {args.fold}] train_in={len(train_ids)} | val_in={len(val_ids)} | holdout(no tocar)={len(holdout_ids)}")

    # 2) Construir listas de muestras y datasets
    train_samples = build_sample_list_from_ids(DATA_DIR, train_ids)
    val_samples = build_sample_list_from_ids(DATA_DIR, val_ids)

    # Usa CacheDataset si cabe en memoria; si no, Dataset normal
    use_cache = True
    if use_cache:
        train_ds = CacheDataset(data=train_samples, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
        val_ds = CacheDataset(data=val_samples, transform=val_transforms, cache_rate=1.0, num_workers=args.num_workers)
    else:
        train_ds = Dataset(data=train_samples, transform=train_transforms)
        val_ds = Dataset(data=val_samples, transform=val_transforms)

    # 3) Trainer
    trainer = ColorectalVesselsTrainer(
        env_type="pytorch",
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=LOG_DIR,
        val_every=5,
        num_gpus=num_gpus,
        master_port=17759,
        training_script=__file__,
    )

    # 4) Entrenar
    logging.info("Datasets cargados. Iniciando entrenamiento...")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
    logging.info("Entrenamiento finalizado.")
  
