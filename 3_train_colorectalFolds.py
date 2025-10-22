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
from torch.cuda.amp import GradScaler as LegacyGradScaler
from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    SpatialPadd,
    EnsureTyped,
)
from monai.data import Dataset, CacheDataset
from monai.inferers import SlidingWindowInferer

# --- MONAI seeding overflow hardening (prevents OverflowError: 2**32) ---
import numpy as _np
import monai.transforms.compose as _monai_comp
SAFE_MAX = (1 << 32) - 1  # 4294967295
_monai_comp.get_seed = lambda: 123

from contextlib import nullcontext

def autocast_ctx(dtype=torch.float16):
    if torch.cuda.is_available():
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=dtype)
        try:
            from torch.cuda.amp import autocast as autocast_legacy
            return autocast_legacy(dtype=dtype)
        except Exception:
            return nullcontext()
    else:
        return nullcontext()


def _safe_set_random_state(self, seed=None, state=None):
    try:
        base = int(seed) if seed is not None else 123
    except Exception:
        base = 123
    base = int(base) % SAFE_MAX
    rng = _np.random.RandomState(base)
    for _t in getattr(self, "transforms", []):
        try:
            child_seed = int(rng.randint(SAFE_MAX))
        except Exception:
            child_seed = 123
        try:
            _t.set_random_state(seed=child_seed)
        except Exception:
            pass
    return self

_monai_comp.Compose.set_random_state = _safe_set_random_state
# --- end hardening ---

from monai.transforms import RandCropByPosNegLabeld
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last

# =====================
# Args
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="colorectal_folds")
parser.add_argument("--data_dir", required=True, help="/home/.../data_nnUnet/nnUNet_raw/Dataset001_Colorectal")
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
# ROI / workers
parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
parser.add_argument("--num_workers", type=int, default=0)  # 0 para errores legibles
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
    Espera layout nnU-Net-like:
      data_dir/
        imagesTr/<ID>_0000.nii.gz
        labelsTr/<ID>.nii.gz (o variantes: _gt/_seg)
    """
    im_dir = os.path.join(data_dir, "imagesTr")
    gt_dir = os.path.join(data_dir, "labelsTr")

    samples = []
    for cid in ids:
        img_p = os.path.join(im_dir, f"{cid}_0000.nii.gz")
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
# Datasets & Transforms (MONAI)
# =====================
# Train: sampler foreground-aware (mejor para clases pequeñas)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),

    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, method="end"),
    RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    RandGaussianNoised(keys=["image"], prob=0.1),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=1, neg=1,
        num_samples=1,
        image_key="image",
        image_threshold=0,
    ),

    EnsureTyped(keys=["image", "label"]),
])

# Val: NO crops (validar en volumen completo con SW)
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, method="end"),
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
        self.writer = getattr(self, "writer", None)
        if self.writer is None:
            try:
                tb_dir = os.path.join(logdir, f"tb_fold{args.fold}")
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tb_dir)
            except Exception as e:
                self.writer = None
                logging.warning(f"No se pudo iniciar TensorBoard SummaryWriter: {e}")

        self.window_infer = SlidingWindowInferer(
            roi_size=ROI_SIZE, sw_batch_size=args.sw_batch_size, overlap=0.5
        )
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
        # Scheduler polinómico (si tu Trainer no lo crea por dentro)
        self.scheduler_type = "poly"

        # Pérdida (Dice+CE) — mejor para clases pequeñas
        from monai.losses import DiceCELoss
        self.loss_fn = DiceCELoss(
            to_onehot_y=True, softmax=True, ce_weight=None,
            include_background=False, lambda_dice=1.0, lambda_ce=1.0
        )

    # Entrenamiento con DataLoader de MONAI (sin batchgenerators)
    def fit_monai(self, train_ds, val_ds, num_workers: int = 0, val_every: int = 5):
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        self.model.to(self.device)
        scaler = getattr(self, "grad_scaler", None)
        if scaler is None:
            try:
                scaler = torch.amp.GradScaler('cuda')  # API nueva
            except Exception:
                from torch.cuda.amp import GradScaler   # fallback legacy
                scaler = GradScaler()
            self.grad_scaler = scaler

        global_step = 0
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.model.train()
            for batch in train_loader:
                image = batch["image"].to(self.device, non_blocking=True)
                label = batch["label"][:, 0].long().to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(dtype=torch.float16):
                    logits = self.model(image)
                    loss = self.loss_fn(logits, label)  # <<< FIX: usar la loss definida

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                global_step += 1
                self.global_step = global_step
                if (global_step % 25) == 0:
                    logging.info(f"[train] epoch={epoch+1}/{self.max_epochs} step={global_step} loss={loss.item():.5f}")

            # Scheduler por época (si existe)
            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()

            # Validación periódica — **volumen completo** con SW
            if ((epoch + 1) % val_every) == 0 or (epoch + 1) == self.max_epochs:
                self.model.eval()
                dices = []
                with torch.no_grad():
                    for batch in val_loader:
                        img = batch["image"].to(self.device, non_blocking=True)
                        lab = batch["label"][:, 0].long().to(self.device, non_blocking=True)

                        logits = self.window_infer(img, self.model)   # [B, C, D, H, W]
                        pred = torch.argmax(logits, dim=1)            # [B, D, H, W]

                        p_np = pred.detach().cpu().numpy().astype(np.uint8)
                        g_np = lab.detach().cpu().numpy().astype(np.uint8)
                        for p, g in zip(p_np, g_np):
                            d = dice(p, g) if (p.sum() > 0 or g.sum() > 0) else 1.0
                            if isinstance(d, np.ndarray):
                                d = float(d)
                            elif torch.is_tensor(d):
                                d = float(d.item())
                            dices.append(d)

                mean_dice = float(np.mean(dices)) if dices else 0.0
                if getattr(self, "writer", None) is not None:
                    self.writer.add_scalar("mean_dice_fullVol", mean_dice, epoch + 1)
                logging.info(f"[val-FULL] epoch={epoch+1} mean_dice={mean_dice:.4f}")

                best_path = os.path.join(FOLD_SAVE, "best_model.pt")
                final_path = os.path.join(FOLD_SAVE, f"final_model_{mean_dice:.4f}.pt")
                if mean_dice > self.best_mean_dice:
                    self.best_mean_dice = mean_dice
                    save_new_model_and_delete_last(self.model, best_path, delete_symbol="best_model")
                save_new_model_and_delete_last(self.model, final_path, delete_symbol="final_model")
                if ((epoch + 1) % 100) == 0:
                    torch.save(self.model.state_dict(), os.path.join(FOLD_SAVE, f"tmp_model_ep{epoch}_{mean_dice:.4f}.pt"))

# =====================
# Main
# =====================
if __name__ == "__main__":
    # 1) Leer IDs de fold y hacer split interno (no tocar hold-out aquí)
    train_ids_all, holdout_ids = load_fold_ids(args.fold_lists_dir, args.fold)
    train_ids, val_ids = split_internal(train_ids_all, args.val_ratio_internal, seed=123)
    logging.info(f"[Fold {args.fold}] train_in={len(train_ids)} | val_in={len(val_ids)} | holdout(no tocar)={len(holdout_ids)}")

    # 2) Construir listas de muestras y datasets MONAI
    train_samples = build_sample_list_from_ids(DATA_DIR, train_ids)
    val_samples = build_sample_list_from_ids(DATA_DIR, val_ids)

    logging.info(f"train_samples={len(train_samples)} | val_samples={len(val_samples)}")
    if len(train_samples) == 0:
        raise RuntimeError("No hay muestras de entrenamiento. Asegúrate de que --data_dir tenga imagesTr/<ID>_0000.nii.gz y labelsTr/<ID>.nii.gz y que los IDs del fold existan.")

    use_cache = True
    if use_cache:
        train_ds = CacheDataset(data=train_samples, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
        val_ds   = CacheDataset(data=val_samples,   transform=val_transforms,   cache_rate=1.0, num_workers=args.num_workers)
    else:
        train_ds = Dataset(data=train_samples, transform=train_transforms)
        val_ds   = Dataset(data=val_samples,   transform=val_transforms)

    # 3) Trainer (mismo optimizador/AMP/guardados)
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

    # 4) ENTRENAR con BYPASS (sin batchgenerators)
    logging.info("Datasets cargados. Iniciando entrenamiento...")
    trainer.fit_monai(train_ds, val_ds, num_workers=args.num_workers, val_every=5)
    logging.info("Entrenamiento finalizado.")
