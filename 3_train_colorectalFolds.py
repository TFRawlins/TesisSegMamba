import os
import sys
import json
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    EnsureTyped,
    CropForegroundd,
    SpatialPadd,
    RandFlipd,
    RandAffined,
    RandCropByPosNegLabeld,
    Lambdad,
)
from monai.data import Dataset as MonaiDataset, CacheDataset, list_data_collate
from monai.inferers import SlidingWindowInferer

# --- MONAI seeding overflow hardening (previene OverflowError: 2**32) ---
import numpy as _np
import monai.transforms.compose as _monai_comp
SAFE_MAX = (1 << 32) - 1
_monai_comp.get_seed = lambda: 123

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

from light_training.trainer import Trainer
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last

# =====================
# Args (SIN agregar nuevos)
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="colorectal_folds")
parser.add_argument("--data_dir", required=True, help="/home/.../data/colorectal/fullres/colorectal")
parser.add_argument("--save_dir", default=None)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--sw_batch_size", type=int, default=1)
# Folds
parser.add_argument("--fold", type=int, default=0, help="Fold index [0-4]")
parser.add_argument(
    "--fold_lists_dir",
    required=True,
    help="Carpeta con fold{n}_train.txt y fold{n}_val.txt"
)
parser.add_argument("--val_ratio_internal", type=float, default=0.1, help="Porción del train para validación interna")
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
# Folds & Dataset (FULLRES .npy)
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

class FullresArrayDataset(Dataset):
    """
    Lee {ID}.npy (imagen) y {ID}_seg.npy (label) desde data_dir.
    Espera que la imagen esté ya reorientada/remuestreada por el preprocesado.
    Devuelve dict con keys 'image' (float32) y 'label' (int64, con canal 1).
    """
    def __init__(self, ids: List[str], data_dir: str):
        self.ids = ids
        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        img_p = os.path.join(self.data_dir, f"{cid}.npy")
        seg_p = os.path.join(self.data_dir, f"{cid}_seg.npy")
        if not (os.path.isfile(img_p) and os.path.isfile(seg_p)):
            raise FileNotFoundError(f"[fullres] Falta para ID {cid}: {img_p} / {seg_p}")

        img = np.load(img_p)  # (Z,Y,X) o (C,Z,Y,X)
        seg = np.load(seg_p)  # (Z,Y,X) o (C,Z,Y,X)

        if img.ndim == 3:  # (Z,Y,X) -> (1,Z,Y,X)
            img = img[None, ...]
        if seg.ndim == 3:
            seg = seg[None, ...]
        # Tipos
        img = img.astype(np.float32, copy=False)
        seg = seg.astype(np.int64,  copy=False)

        return {"image": img, "label": seg, "case_id": cid}

def filter_existing_ids(ids: List[str], data_dir: str) -> List[str]:
    ok = []
    for cid in ids:
        if os.path.isfile(os.path.join(data_dir, f"{cid}.npy")) and \
           os.path.isfile(os.path.join(data_dir, f"{cid}_seg.npy")):
            ok.append(cid)
        else:
            logging.warning(f"[skip] ID {cid}: faltan .npy en {data_dir}")
    return ok

# =====================
# Transforms (SIN re-remuestrear)
# =====================

train_transforms = Compose([
    EnsureTyped(keys=["image", "label"], dtype=("float32", "int64")),

    # *** Fuerza label binario: cualquier valor >0 pasa a 1
    Lambdad(keys=["label"], func=lambda x: (x > 0).astype(x.dtype)),

    #CropForegroundd(keys=["image", "label"], source_key="image", margin=8),
    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, method="symmetric"),

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandAffined(
        keys=["image", "label"], prob=0.2,
        rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest")
    ),

    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=ROI_SIZE, pos=1, neg=1, num_samples=2, image_key="image"
    ),
])

val_transforms = Compose([
    EnsureTyped(keys=["image", "label"], dtype=("float32", "int64")),

    # *** Igual en validación
    Lambdad(keys=["label"], func=lambda x: (x > 0).astype(x.dtype)),

    #CropForegroundd(keys=["image", "label"], source_key="image", margin=8),
    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE, method="symmetric"),
])


# =====================
# Model & Trainer
# =====================
class ColorectalVesselsTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=5, num_gpus=1,
                 logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)

        # TensorBoard
        self.writer = getattr(self, "writer", None)
        if self.writer is None:
            try:
                tb_dir = os.path.join(logdir, f"tb_fold{args.fold}")
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tb_dir)
            except Exception as e:
                self.writer = None
                logging.warning(f"No se pudo iniciar TensorBoard SummaryWriter: {e}")

        # Inferencia deslizante con mayor solape (mejor en bordes finos)
        self.window_infer = SlidingWindowInferer(
            roi_size=ROI_SIZE, sw_batch_size=args.sw_batch_size, overlap=0.75
        )

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

        from monai.losses import DiceCELoss
        self.loss_fn = DiceCELoss(
            to_onehot_y=True, softmax=True,
            include_background=False, lambda_dice=1.0, lambda_ce=1.0
        )

    def fit_monai(self, train_ds, val_ds, num_workers: int = 0, val_every: int = 5):
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, collate_fn=list_data_collate
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True, collate_fn=list_data_collate
        )

        self.model.to(self.device)

        # GradScaler (AMP)
        try:
            scaler = torch.amp.GradScaler('cuda')
        except Exception:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        self.grad_scaler = scaler

        global_step = 0
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.model.train()

            for batch in train_loader:
                image = batch["image"].to(self.device, non_blocking=True)  # [B,1,D,H,W]
                label = batch["label"].long().to(self.device, non_blocking=True)  # [B,1,D,H,W]

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits = self.model(image)           # [B,2,D,H,W]
                    loss = self.loss_fn(logits, label)   # Dice+CE

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                global_step += 1
                self.global_step = global_step
                if (global_step % 25) == 0:
                    logging.info(f"[train] epoch={epoch+1}/{self.max_epochs} step={global_step} loss={loss.item():.5f}")

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()

            # Validación periódica — volumen completo con SWI
            if ((epoch + 1) % val_every) == 0 or (epoch + 1) == self.max_epochs:
                self.model.eval()
                dices = []
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    for batch in val_loader:
                        img = batch["image"].to(self.device, non_blocking=True)     # [1,1,D,H,W]
                        lab = batch["label"].long().to(self.device, non_blocking=True)  # [1,1,D,H,W]

                        logits = self.window_infer(img, self.model)   # [1,2,D,H,W]
                        pred = torch.argmax(logits, dim=1)            # [1,D,H,W]

                        p_np = pred.detach().cpu().numpy().astype(np.uint8)
                        g_np = lab[:, 0].detach().cpu().numpy().astype(np.uint8)
                        for p, g in zip(p_np, g_np):
                            d = dice(p, g) if (p.sum() > 0 or g.sum() > 0) else 1.0
                            d = float(d if not torch.is_tensor(d) else d.item())
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
    # 1) Leer IDs de fold y hacer split interno (no tocar hold-out)
    train_ids_all, holdout_ids = load_fold_ids(args.fold_lists_dir, args.fold)
    train_ids, val_ids = split_internal(train_ids_all, args.val_ratio_internal, seed=123)

    # 2) Filtrar IDs que existan en data_dir como .npy
    train_ids = filter_existing_ids(train_ids, DATA_DIR)
    val_ids   = filter_existing_ids(val_ids,   DATA_DIR)
    logging.info(f"[Fold {args.fold}] train_in={len(train_ids)} | val_in={len(val_ids)} | holdout(no tocar)={len(holdout_ids)}")

    if len(train_ids) == 0:
        raise RuntimeError(
            f"No hay muestras de entrenamiento en {DATA_DIR}. "
            f"Se esperan archivos {{ID}}.npy y {{ID}}_seg.npy por cada ID de los folds."
        )

    # 3) Datasets
    train_base = FullresArrayDataset(train_ids, DATA_DIR)
    val_base   = FullresArrayDataset(val_ids,   DATA_DIR)

    use_cache = True
    if use_cache:
        train_ds = CacheDataset(
            data=[train_base[i] for i in range(len(train_base))],
            transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers
        )
        val_ds   = CacheDataset(
            data=[val_base[i] for i in range(len(val_base))],
            transform=val_transforms,   cache_rate=1.0, num_workers=args.num_workers
        )
    else:
        # Alternativa sin cache
        train_ds = MonaiDataset(data=[train_base[i] for i in range(len(train_base))], transform=train_transforms)
        val_ds   = MonaiDataset(data=[val_base[i]   for i in range(len(val_base))],   transform=val_transforms)

    # 4) Trainer
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

    logging.info("Datasets cargados. Iniciando entrenamiento...")
    trainer.fit_monai(train_ds, val_ds, num_workers=args.num_workers, val_every=5)
    logging.info("Entrenamiento finalizado.")
