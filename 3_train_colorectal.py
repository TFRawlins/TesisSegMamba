import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.utils.files_helper import save_new_model_and_delete_last

# ---------- CONFIG FIJA (sin argumentos) ----------
EXP_NAME = "colorectal"

# RUTAS ABSOLUTAS QUE USAS TÚ
DATA_DIR = "/home/trawlins/tesis/data/colorectal/fullres/colorectal"
LOG_DIR = "/home/trawlins/tesis/logs/colorectal"
MODEL_SAVE_DIR = "/home/trawlins/tesis/ckpts_seg/colorectal/model"

# HYPERPARAMS
MAX_EPOCHS = 200
BATCH_SIZE = 8
VAL_EVERY = 5
NUM_GPUS = 1
DEVICE = "cuda:0"
ROI_SIZE = [128, 128, 128]
SW_BATCH_SIZE = 4
AUGMENTATION = True

# ---------- PREP ----------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
torch.set_float32_matmul_precision("medium")
set_determinism(123)

class ColorectalVesselsTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=5, num_gpus=1,
                 logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)

        self.window_infer = SlidingWindowInferer(roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, overlap=0.5)
        self.augmentation = AUGMENTATION

        # === Modelo ===
        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
        )
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpus)))

        self.patch_size = ROI_SIZE
        self.best_mean_dice = 0.0

        # Optimizador / Scheduler
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True
        )
        self.scheduler_type = "poly"

        # Pérdida
        self.cross = nn.CrossEntropyLoss()

    def get_input(self, batch):
        image = batch["data"]   # (B, 1, D, H, W)
        label = batch["seg"]    # (B, 1, D, H, W)
        label = label[:, 0].long()  # CE espera (B, D, H, W)
        return image, label

    def training_step(self, batch):
        image, label = self.get_input(batch)
        image, label = image.to(self.device), label.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(image)          # (B, 2, D, H, W)
            loss = self.cross(logits, label)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.log("training_loss", loss.detach(), step=self.global_step)
        return loss.detach()

    def validation_step(self, batch):
        self.model.eval()
        image, label = self.get_input(batch)
        image, label = image.to(self.device), label.to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            preds = torch.argmax(logits, dim=1)  # (B, D, H, W)

        preds_np = preds.detach().cpu().numpy().astype(np.uint8)
        labels_np = label.detach().cpu().numpy().astype(np.uint8)

        dices = []
        for p, g in zip(preds_np, labels_np):
            d = dice(p, g) if (p.sum() > 0 or g.sum() > 0) else 1.0
            # fuerza float puro
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

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(MODEL_SAVE_DIR, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model"
            )

        save_new_model_and_delete_last(
            self.model,
            os.path.join(MODEL_SAVE_DIR, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model"
        )

        if (self.epoch + 1) % 100 == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(MODEL_SAVE_DIR, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt")
            )

        print(f"[val] dice_vena={mean_dice:.4f} | mean_dice={mean_dice:.4f}")

if __name__ == "__main__":
    trainer = ColorectalVesselsTrainer(
        env_type="pytorch",
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        logdir=LOG_DIR,
        val_every=VAL_EVERY,
        num_gpus=NUM_GPUS,
        master_port=17759,
        training_script=__file__,
    )

    # DATA_DIR es ruta fija arriba
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(DATA_DIR)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
