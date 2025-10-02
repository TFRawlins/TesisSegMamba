import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
from monai.transforms import Compose, RandSpatialCropd, CenterSpatialCropd, SpatialPadd, EnsureChannelFirstd, Lambda
from torch.utils.data import Dataset

set_determinism(123)
import os
import argparse

ROI_TGT = (128, 128, 128)  # (D, H, W)

def crop_pad_center(arr, target=ROI_TGT):

    assert arr.ndim == 4, f"Esperaba (C,D,H,W), llegó {arr.shape}"
    C, D, H, W = arr.shape
    tD, tH, tW = target

    def _slice(sz, tgt):
        if sz <= tgt:
            return slice(0, sz)
        start = (sz - tgt) // 2
        return slice(start, start + tgt)

    sD, sH, sW = _slice(D, tD), _slice(H, tH), _slice(W, tW)
    arr = arr[:, sD, sH, sW]
    
    C, D2, H2, W2 = arr.shape
    pD1 = (tD - D2) // 2 if D2 < tD else 0
    pD2 = tD - D2 - pD1 if D2 < tD else 0
    pH1 = (tH - H2) // 2 if H2 < tH else 0
    pH2 = tH - H2 - pH1 if H2 < tH else 0
    pW1 = (tW - W2) // 2 if W2 < tW else 0
    pW2 = tW - W2 - pW1 if W2 < tW else 0

    if pD1 or pD2 or pH1 or pH2 or pW1 or pW2:
        arr = np.pad(arr,
                     ((0, 0), (pD1, pD2), (pH1, pH2), (pW1, pW2)),
                     mode="constant", constant_values=0)
    return arr

class ROIWrapper(Dataset):
    """
    Envuelve el dataset base (items dict con 'data','seg', opcionalmente 'properties')
    y garantiza:
      - canal primero (C,D,H,W)
      - tamaño fijo ROI_TGT mediante crop+pad centrado
      - tipos: data float32, seg long
      - incluye 'properties' para satisfacer el dataloader
    """
    def __init__(self, base, roi=ROI_TGT):
        self.base = base
        self.roi = roi

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        data = item["data"]
        seg  = item["seg"]
        props_in = item.get("properties", {})
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
        if data.ndim == 3:  # (D,H,W) -> (1,D,H,W)
            data = data[None, ...]
        if seg.ndim == 3:
            seg = seg[None, ...]

        orig_shape_data = tuple(data.shape)
        orig_shape_seg  = tuple(seg.shape)

        data = crop_pad_center(data, self.roi)
        seg  = crop_pad_center(seg,  self.roi)
        data = torch.from_numpy(data).float()
        seg  = torch.from_numpy(seg).long()
        properties = dict(props_in)
        properties.setdefault("original_data_shape", orig_shape_data)
        properties.setdefault("original_seg_shape",  orig_shape_seg)
        properties["roi_target"] = tuple(self.roi)
        properties["final_data_shape"] = tuple(data.shape)
        properties["final_seg_shape"]  = tuple(seg.shape)

        return {"data": data, "seg": seg, "properties": properties}


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="colorectal")
parser.add_argument("--data_dir", required=True)
parser.add_argument("--save_dir", default=None)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--sw_batch_size", type=int, default=1)
args = parser.parse_args()

EXP_NAME = args.exp_name
data_dir = args.data_dir
logdir = f"./logs/{EXP_NAME}"
model_save_path = args.save_dir or os.path.join("./ckpts_seg", EXP_NAME, "model")

os.makedirs(logdir, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
augmentation = True

env = "pytorch"
max_epoch = 200
batch_size = 2
val_every = 5
num_gpus = 2
device = "cuda:0"
roi_size = [128, 128, 128]

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

class ColorectalVesselsTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=2,
                 logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)

        self.window_infer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.5)
        self.augmentation = augmentation

        # === cambio mínimo: 1 canal de entrada, 2 clases (fondo, vena) ===
        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384]
        )
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpus)))
                     
        self.patch_size = roi_size
        self.best_mean_dice = 0.0

        # mismas elecciones que el original
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True
        )
        self.scheduler_type = "poly"

        # === pérdida: CE como en el original (mismo espíritu del paper) ===
        self.cross = nn.CrossEntropyLoss()

    def get_input(self, batch):
        # Mantener las mismas claves del dataloader original
        image = batch["data"]   # (B, 1, D, H, W)
        label = batch["seg"]    # (B, 1, D, H, W) con {0=fondo, 1=vena}
        label = label[:, 0].long()  # CE espera (B, D, H, W) con enteros
        return image, label

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)               # (B, 2, D, H, W)
        loss = self.cross(pred, label)         # CE 2 clases
        self.log("training_loss", loss, step=self.global_step)
        return loss

    def cal_metric(self, gt, pred):
        # gt/pred booleanos o {0,1}; devolvemos Dice del canal "vena"
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        else:
            return np.array([0.0, 50])

    def train_step(self, batch):
        self.model.train()
        image, label = batch["data"].to(self.device), batch["seg"][:, 0].long().to(self.device)
    
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(image)          # (B, 2, D, H, W)
            loss = self.cross(logits, label)    # CE 2 clases
    
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
    
        preds_np = preds.cpu().numpy()
        labels_np = label.cpu().numpy()
    
        dices = []
        for p, g in zip(preds_np, labels_np):
            d = dice(p, g) if (p.sum() > 0 or g.sum() > 0) else 1.0
            dices.append(d)
    
        return np.mean(dices)
            
    def validation_end(self, val_outputs):

        mean_dice = float(np.mean(val_outputs))
    
        self.log("dice_vena", mean_dice, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)
    
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model"
            )
    
        save_new_model_and_delete_last(
            self.model,
            os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model"
        )
    
        if (self.epoch + 1) % 100 == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt")
            )
    
        print(f"[val] dice_vena={mean_dice:.4f} | mean_dice={mean_dice:.4f}")


if __name__ == "__main__":
    trainer = ColorectalVesselsTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=5,
        num_gpus=num_gpus,
        master_port=17759,
        training_script=__file__
    )

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)
    train_ds = ROIWrapper(train_ds, roi=ROI_TGT)
    val_ds   = ROIWrapper(val_ds,   roi=ROI_TGT)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
