import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss  # (no la usamos, pero la dejamos por paridad)
set_determinism(123)
import os
import argparse

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
val_every = 2
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
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
