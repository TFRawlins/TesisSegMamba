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
import os
os.environ['TQDM_DISABLE'] = os.getenv('TQDM_DISABLE','1')
import logging, sys
from torch.cuda.amp import autocast

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

LOGFILE = os.path.join(logdir, "trainer_log.txt")
class _StreamToLogger(object):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        if message and message != "\n":
            for line in message.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logdir, "trainer_log.txt"), mode="w"),
        logging.StreamHandler(sys.stdout)
    ],
)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

logging.info(f"EXP_NAME={EXP_NAME}")
logging.info(f"data_dir={data_dir}")
logging.info(f"model_save_path={model_save_path}")

augmentation = True

env = "pytorch"
max_epoch = args.epochs
batch_size = args.batch_size
val_every = 5
num_gpus = torch.cuda.device_count()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
roi_size = [128, 128, 128]
logging.info(f"device={device} | num_gpus={num_gpus} | epochs={max_epoch} | batch_size={batch_size} | sw_batch_size={args.sw_batch_size}")


class ColorectalVesselsTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=5, num_gpus=2,
                 logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)

        self.window_infer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=args.sw_batch_size, overlap=0.5)
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
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True
        )
        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()

    def get_input(self, batch):
        image = batch["data"] 
        label = batch["seg"]
        label = label[:, 0].long()
        return image, label

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)               # (B, 2, D, H, W)
        loss = self.cross(pred, label)         # CE 2 clases
        # self.log("training_loss", loss, step=self.global_step)
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
        image, label = batch["data"].to(self.device), batch["seg"][:, 0].long().to(self.device)
    
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(image)          # (B, 2, D, H, W)
            loss = self.cross(logits, label)    # CE 2 clases
    
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # self.log("training_loss", loss.detach(), step=self.global_step)
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
    
        logging.info(f"[val] dice_vena={mean_dice:.4f} | mean_dice={mean_dice:.4f}")


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
    logging.info("Datasets cargados. Iniciando entrenamiento...")
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
    logging.info("Entrenamiento finalizado.")
