import os
import argparse
import torch
import numpy as np
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from light_training.trainer import Trainer
from model_segmamba.segmamba import SegMamba
from datautils.build import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from utils.schedulers import LinearWarmupCosineAnnealingLR

class LiverTrainer(Trainer):
    def __init__(self, data_dir, save_dir="./ckpts_seg", max_epochs=400, batch_size=2):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Modelo para imágenes CT de 1 canal y 2 clases (fondo, hígado)
        self.model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384]
        )

        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=20, max_epochs=self.max_epochs)
        self.inferer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.5)

        self.best_metric = 0
        self.best_metric_epoch = -1

        # Datos
        self.train_loader, self.val_loader, self.test_loader = get_train_val_test_loader_from_train(
            data_dir, batch_size=self.batch_size, fold=0
        )

    def train_step(self, batch):
        data, label = batch["image"], batch["label"]
        label = label[:, 0].long()  # Elimina canal extra
        logits = self.model(data)
        loss = self.loss(logits, label)
        return loss

    def validation_step(self, batch):
        data, label = batch["image"], batch["label"]
        label = label[:, 0].long()

        with torch.no_grad():
            logits = self.inferer(data, self.model)
            preds = torch.argmax(logits, dim=1)
            dice_value = dice(preds.cpu().numpy(), label.cpu().numpy())
            return dice_value

    def test_step(self, batch):
        return self.validation_step(batch)

    def cal_metric(self, gt, pred):
        if pred.sum() > 0 and gt.sum() > 0:
            return np.array([dice(pred, gt), 0])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 0])
        else:
            return np.array([0.0, 0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta a datos preprocesados")
    parser.add_argument("--save_dir", type=str, default="./ckpts_seg", help="Donde guardar el modelo")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    trainer = LiverTrainer(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size
    )
    trainer.run()

if __name__ == "__main__":
    main()

