import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from light_training.trainer import Trainer
from model_segmamba.segmamba import SegMamba
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

class LiverTrainer(Trainer):
    def __init__(self, data_dir, save_dir="./ckpts_seg", max_epochs=300, batch_size=1):
        super().__init__(
            env_type="pytorch",
            max_epochs=max_epochs,
            batch_size=batch_size,
            device="cuda:0",
            val_every=1,
            num_gpus=1,
            logdir=save_dir
        )
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Modelo para imágenes CT de 1 canal y 2 clases (fondo, hígado)
        self.model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[24, 48, 96, 192]
        )
        self.model.to(self.device)
        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=20, max_epochs=self.max_epochs)
        self.inferer = SlidingWindowInferer(
            roi_size=[64, 64, 64],
            sw_batch_size=1, 
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_metric = 0
        self.best_metric_epoch = -1

        # Datos
        train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)
        
        def custom_collate_fn(batch):
            return {
                "data": torch.stack([x["data"] for x in batch]),
                "seg": torch.stack([x["seg"] for x in batch]),
                "properties": [x["properties"] for x in batch]
            }
        
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True)
        self.test_loader = DataLoader(test_ds, batch_size=1, pin_memory=True)
    
    import gc

    def train_step(self, batch):

        data = batch["data"].to(self.device, non_blocking=True)
        label = batch["seg"].to(self.device, non_blocking=True)
        label = (label > 0).long()  # asegúrate de tener clases válidas (0,1)
        print("📦 Model device:", next(self.model.parameters()).device)
        print("📤 Data device:", data.device)
        print("🎯 Label device:", label.device)
        with torch.cuda.amp.autocast():  # 🔁 Mixed precision
            logits = self.model(data)
            print(f"✅ Model inference done, logits shape: {logits.shape}")
            loss = self.loss(logits, label)
            print(f"✅ Loss computed: {loss.item():.4f}")
    
        print("⬅️  Haciendo backward...")
        self.scaler.scale(loss).backward()
        print("✅ backward() hecho")        # backward
        self.scaler.step(self.optimizer)           # optimizer step
        self.scaler.update()                       # update scaler
        self.optimizer.zero_grad()
    
        return loss



    def validation_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        label = batch["seg"].to(self.device, non_blocking=True)
        
        if label.dim() == 4:
            label = label.unsqueeze(1)
    
        label = label.long()
        label = (label > 0).long()
        with torch.no_grad():
            logits = self.inferer(data, self.model)
            preds = torch.argmax(logits, dim=1)
            dice_value = dice(preds.cpu().numpy(), label.squeeze(1).cpu().numpy())
    
        return dice_value


    def cal_metric(self, gt, pred):
        if pred.sum() > 0 and gt.sum() > 0:
            return np.array([dice(pred, gt), 0])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 0])
        else:
            return np.array([0.0, 0])

    def run(self):
        print("Comenzando entrenamiento...\n")
        for epoch in range(self.max_epochs):
            self.model.train()
            losses = []

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            print(f"📚 Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_loss:.4f}")

            # Validación
            if (epoch + 1) % self.val_every == 0:
                self.model.eval()
                dices = []

                for batch in self.val_loader:
                    dice_val = self.validation_step(batch)
                    dices.append(dice_val)

                avg_dice = np.mean(dices)
                print(f"🎯 Validation Dice: {avg_dice:.4f}")

                # Guardar el mejor modelo
                if avg_dice > self.best_metric:
                    self.best_metric = avg_dice
                    self.best_metric_epoch = epoch + 1
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
                    print(f"✅ Nuevo mejor modelo guardado (Epoch {self.best_metric_epoch})")
            print(f"💾 Epoch {epoch + 1} completado, avg loss: {avg_loss:.4f}")

        print(f"\n🏁 Entrenamiento finalizado. Mejor Dice: {self.best_metric:.4f} en epoch {self.best_metric_epoch}")

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

