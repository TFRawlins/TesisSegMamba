import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from light_training.trainer import Trainer
from model_segmamba.segmamba import SegMamba
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", "train.log")

logger = logging.getLogger()  # root logger
logger.setLevel(logging.INFO)

# Rotating file handler (evita archivos enormes)
file_handler = RotatingFileHandler(log_path, maxBytes=50*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(file_fmt)
logger.addHandler(file_handler)

# Stream handler para consola (opcional)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(file_fmt)
logger.addHandler(stream_handler)

# Ejemplo: usar logger.info en vez de print
logger.info(f"📄 Logging en: {log_path}")

torch.backends.cudnn.benchmark = True  # mejor rendimiento en 3D con input size estable

def gpu_summary():
    if not torch.cuda.is_available():
        print("CUDA no disponible; se usará CPU.")
        return 0
    n = torch.cuda.device_count()
    print(f"GPUs detectadas: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  • GPU {i}: {props.name} | VRAM: {props.total_memory/1024**3:.1f} GB")
    return n

class LiverTrainer(Trainer):
    def __init__(self, data_dir, save_dir="./ckpts_seg", max_epochs=300, batch_size=2, sw_batch_size=2):
        n_gpus = gpu_summary()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(
            env_type="pytorch",
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=str(device),
            val_every=5,
            num_gpus=max(1, n_gpus),
            logdir=save_dir
        )
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Modelo para 1 canal (CT) y 2 clases
        base_model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[24, 48, 96, 192]
        ).to(device)

        if n_gpus > 1:
            print("Activando DataParallel en [0, 1] (usa ambas RTX TITAN)")
            self.model = torch.nn.DataParallel(base_model, device_ids=list(range(n_gpus)))
        else:
            self.model = base_model

        self.device = device
        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, warmup_epochs=20, max_epochs=self.max_epochs
        )

        # ↑ subimos el sliding window batch para validar más rápido con 2 GPUs
        ROI = (128, 128, 64)
        self.inferer = SlidingWindowInferer(
            roi_size=list(ROI),
            sw_batch_size=sw_batch_size,
            overlap=0.25,
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_metric = 0.0
        self.best_metric_epoch = -1

        # Datos
        train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

        def custom_collate_fn(batch):
            return {
                "data": torch.stack([x["data"] for x in batch]),
                "seg": torch.stack([x["seg"] for x in batch]),
                "properties": [x["properties"] for x in batch],
            }

        # más workers si hay más GPUs
        num_workers = max(4, 2 * max(1, n_gpus))
        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=custom_collate_fn, pin_memory=True, num_workers=num_workers
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=1, pin_memory=True, num_workers=max(2, n_gpus)
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=1, pin_memory=True, num_workers=max(2, n_gpus)
        )

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        data = batch["data"].to(self.device, non_blocking=True)
        label = batch["seg"].to(self.device, non_blocking=True)
        label = (label > 0).long()

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = self.model(data)
            loss = self.loss(logits, label)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        label = batch["seg"].to(self.device, non_blocking=True)
        if label.dim() == 4:
            label = label.unsqueeze(1)
        label = (label > 0).long()

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = self.inferer(data, self.model)
        preds = torch.argmax(logits, dim=1)
        dice_value = dice(preds.cpu().numpy(), label.squeeze(1).cpu().numpy())
        return dice_value

    def run(self):
        print(" Comenzando entrenamiento...\n")
        logger.info(" Comenzando entrenamiento...\n")
        global_step = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            losses = []

            for batch in self.train_loader:
                #print(batch)
                loss = self.train_step(batch)
                losses.append(loss.item())

                # self.scheduler.step()
                global_step += 1

            # Por-epoch step (recomendado para Cosine con warmup por epoch)
            self.scheduler.step()

            avg_loss = float(np.mean(losses)) if losses else 0.0
            print(f"- Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_loss:.4f}")
            logger.info(f"- Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_loss:.4f}")

            # Validación
            if (epoch + 1) % self.val_every == 0:
                self.model.eval()
                dices = [self.validation_step(b) for b in self.val_loader]
                avg_dice = float(np.mean(dices)) if dices else 0.0
                print(f" Validation Dice: {avg_dice:.4f}")
                logger.info(f" Validation Dice: {avg_dice:.4f}")

                # Guardar el mejor modelo
                if avg_dice > self.best_metric:
                    self.best_metric = avg_dice
                    self.best_metric_epoch = epoch + 1
                    torch.save(
                        (self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model).state_dict(),
                        os.path.join(self.save_dir, "best_model.pt")
                    )
                    print(f"Nuevo mejor modelo guardado (Epoch {self.best_metric_epoch})")
                    logger.info(f"Nuevo mejor modelo guardado (Epoch {self.best_metric_epoch})")

            print(f"Epoch {epoch + 1} completado, avg loss: {avg_loss:.4f}")
            logger.info(f"Epoch {epoch + 1} completado, avg loss: {avg_loss:.4f}")

        print(f"\nEntrenamiento finalizado. Mejor Dice: {self.best_metric:.4f} en epoch {self.best_metric_epoch}")
        logger.info(f"\nEntrenamiento finalizado. Mejor Dice: {self.best_metric:.4f} en epoch {self.best_metric_epoch}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Ruta a datos preprocesados")
    parser.add_argument("--save_dir", type=str, default="/home/trawlins/tesis/ckpts_seg", help="Dónde guardar el modelo")
    parser.add_argument("--epochs", type=int, default=300, help="Número de epochs (por defecto 300)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size de entrenamiento")
    parser.add_argument("--sw_batch_size", type=int, default=2, help="Sliding window batch size para validación")
    parser.add_argument("--test_run", action="store_true", help="Ejecuta un smoke test de 10 epochs")
    args = parser.parse_args()

    # Smoke test para verificar que no se caiga
    if args.test_run:
        print("Modo test activado: forzando epochs=10 para smoke test.")
        args.epochs = 10
        args.save_dir = os.path.join(args.save_dir, "smoketest")

    trainer = LiverTrainer(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        sw_batch_size=args.sw_batch_size,
    )
    trainer.run()

if __name__ == "__main__":
    main()
