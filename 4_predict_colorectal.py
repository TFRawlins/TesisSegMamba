import os
import argparse
import torch
import numpy as np
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.prediction import Predictor
from light_training.evaluation.metric import dice

set_determinism(123)

# --- Args para alinearlo con tu 3_train_colorectal.py ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Ruta a data/fullres/<tu_dataset>")
parser.add_argument("--ckpt", required=True, help="Ruta al .pt entrenado en colorectal")
parser.add_argument("--save_dir", default="./prediction_results/segmamba", help="Dónde guardar los .nii")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--roi", type=int, nargs=3, default=[128,128,128])
parser.add_argument("--sw_batch_size", type=int, default=1)   # mantén 1 si te quedaste sin VRAM
parser.add_argument("--overlap", type=float, default=0.5)
parser.add_argument("--mirror_axes", type=int, nargs="*", default=[0,1,2])
args = parser.parse_args()

# --- Trainer minimal que replica el flujo original pero sin lógicas BraTS ---
class ColorectalPredict(Trainer):
    def __init__(self, device="cuda:0"):
        super().__init__(env_type="pytorch", max_epochs=1, batch_size=1, device=device, val_every=1,
                         num_gpus=1, logdir="", master_ip="localhost", master_port=17750,
                         training_script=__file__)
        self.patch_size = args.roi
        self.augmentation = False

    def get_input(self, batch):
        # batch["data"] -> (B, 1, D, H, W), batch["seg"] -> (B, 1, D, H, W)  (si test tiene GT)
        return batch["data"], batch.get("seg", None), batch["properties"]

    def define_model(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(
            in_chans=1,          # CT
            out_chans=2,         # fondo / vaso-tumor
            depths=[2,2,2,2],
            feat_size=[48, 96, 192, 384]
        )
        sd = torch.load(args.ckpt, map_location="cpu")
        if "module" in sd:
            sd = sd["module"]
        # quitar prefijo "module." si existe
        new_sd = { (k[7:] if k.startswith("module") else k): v for k, v in sd.items() }
        model.load_state_dict(new_sd, strict=True)
        model.eval()

        inferer = SlidingWindowInferer(
            roi_size=args.roi,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            progress=True,
            mode="gaussian"
        )
        predictor = Predictor(window_infer=inferer, mirror_axes=args.mirror_axes)
        os.makedirs(args.save_dir, exist_ok=True)
        return model, predictor

    @torch.no_grad()
    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()

        # logits en (C, D, H, W) tras fusion de ventanas
        logits = predictor.maybe_mirror_and_predict(image, model, device=args.device)
        # Probabilidades recombinadas y "des-crop" a espacio original
        probs = predictor.predict_raw_probability(logits, properties=properties)
        pred = probs.argmax(dim=0, keepdim=True)  # (1, D, H, W) con {0,1}

        # Métrica opcional (si hay GT en test_ds)
        if label is not None:
            gt = label[0,0].cpu().numpy().astype(np.uint8)
            pr = pred[0].cpu().numpy().astype(np.uint8)
            print(f"Dice clase 1: {dice(pr, gt):.4f}")

        # Volver a tamaño original y guardar .nii
        fullres = predictor.predict_noncrop_probability(pred, properties)  # (1, D, H, W)
        predictor.save_to_nii(
            fullres, raw_spacing=[1,1,1],           # usa spacing real si lo guardaste en properties
            case_name=properties["name"][0],
            save_dir=args.save_dir
        )
        return 0

if __name__ == "__main__":
    trainer = ColorectalPredict(device=args.device)
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    trainer.validation_single_gpu(test_ds)
