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
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Ruta a data/fullres/<dataset>")
parser.add_argument("--ckpt", required=True, help="Ruta al best_model.pt")
parser.add_argument("--save_dir", default="/home/trawlins/tesis/prediction_results/segmamba")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--roi", type=int, nargs=3, default=[128,128,128])
parser.add_argument("--sw_batch_size", type=int, default=1)
parser.add_argument("--overlap", type=float, default=0.5)
parser.add_argument("--mirror_axes", type=int, nargs="*", default=[0,1,2])
args = parser.parse_args()

class ColorectalPredict(Trainer):
    def __init__(self, device="cuda:0"):
        super().__init__(
            env_type="pytorch", max_epochs=1, batch_size=1, device=device,
            val_every=1, num_gpus=1, logdir="", master_ip="localhost",
            master_port=17750, training_script=__file__
        )
        self.patch_size = args.roi
        self.augmentation = False

    def get_input(self, batch):
        # batch["data"] -> (B, 1, D, H, W); batch["seg"] puede venir o no en test
        return batch["data"], batch.get("seg", None), batch["properties"]

    def define_model(self):
        from model_segmamba.segmamba import SegMamba

        model = SegMamba(
            in_chans=1,      # CT de 1 canal
            out_chans=2,     # binario (fondo/lesión). Si fue 1-logit, cambia lógica abajo.
            depths=[2,2,2,2],
            feat_size=[48, 96, 192, 384]
        )
        sd = torch.load(args.ckpt, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "module" in sd:
            sd = sd["module"]
        sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
        model.load_state_dict(sd, strict=False)
        model.eval()

        inferer = SlidingWindowInferer(
            roi_size=args.roi,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            progress=True,
            mode="gaussian",
        )
        predictor = Predictor(window_infer=inferer, mirror_axes=args.mirror_axes)

        os.makedirs(args.save_dir, exist_ok=True)
        return model, predictor

    @torch.no_grad()
    def validation_step(self, batch):
        import torch.nn.functional as F
    
        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()
        model.to(args.device)
    
        # ===== 1) Forward SW: logits con batch =====
        with torch.amp.autocast("cuda", enabled=True):
            # logits_sw: [B,C,Ds,Hs,Ws] en el espacio del sliding window (crop)
            logits_sw = predictor.maybe_mirror_and_predict(image, model, device=args.device)
    
        # Normalizar a [1,C,*,*,*]
        if logits_sw.dim() == 4:   # [C,D,H,W] -> [1,C,D,H,W]
            logits_sw = logits_sw.unsqueeze(0)
    
        # ===== 2) Probs en espacio SW (para NON-CROP) =====
        # NO re-escalar aquí: justo así es como lo espera predict_noncrop_probability
        probs_sw = torch.softmax(logits_sw, dim=1)            # [1,C,Ds,Hs,Ws]
        pred_sw  = probs_sw.argmax(dim=1, keepdim=False)      # [1,Ds,Hs,Ws]
        # One-hot [2,Ds,Hs,Ws] para non-crop
        pred_onehot_sw = F.one_hot(
            pred_sw.long().squeeze(0), num_classes=2
        ).permute(3, 0, 1, 2).float()                         # [2,Ds,Hs,Ws]
    
        # ===== 3) Métrica en ROI (192^3): re-escalar a la rejilla del label =====
        if label is not None:
            target_shape = tuple(label.shape[-3:])            # (192,192,192)
            logits_roi = logits_sw
            if tuple(logits_roi.shape[-3:]) != target_shape:
                logits_roi = F.interpolate(
                    logits_roi, size=target_shape, mode="trilinear", align_corners=False
                )                                             # [1,C,192,192,192]
            probs_roi = torch.softmax(logits_roi, dim=1)      # [1,C,192,192,192]
            pred_roi  = probs_roi.argmax(dim=1)               # [1,192,192,192]
    
            gt_roi = label[0, 0].detach().cpu().numpy().astype(np.uint8)
            pr_roi = pred_roi[0].detach().cpu().numpy().astype(np.uint8)
            # (Por seguridad: deberían coincidir ya)
            if pr_roi.shape != gt_roi.shape:
                pr_t = torch.from_numpy(pr_roi)[None, None].float()
                pr_t = F.interpolate(pr_t, size=gt_roi.shape, mode="nearest")
                pr_roi = pr_t.squeeze().byte().numpy()
    
            print(f"[ROI] Dice clase 1: {dice(pr_roi, gt_roi):.4f}")
            # Debug útil
            print("ROI shapes -> probs:", tuple(probs_roi.shape),
                  "pred_roi:", tuple(pred_roi.shape),
                  "label:", tuple(label.shape))
            print("pos_pred:", int((pred_roi[0] > 0).sum()),
                  "pos_gt:", int((label[0,0] > 0).sum()))
        else:
            # Si no hay GT, crea pred_roi desde SW re-escalado a 192^3 solo para consistencia (opcional)
            pred_roi = None
    
        # ===== 4) NON-CROP en espacio SW (forma chica) =====
        fullres_onehot = predictor.predict_noncrop_probability(pred_onehot_sw, properties)
        # Argmax full-res y convertir a numpy para save_to_nii
        if isinstance(fullres_onehot, torch.Tensor):
            fullres_label = fullres_onehot.argmax(dim=0, keepdim=True).detach().cpu().numpy()  # [1,Z,Y,X]
        else:
            fullres_label = np.argmax(fullres_onehot, axis=0, keepdims=True)                   # [1,Z,Y,X]
    
        out_np = fullres_label[0].astype(np.uint8)  # [Z,Y,X]
    
        # ===== 5) Guardar NIfTI =====
        predictor.save_to_nii(
            out_np,
            raw_spacing=[1, 1, 1],  # si tienes spacing/origin/direction en properties, es mejor usarlos aquí
            case_name=properties["name"][0],
            save_dir=args.save_dir,
        )
    
        return 0


if __name__ == "__main__":
    trainer = ColorectalPredict(device=args.device)
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    trainer.validation_single_gpu(test_ds)
