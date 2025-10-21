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
        res = model.load_state_dict(sd, strict=False)
        print(f"[CKPT] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
        if len(res.missing_keys) < 15:  # imprime algunas para inspección rápida
            print("[CKPT] missing keys sample:", res.missing_keys[:10])
        if len(res.unexpected_keys) < 15:
            print("[CKPT] unexpected keys sample:", res.unexpected_keys[:10])
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
        import numpy as np
    
        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()
        model.to(args.device)
    
        # Identificador del caso
        case_name = properties["name"][0] if isinstance(properties["name"], list) else str(properties["name"])
        print(f"[CASE] {case_name}")
    
        # ===== 1) Forward (SW) =====
        x = image.float()
        print("[INPUT] shape:", tuple(x.shape),
              "min:", float(x.min()), "max:", float(x.max()),
              "mean:", float(x.mean()), "std:", float(x.std()))
        with torch.amp.autocast("cuda", enabled=True):
            logits_sw = predictor.maybe_mirror_and_predict(image, model, device=args.device)  # [B,C,Ds,Hs,Ws] o [C,Ds,Hs,Ws]
        if logits_sw.dim() == 4:
            logits_sw = logits_sw.unsqueeze(0)  # -> [1,C,Ds,Hs,Ws]
    
        print("[LOGITS_SW] shape:", tuple(logits_sw.shape))
    
        # ===== 2) Llevar a ROI (192^3) para métrica y guardado =====
        if label is not None:
            target_shape = tuple(label.shape[-3:])  # (192,192,192)
        else:
            target_shape = tuple(logits_sw.shape[-3:])
    
        logits_roi = logits_sw
        if tuple(logits_roi.shape[-3:]) != target_shape:
            logits_roi = F.interpolate(
                logits_roi, size=target_shape, mode="trilinear", align_corners=False
            )  # [1,C,192,192,192]
    
        print("[LOGITS_ROI] shape:", tuple(logits_roi.shape))
    
        probs_roi = torch.softmax(logits_roi, dim=1)
        p = probs_roi[0]  # [C,192,192,192]
        print("[PROBS] per-class mean:",
              [float(p[c].mean()) for c in range(p.shape[0])],
              "max:", float(p.max()), "min:", float(p.min()))
        pred_roi = probs_roi.argmax(dim=1)            # [1,192,192,192]
    
        # ===== 3) Métrica en ROI (como training) =====
        if label is not None:
            gt_roi_raw = label[0, 0].detach().cpu().numpy().astype(np.uint8)
            gt_roi = (gt_roi_raw == 1).astype(np.uint8)
            pr_roi = pred_roi[0].detach().cpu().numpy().astype(np.uint8)
            gt_vals, gt_counts = np.unique(gt_roi, return_counts=True)
            pr_vals, pr_counts = np.unique(pr_roi, return_counts=True)
            print("[UNIQUE] GT:", dict(zip(gt_vals.tolist(), gt_counts.tolist())),
                  "PR:", dict(zip(pr_vals.tolist(), pr_counts.tolist())))
            
            tp = int(np.logical_and(pr_roi == 1, gt_roi == 1).sum())
            fp = int(np.logical_and(pr_roi == 1, gt_roi == 0).sum())
            fn = int(np.logical_and(pr_roi == 0, gt_roi == 1).sum())
            den = (2*tp + fp + fn)
            dice_manual = (2*tp / den) if den > 0 else 1.0
            print(f"[MANUAL] TP={tp} FP={fp} FN={fn}  Dice={dice_manual:.4f}")
            fg_probs = probs_roi[0, 1].detach().cpu().numpy()[gt_roi == 1]
            prop_gt_pos_over_05 = float((fg_probs > 0.5).mean()) if fg_probs.size else -1
            print(f"[ROI] GT positives with p1>0.5: {prop_gt_pos_over_05:.3f}")
            dice_bg = dice(1 - pr_roi, 1 - gt_roi)
            print(f"[ROI] Dice background: {dice_bg:.4f}")
    
            # Probabilidad de la clase 1 en voxeles positivos del GT
            fg_probs = probs_roi[0, 1].detach().cpu().numpy()[gt_roi == 1]
            print("[ROI] fg_prob mean/max on GT==1:",
                  float(fg_probs.mean() if fg_probs.size else -1),
                  float(fg_probs.max() if fg_probs.size else -1))
    
            # Fallback (no debería hacer falta)
            if pr_roi.shape != gt_roi.shape:
                pr_t = torch.from_numpy(pr_roi)[None, None].float()
                pr_t = F.interpolate(pr_t, size=gt_roi.shape, mode="nearest")
                pr_roi = pr_t.squeeze().byte().numpy()
    
            print(f"[ROI] Dice clase 1: {dice(pr_roi, gt_roi):.4f}")
            print("ROI shapes -> probs:", tuple(probs_roi.shape),
                  "pred_roi:", tuple(pred_roi.shape),
                  "label:", tuple(label.shape))
            print("pos_pred:", int((pred_roi[0] > 0).sum()),
                  "pos_gt:", int((label[0,0] > 0).sum()))
    
        # ===== 4) Guardar NIfTI directamente en ROI (sin non-crop) =====
        out_np = pred_roi[0].detach().cpu().numpy().astype(np.uint8)  # [192,192,192]
        predictor.save_to_nii(
            out_np,
            raw_spacing=[1, 1, 1],  # si guardaste spacing/origin/direction reales en properties, mejor úsalos
            case_name=case_name,
            save_dir=args.save_dir,
        )
    
        return 0






if __name__ == "__main__":
    trainer = ColorectalPredict(device=args.device)
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    trainer.validation_single_gpu(test_ds)
