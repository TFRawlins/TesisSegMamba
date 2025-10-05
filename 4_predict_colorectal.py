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
        import numpy as np
    
        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()
        model.to(args.device)
    
        # ===== 1) Forward SW: logits en espacio SW =====
        with torch.amp.autocast("cuda", enabled=True):
            logits_sw = predictor.maybe_mirror_and_predict(image, model, device=args.device)  # [B,C,Ds,Hs,Ws] o [C,Ds,Hs,Ws]
    
        if logits_sw.dim() == 4:  # [C,D,H,W] -> [1,C,D,H,W]
            logits_sw = logits_sw.unsqueeze(0)
    
        B, C, Ds, Hs, Ws = logits_sw.shape
    
        # ===== 2) RUTA ROI (métrica en 192^3, "como training") =====
        if label is not None:
            target_shape = tuple(label.shape[-3:])  # (192,192,192)
            logits_roi = logits_sw
            if tuple(logits_roi.shape[-3:]) != target_shape:
                logits_roi = F.interpolate(
                    logits_roi, size=target_shape, mode="trilinear", align_corners=False
                )  # [1,C,192,192,192]
            probs_roi = torch.softmax(logits_roi, dim=1)  # [1,C,192,192,192]
            pred_roi  = probs_roi.argmax(dim=1)          # [1,192,192,192]
    
            gt_roi = label[0, 0].detach().cpu().numpy().astype(np.uint8)
            pr_roi = pred_roi[0].detach().cpu().numpy().astype(np.uint8)
    
            if pr_roi.shape != gt_roi.shape:  # fallback (no debería ocurrir)
                pr_t = torch.from_numpy(pr_roi)[None, None].float()
                pr_t = F.interpolate(pr_t, size=gt_roi.shape, mode="nearest")
                pr_roi = pr_t.squeeze().byte().numpy()
    
            print(f"[ROI] Dice clase 1: {dice(pr_roi, gt_roi):.4f}")
            print("ROI shapes -> probs:", tuple(probs_roi.shape),
                  "pred_roi:", tuple(pred_roi.shape),
                  "label:", tuple(label.shape))
            print("pos_pred:", int((pred_roi[0] > 0).sum()),
                  "pos_gt:", int((label[0,0] > 0).sum()))
        else:
            pred_roi = None  # opcional
    
        # ===== 3) RUTA SW (para NON-CROP): usar predict_raw_probability EN SW =====
        # Nota: este método te devuelve mapas en el grid SW (no 192^3).
        # Acepta batch o sin batch según tu Predictor; para seguridad, pásale sin batch.
        probs_sw_nc = predictor.predict_raw_probability(
            logits_sw.squeeze(0), properties=properties
        )  # esperado: [C,Ds,Hs,Ws] en SW
    
        # Sanity/debug:
        if isinstance(probs_sw_nc, torch.Tensor):
            print("SW for non-crop (torch) shape:", tuple(probs_sw_nc.shape))
            pred_sw_nc = probs_sw_nc.argmax(dim=0)  # [Ds,Hs,Ws]
            # one-hot SW -> [2,Ds,Hs,Ws]
            pred_onehot_sw = F.one_hot(pred_sw_nc.long(), num_classes=2).permute(3, 0, 1, 2).float()
            pred_onehot_sw_np = pred_onehot_sw.detach().cpu().numpy()  # (2,Ds,Hs,Ws)
        else:
            print("SW for non-crop (numpy) shape:", tuple(probs_sw_nc.shape))
            pred_sw_nc = np.argmax(probs_sw_nc, axis=0)  # [Ds,Hs,Ws]
            # one-hot SW -> (2,Ds,Hs,Ws)
            pred_onehot_sw_np = np.eye(2, dtype=np.float32)[pred_sw_nc]  # (Ds,Hs,Ws,2)
            pred_onehot_sw_np = np.transpose(pred_onehot_sw_np, (3, 0, 1, 2))  # (2,Ds,Hs,Ws)
    
        # Seguridad extra: forzar tamaño SW si hiciera falta (debería coincidir)
        if pred_onehot_sw_np.shape[-3:] != (Ds, Hs, Ws):
            from scipy.ndimage import zoom
            # nearest por ser máscara
            zoom_factors = (
                1.0,
                Ds / pred_onehot_sw_np.shape[1],
                Hs / pred_onehot_sw_np.shape[2],
                Ws / pred_onehot_sw_np.shape[3],
            )
            pred_onehot_sw_np = zoom(pred_onehot_sw_np, zoom_factors, order=0)
    
        # ===== 4) NON-CROP en espacio SW =====
        fullres_onehot = predictor.predict_noncrop_probability(pred_onehot_sw_np, properties)
    
        # ===== 5) Argmax full-res y guardar NIfTI =====
        if isinstance(fullres_onehot, torch.Tensor):
            fullres_label = fullres_onehot.argmax(dim=0, keepdim=True).detach().cpu().numpy()  # [1,Z,Y,X]
        else:
            fullres_label = np.argmax(fullres_onehot, axis=0, keepdims=True)                   # [1,Z,Y,X]
    
        out_np = fullres_label[0].astype(np.uint8)  # [Z,Y,X]
    
        predictor.save_to_nii(
            out_np,
            raw_spacing=[1, 1, 1],  # ideal: usa spacing/origin/direction reales si están en properties
            case_name=properties["name"][0],
            save_dir=args.save_dir,
        )
    
        return 0




if __name__ == "__main__":
    trainer = ColorectalPredict(device=args.device)
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    trainer.validation_single_gpu(test_ds)
