import os
import argparse
import sys
import torch
import numpy as np

from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, EnsureTyped, Lambdad
)

from light_training.trainer import Trainer
from light_training.prediction import Predictor
from light_training.evaluation.metric import dice

# ======= Setup =======
set_determinism(123)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Ruta a nnUNet_raw/Dataset001_Colorectal")
parser.add_argument("--ckpt", required=True, help="Ruta a best_model.pt (o final_model_*.pt)")
parser.add_argument("--save_dir", required=True, help="Directorio base de salida para predicciones")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--roi", type=int, nargs=3, default=[128,128,128])
parser.add_argument("--sw_batch_size", type=int, default=2)
parser.add_argument("--overlap", type=float, default=0.5)
parser.add_argument("--mirror_axes", type=int, nargs="*", default=[0,1,2])

# Folds
parser.add_argument("--fold", type=int, default=0, help="Fold index [0..4]")
parser.add_argument("--fold_lists_dir", required=True, help="Carpeta con fold{n}_train.txt y fold{n}_val.txt")
args = parser.parse_args()

# ======= Utils =======
def _load_holdout_ids(fold_lists_dir: str, fold: int):
    p = os.path.join(fold_lists_dir, f"fold{fold}_val.txt")
    assert os.path.isfile(p), f"No existe {p}"
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]

def _build_samples(data_dir: str, ids):
    imdir = os.path.join(data_dir, "imagesTr")
    lbdir = os.path.join(data_dir, "labelsTr")
    samples = []
    for cid in ids:
        img = os.path.join(imdir, f"{cid}_0000.nii.gz")
        lab_candidates = [
            os.path.join(lbdir, f"{cid}.nii.gz"),
            os.path.join(lbdir, f"{cid}_gt.nii.gz"),
            os.path.join(lbdir, f"{cid}_seg.nii.gz"),
        ]
        lab = next((p for p in lab_candidates if os.path.isfile(p)), None)
        if os.path.isfile(img) and lab and os.path.isfile(lab):
            samples.append({"image": img, "label": lab, "case_id": cid})
        else:
            print(f"[WARN] Saltando {cid}: faltan paths (img={img}, label?={lab})", file=sys.stderr)
    return samples

# ======= Predictor Trainer =======
class ColorectalPredict(Trainer):
    def __init__(self, device="cuda:0"):
        super().__init__(
            env_type="pytorch", max_epochs=1, batch_size=1, device=device,
            val_every=1, num_gpus=1, logdir="", master_ip="localhost",
            master_port=17750, training_script=__file__
        )
        self.patch_size = args.roi
        self.augmentation = False  # sin augment en inferencia

    def get_input(self, batch):
        # Estructura alineada con light_training: data/seg/properties
        return batch["data"], batch.get("seg", None), batch["properties"]

    def define_model(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[2,2,2,2],
            feat_size=[48, 96, 192, 384]
        )

        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # normalizar nombres (remover "module.")
        if isinstance(sd, dict):
            sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

        res = model.load_state_dict(sd, strict=False)
        print(f"[CKPT] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
        if len(res.missing_keys) < 15:
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

        os.makedirs(os.path.join(args.save_dir, f"fold{args.fold}"), exist_ok=True)
        return model, predictor

    @torch.no_grad()
    def validation_step(self, batch):
        import torch.nn.functional as F
        import nibabel as nib
        import numpy as _np
    
        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()
        model.to(args.device)
    
        # case_id
        case_name = properties["name"][0] if isinstance(properties["name"], list) else str(properties["name"])
        print(f"\n[CASE] {case_name}")
    
        # Forward SW + mirror
        x = image.float()
        print("[INPUT] shape:", tuple(x.shape),
              "min:", float(x.min()), "max:", float(x.max()),
              "mean:", float(x.mean()), "std:", float(x.std()))
    
        with torch.amp.autocast(device_type="cuda", enabled=("cuda" in args.device)):
            logits_sw = predictor.maybe_mirror_and_predict(image, model, device=args.device)
        if logits_sw.dim() == 4:  # [C,D,H,W] -> [1,C,D,H,W]
            logits_sw = logits_sw.unsqueeze(0)
    
        # Ajuste a shape del label si existe (para calcular métricas en el mismo ROI)
        if label is not None:
            target_shape = tuple(label.shape[-3:])
        else:
            target_shape = tuple(logits_sw.shape[-3:])
    
        if tuple(logits_sw.shape[-3:]) != target_shape:
            logits_sw = F.interpolate(logits_sw, size=target_shape, mode="trilinear", align_corners=False)
    
        probs = torch.softmax(logits_sw, dim=1)
        pred = probs.argmax(dim=1)  # [1,D,H,W]
    
        # Métricas rápidas (Dice binario en ROI) - opcional, para logging
        if label is not None:
            gt = (label[0,0] > 0).to(torch.uint8).cpu().numpy()
            pr = pred[0].to(torch.uint8).cpu().numpy()
            d = float(dice(pr, gt)) if (gt.sum() > 0 or pr.sum() > 0) else 1.0
            print(f"[ROI] Dice clase 1 = {d:.4f}  (pos_pred={int(pr.sum())}, pos_gt={int(gt.sum())})")
    
        # ---- Guardado NIfTI con affine/header reales ----
        out_np = pred[0].cpu().numpy().astype(np.uint8)
        out_dir = os.path.join(args.save_dir, f"fold{args.fold}")
        os.makedirs(out_dir, exist_ok=True)
    
        # Tomamos como referencia el label si está, si no la imagen
        ref_path = properties.get("label_path") or properties.get("img_path")
        affine, header = None, None
        try:
            if ref_path:
                ref_nii = nib.load(ref_path)
                affine, header = ref_nii.affine, ref_nii.header
        except Exception as e:
            print(f"[WARN] no se pudo leer referencia '{ref_path}': {e} -> uso identidad")
    
        if affine is None:
            affine = _np.eye(4)
    
        out_nii = nib.Nifti1Image(out_np, affine=affine, header=header)
        save_path = os.path.join(out_dir, f"{case_name}.nii.gz")
        nib.save(out_nii, save_path)
        print(f"[SAVE] {save_path}")
    
        return 0


# ======= Main =======
if __name__ == "__main__":
    # 1) IDs del hold-out del fold
    holdout_ids = _load_holdout_ids(args.fold_lists_dir, args.fold)
    print(f"[FOLD {args.fold}] holdout_ids: {len(holdout_ids)}")

    # 2) Samples
    samples = _build_samples(args.data_dir, holdout_ids)
    assert len(samples) > 0, "No hay muestras en hold-out. Revisa rutas e IDs."

    # 3) Transforms (sin augmentations) + renombre de claves a data/seg/properties
    tf = Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image","label"]),
        Lambdad(keys=["image","label","case_id"], func=lambda x: x),  # no-op (para asegurar presencia)
        # Renombrar/inyectar propiedades para que get_input funcione sin tocar Trainer base
        Lambdad(
            keys=["image","label","case_id"],
            func=lambda x: x
        ),
    ])

    # MONAI no renombra claves; construimos Dataset y en __getitem__ transformamos a data/seg/properties
    class _WrapDataset(Dataset):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            data = item["image"]
            seg  = item["label"]
            props = {
                "name": item.get("case_id", f"case_{index}"),
                "img_path": item["image_meta_dict"]["filename_or_obj"],
                "label_path": item["label_meta_dict"]["filename_or_obj"],
            }
            return {"data": data, "seg": seg, "properties": props}

    base_ds = Dataset(data=samples, transform=tf)
    ds = _WrapDataset(data=base_ds.data, transform=base_ds.transform)
    
    predictor_trainer = ColorectalPredict(device=args.device)
    predictor_trainer.validation_single_gpu(ds)

