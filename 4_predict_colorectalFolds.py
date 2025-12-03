import os
import argparse
import sys
import torch
import numpy as np
import logging
import pickle

from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from monai.data import Dataset as MonaiDataset
from monai.transforms import Compose, EnsureTyped, SpatialPadd, Lambdad

from light_training.trainer import Trainer
from light_training.prediction import Predictor
from light_training.evaluation.metric import dice

# ======= Setup =======
set_determinism(123)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Carpeta FULLRES con {ID}.npy, {ID}_seg.npy, {ID}.pkl/.npz")
parser.add_argument("--ckpt", required=True, help="Ruta a best_model.pt (o final_model_*.pt)")
parser.add_argument("--save_dir", required=True, help="Directorio base de salida para predicciones")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
parser.add_argument("--sw_batch_size", type=int, default=2)
parser.add_argument("--overlap", type=float, default=0.5)
parser.add_argument("--mirror_axes", type=int, nargs="*", default=[])
# Folds
parser.add_argument("--fold", type=int, default=0, help="Fold index [0..4]")
parser.add_argument("--fold_lists_dir", required=True, help="Carpeta con fold{n}_train.txt y fold{n}_val.txt")
parser.add_argument("--prob_thresh", type=float, default=0.40)       # compat
parser.add_argument("--empty_fallback_q", type=float, default=0.999)  # compat
args = parser.parse_args()

# ======= Utils =======
def _load_holdout_ids(fold_lists_dir: str, fold: int):
    p = os.path.join(fold_lists_dir, f"fold{fold}_val.txt")
    assert os.path.isfile(p), f"No existe {p}"
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]

def _find_meta_paths(data_dir: str, cid: str):
    pkl = os.path.join(data_dir, f"{cid}.pkl")
    npz = os.path.join(data_dir, f"{cid}.npz")
    img_ref = ""
    lbl_ref = ""
    try:
        if os.path.isfile(pkl):
            with open(pkl, "rb") as f:
                d = pickle.load(f)
            for k in ["image_path", "img_path", "raw_image", "source_image", "image", "img"]:
                if k in d and isinstance(d[k], (str, list)):
                    img_ref = d[k][0] if isinstance(d[k], list) and d[k] else d[k]
                    break
            for k in ["label_path", "raw_label", "source_label", "label", "seg"]:
                if k in d and isinstance(d[k], (str, list)):
                    lbl_ref = d[k][0] if isinstance(d[k], list) and d[k] else d[k]
                    break
        elif os.path.isfile(npz):
            d = dict(np.load(npz, allow_pickle=True))
            def _take(v):
                if isinstance(v, np.ndarray):
                    try:
                        return v.item()
                    except Exception:
                        pass
                return v
            for k in ["image_path", "img_path", "raw_image", "source_image", "image", "img"]:
                if k in d:
                    img_ref = _take(d[k]); break
            for k in ["label_path", "raw_label", "source_label", "label", "seg"]:
                if k in d:
                    lbl_ref = _take(d[k]); break
    except Exception as e:
        logging.warning(f"[meta] {cid}: no se pudo leer pkl/npz ({e})")
    return str(img_ref or ""), str(lbl_ref or "")

def _build_ids_existing(data_dir: str, ids):
    ok = []
    for cid in ids:
        if os.path.isfile(os.path.join(data_dir, f"{cid}.npy")):
            ok.append(cid)
        else:
            print(f"[WARN] Saltando {cid}: falta {cid}.npy en {data_dir}", file=sys.stderr)
    return ok

def _fit_to_shape(arr, target):
    """
    Recorte/padding centrado para alinear pred (arr) a la forma target (Z,Y,X).
    """
    z = np.zeros(target, dtype=arr.dtype)
    s_in = [0, 0, 0]
    s_out = [0, 0, 0]
    e_in = list(arr.shape)
    for i in range(3):
        if arr.shape[i] >= target[i]:
            start = (arr.shape[i] - target[i]) // 2
            s_in[i] = start
            e_in[i] = start + target[i]
            s_out[i] = 0
        else:
            start = (target[i] - arr.shape[i]) // 2
            s_out[i] = start
            e_in[i] = arr.shape[i]
    z[s_out[0]:s_out[0] + (e_in[0] - s_in[0]),
      s_out[1]:s_out[1] + (e_in[1] - s_in[1]),
      s_out[2]:s_out[2] + (e_in[2] - s_in[2])] = arr[s_in[0]:e_in[0], s_in[1]:e_in[1], s_in[2]:e_in[2]]
    return z

# ======= Dataset FULLRES =======
class FullresPredDataset(MonaiDataset):
    """
    Devuelve dict con:
      image: np.float32 (1,D,H,W)
      label: opcional np.int64 (1,D,H,W) si existe {ID}_seg.npy (para métrica rápida)
      properties: dict con 'name', 'img_ref', 'lbl_ref'
    """
    def __init__(self, ids, data_dir, transform=None):
        self.ids = ids
        self.data_dir = data_dir
        super().__init__(data=[], transform=transform)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        cid = self.ids[index]
        img_p = os.path.join(self.data_dir, f"{cid}.npy")
        seg_p = os.path.join(self.data_dir, f"{cid}_seg.npy")
        if not os.path.isfile(img_p):
            raise FileNotFoundError(f"Falta imagen para {cid}: {img_p}")
        img = np.load(img_p)  # (Z,Y,X) o (C,Z,Y,X)
        if img.ndim == 3:
            img = img[None, ...]
        img = img.astype(np.float32, copy=False)

        sample = {"image": img, "case_id": cid}

        if os.path.isfile(seg_p):
            seg = np.load(seg_p)
            if seg.ndim == 3:
                seg = seg[None, ...]
            sample["label"] = (seg > 0).astype(np.int64, copy=False)

        img_ref, lbl_ref = _find_meta_paths(self.data_dir, cid)
        sample["properties"] = {"name": cid, "img_ref": img_ref, "lbl_ref": lbl_ref}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

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
        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(in_chans=1, out_chans=2, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {(k[7:] if isinstance(k, str) and k.startswith("module.") else k): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(args.device)

        self.inferer = SlidingWindowInferer(
            roi_size=args.roi, sw_batch_size=args.sw_batch_size,
            overlap=args.overlap, mode="gaussian"
        )
        self.predictor = Predictor(window_infer=self.inferer, mirror_axes=args.mirror_axes)

    def get_input(self, batch):
        img = batch["image"]          # [1,1,D,H,W] tras collate
        seg = batch.get("label", None)
        props = batch["properties"]
        return img, seg, props

    def define_model(self):
        os.makedirs(os.path.join(args.save_dir, f"fold{args.fold}"), exist_ok=True)
        return self.model, self.predictor

    @torch.no_grad()
    def validation_step(self, batch):
        import torch.nn.functional as F
        import nibabel as nib

        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()

        # case_id
        case_name = properties.get("name", "case_0")
        case_name = case_name[0] if isinstance(case_name, list) else case_name
        case_name = str(case_name)
        print(f"\n[CASE] {case_name}")

        x = image.float()
        print("[INPUT] shape:", tuple(x.shape),
              "min:", float(x.min()), "max:", float(x.max()),
              "mean:", float(x.mean()), "std:", float(x.std()))

        with torch.amp.autocast(device_type="cuda", enabled=("cuda" in args.device)):
            logits_sw = predictor.maybe_mirror_and_predict(image, model, device=args.device)
        if logits_sw.dim() == 4:  # [C,D,H,W] -> [1,C,D,H,W]
            logits_sw = logits_sw.unsqueeze(0)

        # Ajuste a shape del label si existe (para métrica rápida en el mismo ROI padded)
        if label is not None:
            target_shape = tuple(label.shape[-3:])
        else:
            target_shape = tuple(logits_sw.shape[-3:])
        if tuple(logits_sw.shape[-3:]) != target_shape:
            logits_sw = F.interpolate(logits_sw, size=target_shape, mode="trilinear", align_corners=False)

        pred = torch.argmax(logits_sw, dim=1).to(torch.uint8)   # [1,D,H,W]

        # --- Métrica rápida (ROI padded) ---
        if label is not None:
            gt_roi = (label[0, 0] > 0).to(torch.uint8).cpu().numpy()
            pr_roi = pred[0].to(torch.uint8).cpu().numpy()
            d = float(dice(pr_roi, gt_roi)) if (gt_roi.sum() > 0 or pr_roi.sum() > 0) else 1.0
            print(f"[ROI] Dice clase 1 = {d:.4f}  (pos_pred={int(pr_roi.sum())}, pos_gt={int(gt_roi.sum())})")

        # ---- Guardados (alineado a **GT en disco** si existe) ----
        out_np = pred[0].cpu().numpy().astype(np.uint8)   # (D,H,W)
        out_dir = os.path.join(args.save_dir, f"fold{args.fold}")
        os.makedirs(out_dir, exist_ok=True)

        # Si hay GT en disco (fullres), alineamos pred a su forma (quita pad / aplica pad centrado)
        gt_disk_path = os.path.join(args.data_dir, f"{case_name}_seg.npy")
        if os.path.isfile(gt_disk_path):
            gt_disk = np.load(gt_disk_path)
            if gt_disk.ndim == 4:  # (1,D,H,W)
                gt_disk = gt_disk[0]
            if out_np.shape != gt_disk.shape:
                out_np = _fit_to_shape(out_np, gt_disk.shape)

        # 1) Guardar .npy ya alineado a GT-disk (si existía)
        npy_path = os.path.join(out_dir, f"{case_name}_pred.npy")
        np.save(npy_path, out_np)
        print(f"[SAVE] {npy_path}")

        # 2) Intentar NIfTI (si tenemos referencia legible); si no, identidad
        img_ref = properties.get("img_ref", "") or ""
        lbl_ref = properties.get("lbl_ref", "") or ""
        ref_path = str(img_ref) if isinstance(img_ref, str) else ""
        if not ref_path and isinstance(lbl_ref, str):
            ref_path = lbl_ref
        try:
            if ref_path and os.path.isfile(ref_path):
                ref_nii = nib.load(ref_path)
                affine, header = ref_nii.affine, ref_nii.header
            else:
                affine, header = np.eye(4), None
            out_nii = nib.Nifti1Image(out_np, affine=affine, header=header)
            nii_path = os.path.join(out_dir, f"{case_name}.nii.gz")
            nib.save(out_nii, nii_path)
            print(f"[SAVE] {nii_path} (ref={ref_path if ref_path else 'identity'})")
        except Exception as e:
            print(f"[WARN] No se pudo guardar NIfTI para {case_name}: {e}")

        return 0

# ======= Main =======
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 1) IDs del hold-out del fold
    holdout_ids = _load_holdout_ids(args.fold_lists_dir, args.fold)
    print(f"[FOLD {args.fold}] holdout_ids: {len(holdout_ids)}")

    # 2) Filtrar por existencia en FULLRES
    ids = _build_ids_existing(args.data_dir, holdout_ids)
    assert len(ids) > 0, "No hay muestras en hold-out. Revisa rutas e IDs."

    # 3) Transforms mínimos (sin augmentations). Permitir ausencia de label.
    tf = Compose([
        EnsureTyped(keys=["image", "label"], dtype=("float32", "int64"), allow_missing_keys=True),
        Lambdad(keys=["label"], func=lambda x: (x > 0).astype(x.dtype), allow_missing_keys=True),
        SpatialPadd(keys=["image", "label"], spatial_size=tuple(args.roi), method="symmetric", allow_missing_keys=True),
    ])

    ds = FullresPredDataset(ids=ids, data_dir=args.data_dir, transform=tf)

    predictor_trainer = ColorectalPredict(device=args.device)
    # El Trainer construye su propio DataLoader; le pasamos el dataset directamente
    predictor_trainer.validation_single_gpu(ds)
