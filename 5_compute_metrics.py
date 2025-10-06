#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from medpy import metric
from monai.utils import set_determinism
from light_training.dataloading.dataset import get_train_val_test_loader_from_train

set_determinism(123)

def _as_uint8_01(a):
    a = a.astype(np.uint8, copy=False)
    a[a == 255] = 0
    a = (a == 1).astype(np.uint8, copy=False)
    return a

def _ensure_3d(arr, name="", case=""):
    """Devuelve (D,H,W) añadiendo una dimensión si viene 2D."""
    a = np.asarray(arr)
    if a.ndim == 5:        # (B,C,D,H,W)
        a = a[0, 0]
    elif a.ndim == 4:      # (B,D,H,W) o (C,D,H,W) con C=1
        a = a[0]
    elif a.ndim == 3:      # (D,H,W)
        pass
    elif a.ndim == 2:      # (H,W) -> (1,H,W)
        a = a[None, ...]
        print(f"[WARN] {case}: {name} venía 2D {arr.shape}, se asume D=1 -> {a.shape}")
    else:
        print(f"[SKIP] {case}: {name}.ndim={a.ndim} shape={a.shape} (no soportado)")
        return None
    return a

def load_pred_nii(path, case=""):
    """Carga NIfTI y devuelve máscara binaria (D,H,W) uint8."""
    arr = nib.load(path).get_fdata()
    # (C,D,H,W) -> argmax canal
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = np.argmax(arr, axis=0)
    arr = _ensure_3d(arr, "pred", case)
    if arr is None:
        return None
    arr_u = arr.astype(np.float32)
    # Heurística de binarización:
    if arr_u.max() <= 1.0:
        arr_b = (arr_u > 0.5).astype(np.uint8)
    else:
        arr_b = (arr_u > 0).astype(np.uint8)
    return arr_b

def dice_hd95(gt_bool, pr_bool, voxel_spacing=(1, 1, 1)):
    if pr_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pr_bool, gt_bool)
        hd = metric.binary.hd95(pr_bool, gt_bool, voxelspacing=voxel_spacing)
        return float(dsc), float(hd)
    return 0.0, 50.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Ruta a data/fullres/<dataset> (la misma que usaste en inferencia)")
    parser.add_argument("--pred_dir", required=True,
                        help="Carpeta con los .nii.gz de predicción (p.ej. .../prediction_results/segmamba)")
    parser.add_argument("--out_dir", default="/home/trawlins/tesis/prediction_results/result_metrics")
    parser.add_argument("--csv_name", default="colorectal_metrics_roi_from_dataloader.csv")
    parser.add_argument("--npy_name", default="colorectal_metrics_roi_from_dataloader.npy")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Misma partición de test que en inferencia
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    print("data length is", len(test_ds))
    print("Total casos test:", len(test_ds))

    per_case = []
    case_ids = []
    skipped = 0

    for batch in tqdm(test_ds, total=len(test_ds)):
        props = batch["properties"]
        case = props["name"][0] if isinstance(props["name"], (list, tuple)) else props["name"]

        # === GT (ROI del dataloader, como en inferencia) ===
        seg = batch.get("seg", None)
        if seg is None:
            print(f"[SKIP] {case}: batch sin 'seg'.")
            skipped += 1
            continue

        gt_raw = seg.detach().cpu().numpy()  # puede ser 5D/4D/3D/2D
        gt = _ensure_3d(gt_raw, "gt", case)
        if gt is None:
            skipped += 1
            continue
        gt = _as_uint8_01(gt)  # binario 0/1

        # === Pred ROI guardada (nii.gz) ===
        pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")
        if not os.path.exists(pred_path):
            print(f"[SKIP] {case}: pred no encontrada en {pred_path}")
            skipped += 1
            continue

        pred = load_pred_nii(pred_path, case)
        if pred is None:
            skipped += 1
            continue

        # Igualar formas (si difieren, resize nearest a la forma del GT)
        if pred.shape != gt.shape:
            if pred.ndim == 3 and gt.ndim == 3:
                t = torch.from_numpy(pred.astype(np.float32))[None, None]  # (1,1,D,H,W)
                t = F.interpolate(t, size=gt.shape, mode="nearest")
                pred = t[0, 0].byte().numpy()
                print(f"[INFO] {case}: pred reshaped {pred.shape} -> {gt.shape}")
            else:
                print(f"[SKIP] {case}: pred.shape={pred.shape} gt.shape={gt.shape} (dims incompatibles)")
                skipped += 1
                continue

        dsc, hd = dice_hd95(gt.astype(bool), pred.astype(bool), voxel_spacing=(1, 1, 1))
        per_case.append([dsc, hd])
        case_ids.append(case)

    if not per_case:
        print("❌ No se calcularon métricas (0 casos válidos). Revisa rutas/nombres.")
        return

    arr = np.asarray(per_case, dtype=np.float32)
    np.save(os.path.join(args.out_dir, args.npy_name), arr)

    # CSV
    import csv
    csv_path = os.path.join(args.out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for cid, (d, h) in zip(case_ids, arr):
            w.writerow([cid, float(d), float(h)])

    print("✅ Métricas ROI (desde dataloader) calculadas")
    print("Casos:", len(arr), f"(saltados: {skipped})")
    print("Dice promedio:", float(arr[:, 0].mean()))
    print("HD95 promedio:", float(arr[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)

if __name__ == "__main__":
    main()
