#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
from medpy import metric

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from monai.utils import set_determinism

set_determinism(123)

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def to_3d_uint8(x):
    """Convierte array a (D,H,W) uint8, colapsando canal si viene (1,D,H,W) o (D,H,W,1)."""
    a = np.asarray(x)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]
    assert a.ndim == 3, f"Se esperaba 3D, llegó {a.shape}"
    return a.astype(np.uint8)

def load_pred_roi(pred_dir, case_id):
    """
    Carga predicción en ROI (192³) desde NIfTI o NPY y devuelve (D,H,W) uint8 en {0,1}.
    """
    nii_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
    npy_path = os.path.join(pred_dir, f"{case_id}_pred.npy")

    if os.path.exists(nii_path):
        arr = nib.load(nii_path).get_fdata()
        # Si viniera one-hot (C,D,H,W), argmax
        if arr.ndim == 4 and arr.shape[0] > 1:
            arr = np.argmax(arr, axis=0)
        # binariza: si [0,1], umbral 0.5; si >1, >0
        if arr.max() > 1.0:
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = (arr > 0.5).astype(np.uint8)
        return to_3d_uint8(arr)

    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        if arr.ndim == 4 and arr.shape[0] > 1:
            arr = np.argmax(arr, axis=0)
        if arr.max() > 1.0:
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = (arr > 0.5).astype(np.uint8)
        return to_3d_uint8(arr)

    raise FileNotFoundError(f"No existe predicción ROI para {case_id} en {pred_dir}")

def binarize_gt_roi(gt):
    """
    Convierte el GT ROI a binario 0/1 (clase 1 como foreground).
    El loader suele entregar label con shape (1,1,192,192,192).
    """
    g = np.asarray(gt)
    # esperado: (B=1, C=1, D, H, W) -> (D,H,W)
    if g.ndim == 5 and g.shape[0] == 1 and g.shape[1] == 1:
        g = g[0, 0]
    elif g.ndim == 4 and g.shape[0] == 1:
        g = g[0]
    g = g.astype(np.uint8)
    # algunos pipelines usan 255 como padding; ignóralo
    g[g == 255] = 0
    # clase positiva == 1
    g = (g == 1).astype(np.uint8)
    return g

def cal_metric(gt_bool: np.ndarray, pred_bool: np.ndarray, spacing=(1,1,1)):
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pred_bool, gt_bool)
        hd95 = metric.binary.hd95(pred_bool, gt_bool, voxelspacing=tuple(spacing))
        return np.array([dsc, hd95], dtype=np.float32)
    else:
        return np.array([0.0, 50.0], dtype=np.float32)

# ------------------------------------------------------------
# Main (ROI metrics, comparables con la validación)
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Ruta a data/fullres/<dataset> que usa get_train_val_test_loader_from_train")
    ap.add_argument("--pred_dir", required=True,
                    help="Directorio con las predicciones ROI (192³): NIfTI .nii.gz o *_pred.npy")
    ap.add_argument("--out_dir", default="prediction_results/result_metrics",
                    help="Dónde guardar métricas y CSV")
    ap.add_argument("--csv_name", default="colorectal_metrics_roi.csv")
    ap.add_argument("--npy_name", default="colorectal_metrics_roi.npy")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Carga el mismo test set que se usó en inferencia (mismas transformaciones/ROI)
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    print("Total casos test:", len(test_ds))

    results = []
    rows = []

    for batch in tqdm(test_ds, total=len(test_ds)):
        props = batch["properties"]
        case_id = props["name"]
        # a veces viene como lista de 1
        if isinstance(case_id, (list, tuple)):
            case_id = case_id[0]

        # GT ROI (192³) binario
        gt = binarize_gt_roi(batch.get("seg", None))
        if gt is None:
            # si el loader no trae seg en test, no podemos evaluar ROI
            continue

        try:
            pred = load_pred_roi(args.pred_dir, case_id)  # 192³
        except Exception as e:
            print(f"[WARN] {case_id}: {e}")
            continue

        # sanity shapes
        if pred.shape != gt.shape:
            # Si por alguna razón el loader nos dio otro tamaño, forzamos a 192³ con nearest.
            # (pero lo esperable es que ya sea igual)
            import monai
            from monai.transforms import Resize
            rsz = Resize(spatial_size=gt.shape, mode="nearest")
            pred = rsz(torch.from_numpy(pred)[None, ...]).squeeze(0).numpy().astype(np.uint8)

        # booleanos
        gt_b = gt.astype(bool)
        pr_b = pred.astype(bool)

        m = cal_metric(gt_b, pr_b, spacing=(1,1,1))
        results.append(m)
        rows.append((case_id, float(m[0]), float(m[1])))

    if not results:
        print("❌ No se pudo evaluar (no hay casos con GT ROI disponible).")
        return

    arr = np.stack(results, axis=0)
    np.save(os.path.join(args.out_dir, args.npy_name), arr)

    # CSV
    csv_path = os.path.join(args.out_dir, args.csv_name)
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for r in rows:
            w.writerow(r)

    print("✅ Métricas ROI calculadas (comparables con inferencia)")
    print("Casos:", len(arr))
    print("Dice promedio:", float(arr[:,0].mean()))
    print("HD95 promedio:", float(arr[:,1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)

if __name__ == "__main__":
    main()
