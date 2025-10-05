#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import pickle
import csv
from tqdm import tqdm
from medpy import metric
import nibabel as nib

# -------- utilidades --------

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_pred_mask(pred_path: str) -> np.ndarray:
    """
    Carga predicción como máscara binaria (0/1) con forma (D,H,W).
    Soporta .nii/.nii.gz y .npy (en varias variantes).
    """
    if pred_path.endswith(".nii") or pred_path.endswith(".nii.gz"):
        arr = nib.load(pred_path).get_fdata()
        # posibles formas: (D,H,W) o (1,D,H,W) o (C,D,H,W)
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:  # one-hot
                arr = np.argmax(arr, axis=0)
        # binarizar si hay valores >1
        if arr.max() > 1:
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = (arr > 0.5).astype(np.uint8)
        return arr.astype(np.uint8)

    # npy
    arr = np.load(pred_path)
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:  # one-hot
            arr = np.argmax(arr, axis=0)
    arr = arr.astype(np.uint8)
    if arr.max() > 1:
        arr = (arr > 0).astype(np.uint8)
    return arr

def find_gt_roi_path(gt_roi_dir: str, case_id: str):
    # Orden de búsqueda para GT en ROI (192³)
    cands = [
        os.path.join(gt_roi_dir, f"{case_id}_seg.npy"),
        os.path.join(gt_roi_dir, f"{case_id}_mask.npy"),
        os.path.join(gt_roi_dir, f"{case_id}_segTr.npy"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def load_gt_roi(gt_roi_dir: str, case_id: str):
    """
    Carga GT ROI (192,192,192). Devuelve (gt_uint8_0_1, spacing, msg_error).
    Hace fail si no es 192³ o no existe.
    """
    seg_path = find_gt_roi_path(gt_roi_dir, case_id)
    if seg_path is None:
        return None, None, f"[{case_id}] GT ROI no encontrado en {gt_roi_dir}"

    gt = np.load(seg_path)
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    elif gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]

    if gt.shape != (192, 192, 192):
        return None, None, f"[{case_id}] GT ROI encontrado pero con shape {gt.shape}, se esperaba (192,192,192)."

    gt = gt.astype(np.uint8)
    gt[gt == 255] = 0
    gt = (gt == 1).astype(np.uint8)

    # spacing: intentamos leer pkl (si existe) pero no es crítico para Dice
    pkl_path = os.path.join(gt_roi_dir, f"{case_id}.pkl")
    spacing = [1, 1, 1]
    if os.path.exists(pkl_path):
        try:
            info = load_pkl(pkl_path)
            spacing = info.get("spacing", [1, 1, 1])
        except Exception:
            pass

    return gt, spacing, None

def cal_metric(gt_bool: np.ndarray, pred_bool: np.ndarray, voxel_spacing):
    """
    Dice + HD95 (si ambos no vacíos). Si uno está vacío: Dice=0, HD95=50.
    """
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pred_bool, gt_bool)
        hd95 = metric.binary.hd95(pred_bool, gt_bool, voxelspacing=tuple(voxel_spacing))
        return float(dsc), float(hd95)
    else:
        return 0.0, 50.0

# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir",
                    default="/home/trawlins/tesis/prediction_results/segmamba",
                    help="Directorio de predicciones (.nii.gz o *_pred.npy)")
    ap.add_argument("--gt_roi_dir",
                    default="/home/trawlins/tesis/data/colorectal/fullres/colorectal",
                    help="Directorio con GT ROI (192x192x192) y opcionalmente *.pkl")
    ap.add_argument("--out_dir",
                    default="/home/trawlins/tesis/prediction_results/result_metrics",
                    help="Directorio de salida para .npy y .csv")
    ap.add_argument("--csv_name", default="colorectal_metrics_roi.csv")
    ap.add_argument("--npy_name", default="colorectal_metrics_roi.npy")
    ap.add_argument("--log_name", default="metrics_debug_roi.log")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_name)

    pred_files = sorted(f for f in os.listdir(args.pred_dir)
                        if (f.endswith(".nii.gz") or f.endswith("_pred.npy")))
    if not pred_files:
        print(f"❌ No se encontraron predicciones en: {args.pred_dir}")
        return

    results = []
    case_ids = []

    with open(log_path, "w") as L:
        L.write("# Log ROI estricto (GT y pred deben ser 192x192x192)\n\n")

        print("Evaluando casos (ROI estricto):")
        for fname in tqdm(pred_files):
            case_id = fname.replace(".nii.gz", "") if fname.endswith(".nii.gz") else fname.replace("_pred.npy", "")
            pred_path = os.path.join(args.pred_dir, fname)

            # --- cargar GT ROI ---
            gt, spacing, err = load_gt_roi(args.gt_roi_dir, case_id)
            if err is not None:
                L.write(f"[SKIP] {err}\n")
                continue

            # --- cargar pred ROI ---
            try:
                pred = load_pred_mask(pred_path)  # (D,H,W) uint8 0/1
            except Exception as e:
                L.write(f"[SKIP] [{case_id}] Error cargando pred: {e}\n")
                continue

            # --- chequeos estrictos de shape/valores ---
            ok_shape = (pred.shape == gt.shape == (192, 192, 192))
            ok_vals_pred = set(np.unique(pred).tolist()).issubset({0, 1})
            ok_vals_gt   = set(np.unique(gt).tolist()).issubset({0, 1})

            L.write(f"[CASE] {case_id}\n")
            L.write(f"  pred.shape: {pred.shape}  gt.shape: {gt.shape}\n")
            L.write(f"  pred.sum: {int(pred.sum())}  gt.sum: {int(gt.sum())}\n")
            L.write(f"  pred.unique: {sorted(set(np.unique(pred).tolist()))}\n")
            L.write(f"  gt.unique:   {sorted(set(np.unique(gt).tolist()))}\n")

            if not ok_shape:
                L.write(f"  -> SKIP por shape no-ROI (se exige 192x192x192)\n\n")
                continue
            if not (ok_vals_pred and ok_vals_gt):
                L.write(f"  -> SKIP por valores no binarios en pred/gt\n\n")
                continue

            # --- métricas ---
            dsc, hd = cal_metric(gt.astype(bool), pred.astype(bool), spacing)
            results.append([dsc, hd])
            case_ids.append(case_id)

            L.write(f"  Dice: {dsc:.4f}  HD95: {hd:.2f}\n\n")

    if not results:
        print("❌ No se calcularon métricas (0 casos válidos en ROI). Revisa rutas/nombres.")
        print(f"🪵 Log: {log_path}")
        return

    results = np.asarray(results, dtype=np.float32)
    np.save(os.path.join(args.out_dir, args.npy_name), results)

    csv_path = os.path.join(args.out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for cid, (dsc, hd) in zip(case_ids, results):
            w.writerow([cid, float(dsc), float(hd)])

    print("✅ Métricas (ROI) calculadas")
    print("Total casos:", len(results))
    print("Dice promedio:", float(results[:, 0].mean()))
    print("HD95 promedio:", float(results[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)
    print("🪵 Log:", log_path)

if __name__ == "__main__":
    main()
