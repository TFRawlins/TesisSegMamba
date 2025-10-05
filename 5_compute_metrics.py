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
from monai.transforms import Resize

# -------------------------
# Utilidades
# -------------------------
def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def _to_bool_mask(arr: np.ndarray) -> np.ndarray:
    """Normaliza a máscara booleana 0/1 con shape (D,H,W)."""
    if arr.ndim == 4:
        # (1,D,H,W) o (C,D,H,W) o (D,H,W,1)
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            # Probable one-hot (C,D,H,W)
            arr = np.argmax(arr, axis=0)
    arr = arr.astype(np.uint8)
    if arr.max() > 1:
        arr = (arr > 0).astype(np.uint8)
    return arr.astype(bool)

def load_pred_mask(pred_path: str) -> np.ndarray:
    """
    Carga predicción como máscara booleana (D,H,W).
    Soporta .nii/.nii.gz y .npy (también onehot/canal).
    """
    if pred_path.endswith(".nii") or pred_path.endswith(".nii.gz"):
        nii = nib.load(pred_path)
        arr = nii.get_fdata()
        return _to_bool_mask(arr)
    # npy
    arr = np.load(pred_path)
    return _to_bool_mask(arr)

def load_gt_mask_and_spacing(gt_dir: str, case_id: str):
    """Carga GT (como máscara booleana (D,H,W)) y spacing desde .pkl si existe."""
    if gt_dir is None:
        return None, None, "GT dir no provisto"
    seg_candidates = [
        os.path.join(gt_dir, f"{case_id}_seg.npy"),
        os.path.join(gt_dir, f"{case_id}_mask.npy"),
        os.path.join(gt_dir, f"{case_id}_segTr.npy"),
    ]
    seg_path = next((p for p in seg_candidates if os.path.exists(p)), None)
    if seg_path is None:
        return None, None, f"GT no encontrado en {gt_dir}"
    gt = np.load(seg_path)
    # Normaliza a (D,H,W) y booleana 0/1
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    elif gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]
    gt = gt.astype(np.uint8)
    gt[gt == 255] = 0
    gt = (gt == 1)
    # spacing opcional
    pkl_path = os.path.join(gt_dir, f"{case_id}.pkl")
    spacing = [1, 1, 1]
    if os.path.exists(pkl_path):
        info = load_pkl(pkl_path)
        spacing = info.get("spacing", [1, 1, 1])
    return gt.astype(bool), spacing, None

def cal_metric(gt_bool: np.ndarray, pred_bool: np.ndarray, voxel_spacing):
    """
    gt_bool, pred_bool: bool (D,H,W)
    voxel_spacing: tupla/lista de 3 floats (z,y,x)
    """
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pred_bool, gt_bool)
        hd95 = metric.binary.hd95(pred_bool, gt_bool, voxelspacing=tuple(voxel_spacing))
        return np.array([dsc, hd95], dtype=np.float32)
    else:
        # Si una de las máscaras está vacía, devuelve 0 y un HD95 "grande" estable
        return np.array([0.0, 50.0], dtype=np.float32)

# -------------------------
# Script principal
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        default="/home/trawlins/tesis/prediction_results/segmamba",
        help="Directorio con predicciones (.nii.gz o *_pred.npy).",
    )
    parser.add_argument(
        "--gt_roi_dir",
        default="/home/trawlins/tesis/data/fullres/train",
        help="Directorio con GT en ROI 192x192x192 (el que usa tu dataloader en validation).",
    )
    parser.add_argument(
        "--gt_fullres_dir",
        default="/home/trawlins/tesis/data/colorectal/fullres/colorectal",
        help="Directorio con GT full-res (por si tus pred están en full-res).",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/trawlins/tesis/prediction_results/result_metrics",
        help="Directorio de salida para .npy/.csv y logs.",
    )
    parser.add_argument("--csv_name", default="colorectal_metrics.csv")
    parser.add_argument("--npy_name", default="colorectal_metrics.npy")
    parser.add_argument(
        "--debug_sample", type=int, default=5,
        help="Número de casos a loguear detalladamente (para no saturar el log)."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    debug_log = os.path.join(args.out_dir, "metrics_debug.log")
    with open(debug_log, "w") as _f:
        _f.write("# Log de métricas/depuración por caso\n")

    pred_files = sorted(
        f for f in os.listdir(args.pred_dir)
        if (f.endswith(".nii.gz") or f.endswith("_pred.npy"))
    )
    if not pred_files:
        print(f"❌ No se encontraron predicciones en: {args.pred_dir}")
        return

    result_list = []
    case_ids = []

    print("Evaluando casos:")
    for idx, fname in enumerate(tqdm(pred_files)):
        case_id = fname.replace(".nii.gz", "").replace("_pred.npy", "")
        pred_path = os.path.join(args.pred_dir, fname)

        # 1) Cargar pred (boolean)
        try:
            pred = load_pred_mask(pred_path)  # (D,H,W) bool
        except Exception as e:
            print(f"⚠️ Error cargando pred de {case_id}: {e}")
            continue

        # 2) Cargar GT: preferir ROI; si no, intentar full-res
        gt, spacing, err = load_gt_mask_and_spacing(args.gt_roi_dir, case_id)
        used_dir = "gt_roi_dir"
        if err is not None:
            gt, spacing, err = load_gt_mask_and_spacing(args.gt_fullres_dir, case_id)
            used_dir = "gt_fullres_dir"
        if err is not None:
            print(f"⚠️ {err} para {case_id}, se omite.")
            continue

        # 3) Alinear shapes (ideal: ya coinciden en ROI)
        need_resize = (gt.shape != pred.shape)
        if need_resize:
            # WARNING: este resize no corrige *offset* si GT es full-res y pred es ROI.
            # Lo hacemos solo para poder medir y dejar rastro en el log.
            resizer = Resize(spatial_size=gt.shape, mode="nearest")
            pred_t = torch.from_numpy(pred.astype(np.float32))[None, ...]  # (1,D,H,W)
            pred_rs = resizer(pred_t).squeeze(0).numpy().astype(bool)
            pred_eval = pred_rs
        else:
            pred_eval = pred

        # 4) Métrica
        m = cal_metric(gt, pred_eval, spacing)
        result_list.append(m)
        case_ids.append(case_id)

        # 5) Log de depuración (limitado por --debug_sample)
        if idx < args.debug_sample:
            with open(debug_log, "a") as f:
                f.write(f"\n[CASE] {case_id}\n")
                f.write(f"  pred_file: {fname}\n")
                f.write(f"  used_gt:   {used_dir}\n")
                f.write(f"  pred.shape: {tuple(pred.shape)}  gt.shape: {tuple(gt.shape)}\n")
                f.write(f"  pred.sum: {int(pred.sum())}  gt.sum: {int(gt.sum())}\n")
                f.write(f"  resized_for_eval: {need_resize}\n")
                f.write(f"  Dice: {float(m[0]):.4f}  HD95: {float(m[1]):.2f}\n")

    if not result_list:
        print("❌ No se calcularon métricas (0 casos válidos). Revisa rutas y nombres.")
        return

    result_array = np.stack(result_list, axis=0)
    np.save(os.path.join(args.out_dir, args.npy_name), result_array)

    # CSV por caso
    csv_path = os.path.join(args.out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for cid, (dsc, hd) in zip(case_ids, result_array):
            w.writerow([cid, float(dsc), float(hd)])

    # Resumen
    print("✅ Métricas calculadas")
    print("Total casos:", len(result_array))
    print("Dice promedio:", float(result_array[:, 0].mean()))
    print("HD95 promedio:", float(result_array[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)
    print(" 🪵 Log de depuración:", debug_log)

if __name__ == "__main__":
    main()
