import os
import argparse
import numpy as np
import torch
import pickle
from medpy import metric
from tqdm import tqdm
import csv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",  type=str, default="/home/trawlins/tesis/prediction_results/segmamba")
    p.add_argument("--data_dir",  type=str, default="/home/trawlins/tesis/data/fullres/train")
    p.add_argument("--out_dir",   type=str, default="/home/trawlins/tesis/prediction_results/result_metrics")
    return p.parse_args()

def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dsc = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return float(dsc), float(hd95)
    else:
        return 0.0, 50.0

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
    return info

def match_shape_to_gt(pred, gt_shape):
    """Alinea pred al tamaño de GT con crop/pad centrado (sin interpolar)."""
    pred = np.asarray(pred)
    out = np.zeros(gt_shape, dtype=pred.dtype)
    # recortes/padding centrados
    def bounds(src, dst):
        # devuelve (s0, s1) en src y (d0, d1) en dst
        src_len, dst_len = src, dst
        if src_len >= dst_len:
            s0 = (src_len - dst_len) // 2
            s1 = s0 + dst_len
            d0, d1 = 0, dst_len
        else:
            d0 = (dst_len - src_len) // 2
            d1 = d0 + src_len
            s0, s1 = 0, src_len
        return (s0, s1), (d0, d1)

    (sz, sy, sx) = pred.shape
    (gz, gy, gx) = gt_shape

    (s0z, s1z), (d0z, d1z) = bounds(sz, gz)
    (s0y, s1y), (d0y, d1y) = bounds(sy, gy)
    (s0x, s1x), (d0x, d1x) = bounds(sx, gx)

    out[d0z:d1z, d0y:d1y, d0x:d1x] = pred[s0z:s1z, s0y:s1y, s0x:s1x]
    return out

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith("_pred.npy")])
    per_case = []  # (case_id, dice, hd95)

    for fname in tqdm(pred_files, desc="Metrics"):
        case_id = fname.replace("_pred.npy", "")

        pred_path = os.path.join(args.pred_dir, fname)
        seg_path  = os.path.join(args.data_dir, f"{case_id}_seg.npy")
        pkl_path  = os.path.join(args.data_dir, f"{case_id}.pkl")

        if not (os.path.exists(seg_path) and os.path.exists(pkl_path)):
            print(f"⚠️  Archivos faltantes para {case_id}, se omite.")
            continue

        pred = np.load(pred_path)
        gt   = np.load(seg_path)

        # Quita canal si viniera con (1, D, H, W)
        if pred.ndim == 4: pred = pred[0]
        if gt.ndim   == 4: gt   = gt[0]

        # Alinea forma SIN reescalar (mejor para HD95)
        if pred.shape != gt.shape:
            pred = match_shape_to_gt(pred, gt.shape)

        # Binariza
        pred = pred.astype(bool)
        gt   = gt.astype(bool)

        info = load_pkl(pkl_path)
        spacing = info.get("spacing", info.get("itk_spacing", [1,1,1]))

        dsc, hd = cal_metric(gt, pred, spacing)
        per_case.append((case_id, dsc, hd))

    if not per_case:
        print("❌ No se calcularon métricas (0 casos válidos).")
        return

    # Save per-case CSV
    csv_path = os.path.join(args.out_dir, "segmamba_liver.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for cid, dsc, hd in per_case:
            w.writerow([cid, dsc, hd])

    # Save npy (solo métricas)
    arr = np.array([[d, h] for _, d, h in per_case], dtype=np.float32)
    npy_path = os.path.join(args.out_dir, "segmamba_liver.npy")
    np.save(npy_path, arr)

    # Resumen
    dice_vals = arr[:,0]; hd_vals = arr[:,1]
    print("✅ Métricas calculadas")
    print(f"Total casos: {len(per_case)}")
    print(f"Dice  promedio: {dice_vals.mean():.4f} | mediana: {np.median(dice_vals):.4f} | std: {dice_vals.std():.4f}")
    print(f"HD95  promedio: {hd_vals.mean():.2f}  | mediana: {np.median(hd_vals):.2f}  | std: {hd_vals.std():.2f}")
    print(f"📄 Guardado en:\n - {npy_path}\n - {csv_path}")

if __name__ == "__main__":
    main()
