import os
import numpy as np
import torch
import pickle
from medpy import metric
from tqdm import tqdm
import csv

def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    else:
        return np.array([0.0, 50.0])  # penalización

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
    return info

if __name__ == "__main__":
    pred_dir = "./prediction_results/segmamba"
    data_dir = "/workspace/data/content/data/fullres/train"
    metrics_dir = "./prediction_results/result_metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    result_list = []
    case_ids = []

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith("_pred.npy")])

    for fname in tqdm(pred_files):
        case_id = fname.replace("_pred.npy", "")
        case_ids.append(case_id)

        pred_path = os.path.join(pred_dir, fname)
        seg_path = os.path.join(data_dir, f"{case_id}_seg.npy")
        pkl_path = os.path.join(data_dir, f"{case_id}.pkl")

        if not os.path.exists(seg_path) or not os.path.exists(pkl_path):
            print(f"⚠️  Archivos faltantes para {case_id}, se omite.")
            continue

        pred = np.load(pred_path)
        gt = np.load(seg_path)

        # Corregir dimensiones si tienen canal extra
        if pred.ndim == 4:
            pred = pred[0]
        if gt.ndim == 4:
            gt = gt[0]

        if pred.shape != gt.shape:
            print(f"❌ Shape mismatch for {case_id}: pred {pred.shape}, gt {gt.shape}")
            continue

        info = load_pkl(pkl_path)
        voxel_spacing = info.get("spacing", [1, 1, 1])
        m = cal_metric(gt.astype(np.bool_), pred.astype(np.bool_), voxel_spacing)
        result_list.append(m)

    if not result_list:
        print("❌ No se calcularon métricas. Verifica las predicciones y las máscaras.")
        exit()

    result_array = np.stack(result_list, axis=0)
    np.save(os.path.join(metrics_dir, "segmamba_liver.npy"), result_array)

    # Exportar a CSV
    csv_path = os.path.join(metrics_dir, "segmamba_liver.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case ID", "Dice", "HD95"])
        for cid, metrics in zip(case_ids, result_array):
            writer.writerow([cid, metrics[0], metrics[1]])

    # Mostrar resumen
    print("✅ Métricas calculadas")
    print("Total casos:", len(result_array))
    print("Dice promedio:", result_array[:, 0].mean())
    print("HD95 promedio:", result_array[:, 1].mean())
    print(f"📄 Guardado en:\n - {metrics_dir}/segmamba_liver.npy\n - {metrics_dir}/segmamba_liver.csv")
