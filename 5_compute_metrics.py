import os
import numpy as np
import torch
import pickle
from medpy import metric
from tqdm import tqdm

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

    result_list = []

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith("_pred.npy")])

    for fname in tqdm(pred_files):
        case_id = fname.replace("_pred.npy", "")

        pred_path = os.path.join(pred_dir, fname)
        seg_path = os.path.join(data_dir, f"{case_id}_seg.npy")
        pkl_path = os.path.join(data_dir, f"{case_id}.pkl")

        pred = np.load(pred_path)
        gt = np.load(seg_path)
        info = load_pkl(pkl_path)

        voxel_spacing = info.get("spacing", [1, 1, 1])  # fallback si no existe
        m = cal_metric(gt.astype(np.bool_), pred.astype(np.bool_), voxel_spacing)
        result_list.append(m)

    result_array = np.stack(result_list, axis=0)
    os.makedirs("./prediction_results/result_metrics", exist_ok=True)
    np.save("./prediction_results/result_metrics/segmamba_liver.npy", result_array)

    print("✅ Métricas calculadas")
    print("Shape:", result_array.shape)
    print("Dice promedio:", result_array[:, 0].mean())
    print("HD95 promedio:", result_array[:, 1].mean())
