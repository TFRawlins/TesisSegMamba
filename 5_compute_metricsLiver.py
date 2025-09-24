import os
import numpy as np
import torch
import pickle
from medpy import metric
from tqdm import tqdm
import csv
from monai.transforms import Resize

# --- Configuración ---
LIVER_ID = 1  # Ajusta si tu etiqueta de hígado es distinta

# --- Utilidades ---
def cal_metric(gt_bin: np.ndarray, pred_bin: np.ndarray, voxel_spacing):
    """
    gt_bin, pred_bin: booleanos o {0,1} con misma forma (D,H,W)
    voxel_spacing: (z, y, x)
    """
    if pred_bin.sum() > 0 and gt_bin.sum() > 0:
        dice = metric.binary.dc(pred_bin, gt_bin)
        hd95 = metric.binary.hd95(pred_bin, gt_bin, voxelspacing=voxel_spacing)
        return np.array([dice, hd95], dtype=np.float32)
    else:
        # Caso degenerado: al menos una máscara vacía
        return np.array([0.0, 50.0], dtype=np.float32)

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
    return info

def load_prediction_label(case_base_path: str) -> np.ndarray:
    """
    Intenta cargar primero *_probs.npy (C,D,H,W), si no existe usa *_pred.npy (D,H,W).
    Devuelve etiquetas (D,H,W) en int (0..C-1) listas para filtrar por clase.
    """
    probs_path = case_base_path + "_probs.npy"
    pred_path  = case_base_path + "_pred.npy"

    if os.path.exists(probs_path):
        pred = np.load(probs_path)  # (C,D,H,W) o (1,D,H,W) dependiendo del guardado
        if pred.ndim == 4 and pred.shape[0] > 1:
            # multicanal: tomar clase más probable
            pred_lbl = np.argmax(pred, axis=0).astype(np.uint8)  # (D,H,W)
        elif pred.ndim == 4 and pred.shape[0] == 1:
            # canal único con probs -> umbral 0.5
            pred_lbl = (pred[0] >= 0.5).astype(np.uint8)          # (D,H,W) binario
        else:
            # ya viene (D,H,W)
            pred_lbl = pred.astype(np.uint8)
    else:
        # Compatibilidad: solo etiquetas guardadas
        pred = np.load(pred_path)  # (D,H,W)
        # Si por error viniera (1,D,H,W), aplastamos
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        pred_lbl = pred.astype(np.uint8)

    return pred_lbl

def align_pred_to_gt(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    Alinea sin desplazar: resize nearest de pred al tamaño de GT.
    Evita center-crop/pad para no introducir offset espacial.
    """
    if pred_bin.shape == gt_bin.shape:
        return pred_bin
    resizer = Resize(spatial_size=gt_bin.shape, mode="nearest")
    pred_t = torch.as_tensor(pred_bin[None, ...])  # [1, D, H, W]
    return resizer(pred_t).squeeze(0).numpy()

# --- Main ---
if __name__ == "__main__":
    pred_dir = "/home/trawlins/tesis/prediction_results/segmamba"
    data_dir = "/home/trawlins/tesis/data/fullres/train"
    metrics_dir = "/home/trawlins/tesis/prediction_results/result_metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    result_list = []
    case_ids = []

    # Lista basada en *_pred.npy para identificar casos; luego preferimos *_probs.npy si existe
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith("_pred.npy")])

    for fname in tqdm(pred_files):
        case_id = fname.replace("_pred.npy", "")

        pred_base = os.path.join(pred_dir, case_id)               # sin sufijo
        seg_path  = os.path.join(data_dir, f"{case_id}_seg.npy")
        pkl_path  = os.path.join(data_dir, f"{case_id}.pkl")

        if not os.path.exists(seg_path) or not os.path.exists(pkl_path):
            print(f"⚠️ Archivos faltantes para {case_id}, se omite.")
            continue

        # Cargar predicción (etiquetas) y GT
        pred_lbl = load_prediction_label(pred_base)               # (D,H,W) int
        gt       = np.load(seg_path)
        if gt.ndim == 4 and gt.shape[0] == 1:
            gt = gt[0]
        elif gt.ndim == 4:
            # Si viniera con canales múltiples inesperados, toma el primero
            gt = gt[0]

        # Binarizar SOLO hígado
        gt_bin   = (gt == LIVER_ID).astype(np.uint8)
        pred_bin = (pred_lbl == LIVER_ID).astype(np.uint8)

        # Alinear formas sin shift
        pred_bin = align_pred_to_gt(pred_bin, gt_bin)

        # Spacing para HD95 en orden (z,y,x)
        info = load_pkl(pkl_path)
        spacing = info.get("spacing", info.get("itk_spacing", [1, 1, 1]))
        if spacing is None:
            spacing = [1, 1, 1]
        # Si venía (x,y,z), invertimos a (z,y,x)
        spacing = spacing[::-1]

        # Métricas
        m = cal_metric(gt_bin.astype(bool), pred_bin.astype(bool), spacing)
        result_list.append(m)
        case_ids.append(case_id)

    if not result_list:
        print("❌ No se calcularon métricas (no hubo casos válidos).")
        exit(0)

    result_array = np.stack(result_list, axis=0)  # [N, 2] -> dice, hd95
    np.save(os.path.join(metrics_dir, "segmamba_liver.npy"), result_array)

    # CSV
    csv_path = os.path.join(metrics_dir, "segmamba_liver.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case ID", "Dice", "HD95"])
        for cid, metrics in zip(case_ids, result_array):
            writer.writerow([cid, float(metrics[0]), float(metrics[1])])

    # Resumen
    print("✅ Métricas calculadas")
    print("Total casos:", len(result_array))
    print("Dice promedio:", float(result_array[:, 0].mean()))
    print("HD95 promedio:", float(result_array[:, 1].mean()))
    print(f"📄 Guardado en:\n - {metrics_dir}/segmamba_liver.npy\n - {metrics_dir}/segmamba_liver.csv")
