import os
import numpy as np
import torch
import pickle
from medpy import metric
from tqdm import tqdm
import csv
from monai.transforms import Resize

def cal_metric(gt, pred, voxel_spacing):
    # gt, pred: binarios (0/1) y misma forma
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95], dtype=np.float32)
    else:
        # Si una de las máscaras no tiene positivos, devuelve algo neutro
        return np.array([0.0, 50.0], dtype=np.float32)

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
    return info

def load_pred_case(pred_dir, case_id):
    """
    Intenta cargar predicción desde NIfTI (nuevo) y, si no existe, desde *_pred.npy (legacy).
    Devuelve np.ndarray 3D (Z, Y, X).
    """
    nii_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
    npy_path = os.path.join(pred_dir, f"{case_id}_pred.npy")

    if os.path.exists(nii_path):
        # Leer NIfTI
        try:
            import nibabel as nib
            pred = nib.load(nii_path).get_fdata().astype(np.float32)
        except Exception:
            # Fallback con SimpleITK si nib no está disponible
            import SimpleITK as sitk
            pred = sitk.GetArrayFromImage(sitk.ReadImage(nii_path)).astype(np.float32)
            # sitk devuelve (Z, Y, X); nib suele devolver (X, Y, Z) -> ya lo tratamos como (Z,Y,X) aquí
        # Asegurar 3D
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        # Si vino (X,Y,Z), lo pasamos a (Z,Y,X)
        if pred.shape[0] not in (1, 2) and pred.shape[0] not in (pred.shape[1], pred.shape[2]):
            # Heurística rápida: si la primera dimensión no parece Z, permuta
            # (X,Y,Z) -> (Z,Y,X)
            pred = np.transpose(pred, (2, 1, 0))
        return pred
    elif os.path.exists(npy_path):
        pred = np.load(npy_path)
        if pred.ndim == 4:
            pred = pred[0]
        return pred
    else:
        return None

resize_to = (192, 192, 192)  # ROI de inferencia
resizer = Resize(spatial_size=resize_to, mode="nearest")

if __name__ == "__main__":
    pred_dir = "/home/trawlins/tesis/prediction_results/segmamba"
    data_dir = "/home/trawlins/tesis/data/colorectal/fullres/colorectal"
    metrics_dir = "/home/trawlins/tesis/prediction_results/result_metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Detecta qué lista usar: NIfTI (nuevo) o *_pred.npy (legacy)
    nii_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
    npy_files = sorted([f for f in os.listdir(pred_dir) if f.endswith("_pred.npy")])

    if len(nii_files) > 0:
        case_ids = [os.path.splitext(os.path.splitext(f)[0])[0] for f in nii_files]  # quita .nii.gz
    elif len(npy_files) > 0:
        case_ids = [f.replace("_pred.npy", "") for f in npy_files]
    else:
        print("❌ No se encontraron predicciones en NIfTI ni *_pred.npy.")
        raise SystemExit

    result_list = []
    out_rows = []

    for case_id in tqdm(case_ids, desc="Evaluando casos"):
        pred = load_pred_case(pred_dir, case_id)
        if pred is None:
            print(f"⚠️ Predicción no encontrada para {case_id}, se omite.")
            continue

        seg_path = os.path.join(data_dir, f"{case_id}_seg.npy")
        pkl_path = os.path.join(data_dir, f"{case_id}.pkl")
        if not os.path.exists(seg_path) or not os.path.exists(pkl_path):
            print(f"⚠️ Archivos faltantes para {case_id} (seg o pkl), se omite.")
            continue

        gt = np.load(seg_path)

        if gt.ndim == 4:
            gt = gt[0]
        if pred.ndim == 4:
            pred = pred[0]

        gt = gt.astype(np.uint8)
        # Si tu 255 significa 'fondo', remapea:
        gt[gt == 255] = 0
        gt = (gt == 1).astype(np.uint8)

        pred = (pred > 0).astype(np.uint8)

        if tuple(pred.shape) != tuple(gt.shape):
            # redimensionar pred a la forma de gt
            pred_t = torch.from_numpy(pred)[None, None].float()  # (1,1,D,H,W)
            pred_t = Resize(spatial_size=gt.shape, mode="nearest")(pred_t)
            pred = pred_t.squeeze().numpy().astype(np.uint8)

        info = load_pkl(pkl_path)
        spacing_from_pkl = info.get("spacing", [1, 1, 1])

        voxel_spacing = tuple(float(x) for x in spacing_from_pkl)
        m = cal_metric(gt.astype(bool), pred.astype(bool), voxel_spacing)
        result_list.append(m)
        out_rows.append([case_id, float(m[0]), float(m[1])])

    if not result_list:
        print("❌ No se calcularon métricas.")
        raise SystemExit

    result_array = np.stack(result_list, axis=0)

    # Guardar NPY y CSV con nombre de proyecto/dataset
    npy_out = os.path.join(metrics_dir, "colorectal_metrics.npy")
    csv_out = os.path.join(metrics_dir, "colorectal_metrics.csv")
    np.save(npy_out, result_array)

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case ID", "Dice", "HD95"])
        writer.writerows(out_rows)

    # Resumen
    print("✅ Métricas calculadas")
    print("Total casos:", len(result_array))
    print("Dice promedio:", float(result_array[:, 0].mean()))
    print("HD95 promedio:", float(result_array[:, 1].mean()))
    print(f"📄 Guardado en:\n - {npy_out}\n - {csv_out}")
