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

pred_path="/home/trawlins/tesis/prediction_results/segmamba/1109.nii.gz"
arr=nib.load(pred_path).get_fdata()
print("pred:", arr.shape, arr.min(), arr.max(), np.unique(arr, return_counts=True))
data_dir="/home/trawlins/tesis/data/colorectal/fullres/colorectal"
gt=np.load(os.path.join(data_dir,"1109_seg.npy"))
if gt.ndim==4 and gt.shape[0]==1: gt=gt[0]
gt[gt==255]=0
gt=(gt==1).astype(np.uint8)
print("gt:", gt.shape, gt.min(), gt.max(), np.unique(gt, return_counts=True))
pred=(arr>0.5) if arr.max()<=1 else (arr>0)
pred=pred.astype(bool)
gt=gt.astype(bool)
print("Dice:", metric.binary.dc(pred, gt))

def cal_metric(gt_bool: np.ndarray, pred_bool: np.ndarray, voxel_spacing):
    """
    gt_bool, pred_bool: booleanos con forma (D, H, W)
    voxel_spacing: tupla/lista de 3 floats (z, y, x)
    """
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pred_bool, gt_bool)
        hd95 = metric.binary.hd95(pred_bool, gt_bool, voxelspacing=tuple(voxel_spacing))
        return np.array([dsc, hd95], dtype=np.float32)
    else:
        # Si una de las máscaras está vacía, devuelve 0 y un HD95 "grande" estable
        return np.array([0.0, 50.0], dtype=np.float32)


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
    return info


def load_pred(pred_path: str) -> np.ndarray:
    """
    Carga predicción como máscara binaria (0/1) con forma (D,H,W).
    Soporta .nii/.nii.gz y .npy (en varias variantes).
    """
    if pred_path.endswith(".nii") or pred_path.endswith(".nii.gz"):
        nii = nib.load(pred_path)
        arr = nii.get_fdata()
        # posibles formas: (D,H,W) o (1,D,H,W) o (C,D,H,W)
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                # Si viene one-hot (C,D,H,W), elegir clase por argmax
                arr = np.argmax(arr, axis=0)
        arr = arr.astype(np.uint8)
        # Si viniera con valores >1, binarizamos (clase 1 como foreground)
        if arr.max() > 1:
            arr = (arr > 0).astype(np.uint8)
        return arr

    # npy
    arr = np.load(pred_path)
    if arr.ndim == 4:
        # (1,D,H,W) o (C,D,H,W) o (D,H,W,1)
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            # probablemente one-hot (C,D,H,W)
            arr = np.argmax(arr, axis=0)
    arr = arr.astype(np.uint8)
    if arr.max() > 1:
        arr = (arr > 0).astype(np.uint8)
    return arr


def load_gt_and_spacing(data_dir: str, case_id: str):
    """
    Carga GT como (D,H,W) uint8 y spacing desde pkl.
    Intenta varios nombres de archivo para el GT.
    """
    seg_candidates = [
        os.path.join(data_dir, f"{case_id}_seg.npy"),
        os.path.join(data_dir, f"{case_id}_mask.npy"),
        os.path.join(data_dir, f"{case_id}_segTr.npy"),
    ]
    seg_path = next((p for p in seg_candidates if os.path.exists(p)), None)
    if seg_path is None:
        return None, None, f"GT no encontrado para {case_id}"

    pkl_path = os.path.join(data_dir, f"{case_id}.pkl")
    if not os.path.exists(pkl_path):
        return None, None, f"PKL no encontrado para {case_id}"

    gt = np.load(seg_path)  # típicamente (1,192,192,192) o (D,H,W)
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    elif gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]

    gt = gt.astype(np.uint8)
    # Normaliza el GT: ignora 255 como fondo (padding) y deja binario 0/1
    gt[gt == 255] = 0
    gt = (gt == 1).astype(np.uint8)

    info = load_pkl(pkl_path)
    spacing = info.get("spacing", [1, 1, 1])

    return gt, spacing, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        default="/home/trawlins/tesis/prediction_results/segmamba",
        help="Directorio donde están las predicciones (.nii.gz o *_pred.npy)",
    )
    parser.add_argument(
        "--data_dir",
        default="/home/trawlins/tesis/data/colorectal/fullres/colorectal",
        help="Directorio de data/fullres/<dataset> con *_seg.npy y *.pkl",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/trawlins/tesis/prediction_results/result_metrics",
        help="Directorio de salida para .npy y .csv",
    )
    parser.add_argument(
        "--csv_name",
        default="colorectal_metrics.csv",
        help="Nombre del CSV de salida",
    )
    parser.add_argument(
        "--npy_name",
        default="colorectal_metrics.npy",
        help="Nombre del NPY de salida",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir
    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Listar casos desde pred_dir: soportamos nii.gz y _pred.npy
    pred_files = sorted(
        f for f in os.listdir(pred_dir) if (f.endswith(".nii.gz") or f.endswith("_pred.npy"))
    )
    if not pred_files:
        print(f"❌ No se encontraron predicciones en: {pred_dir}")
        return

    result_list = []
    case_ids = []

    print("Evaluando casos:")
    for fname in tqdm(pred_files):
        # case_id: "1109" desde "1109.nii.gz" o "1109_pred.npy"
        if fname.endswith(".nii.gz"):
            case_id = fname.replace(".nii.gz", "")
        else:
            case_id = fname.replace("_pred.npy", "")

        pred_path = os.path.join(pred_dir, fname)
        gt, spacing, err = load_gt_and_spacing(data_dir, case_id)
        if err is not None:
            print(f"⚠️ {err}, se omite.")
            continue

        try:
            pred = load_pred(pred_path)  # (D,H,W) uint8 0/1
        except Exception as e:
            print(f"⚠️ Error cargando predicción de {case_id}: {e}")
            continue

        # A tensores MONAI con canal (C,D,H,W)
        pred_t = torch.as_tensor(pred, dtype=torch.float32)[None, ...]  # (1,D,H,W)
        gt_t   = torch.as_tensor(gt,   dtype=torch.float32)[None, ...]  # (1,D,H,W)

        # Redimensionar pred al tamaño espacial del GT
        target_size = tuple(gt_t.shape[-3:])
        resizer = Resize(spatial_size=target_size, mode="nearest")
        pred_resized = resizer(pred_t).squeeze(0).numpy()  # (D,H,W)
        gt_resized   = gt_t.squeeze(0).numpy()             # (D,H,W)

        # Booleanos para medpy
        pred_bool = pred_resized.astype(bool)
        gt_bool   = gt_resized.astype(bool)

        m = cal_metric(gt_bool, pred_bool, spacing)
        result_list.append(m)
        case_ids.append(case_id)

    if not result_list:
        print("❌ No se calcularon métricas (0 casos válidos). Revisa rutas y nombres.")
        return

    result_array = np.stack(result_list, axis=0)
    np.save(os.path.join(out_dir, args.npy_name), result_array)

    # CSV por caso
    csv_path = os.path.join(out_dir, args.csv_name)
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
    print(" -", os.path.join(out_dir, args.npy_name))
    print(" -", csv_path)


if __name__ == "__main__":
    main()
