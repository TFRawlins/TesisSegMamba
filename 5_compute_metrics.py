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

# -------------------------- utilidades --------------------------

def safe_bool(arr) -> np.ndarray:
    """Asegura booleano 3D (D,H,W)."""
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    elif a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]
    assert a.ndim == 3, f"Se esperaba 3D, llegó {a.shape}"
    return a.astype(bool)

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def cal_metric(gt_bool: np.ndarray, pred_bool: np.ndarray, voxel_spacing):
    """gt_bool, pred_bool: (D,H,W) boolean; voxel_spacing: (z,y,x)."""
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pred_bool, gt_bool)
        hd95 = metric.binary.hd95(pred_bool, gt_bool, voxelspacing=tuple(voxel_spacing))
        return np.array([dsc, hd95], dtype=np.float32)
    else:
        return np.array([0.0, 50.0], dtype=np.float32)

def load_pred_any(pred_path: str) -> np.ndarray:
    """
    Carga predicción (NIfTI .nii/.nii.gz o .npy) y devuelve máscara 0/1 como (D,H,W) uint8.
    Si viene one-hot (C,D,H,W), hace argmax.
    Si viene probabilities, umbraliza >0 si es máscara de clase; si [0,1], usa >0.5.
    """
    if pred_path.endswith(".nii") or pred_path.endswith(".nii.gz"):
        arr = nib.load(pred_path).get_fdata()
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                # C,D,H,W -> argmax canal
                arr = np.argmax(arr, axis=0)
        # binariza
        if arr.max() > 1.0:
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = (arr > 0.5).astype(np.uint8)
        return arr.astype(np.uint8)

    # .npy
    arr = np.load(pred_path)
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            # posiblemente one-hot (C,D,H,W)
            arr = np.argmax(arr, axis=0)
    # binariza
    if arr.max() > 1.0:
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = (arr > 0.5).astype(np.uint8)
    return arr

def find_gt_file(data_dir: str, case_id: str):
    cand = [
        os.path.join(data_dir, f"{case_id}_seg.npy"),
        os.path.join(data_dir, f"{case_id}_mask.npy"),
        os.path.join(data_dir, f"{case_id}_segTr.npy"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None

# -------------------------- script principal --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_name", required=True,
                    help="Nombre de carpeta bajo results_root o ruta absoluta al dir con NIfTI")
    ap.add_argument("--data_dir", required=True,
                    help="Directorio con GT fullres (*.pkl + *_seg.npy)")
    ap.add_argument("--results_root", default="prediction_results",
                    help="Raíz donde está la carpeta de predicciones si pred_name no es ruta absoluta")
    ap.add_argument("--out_dir", default="prediction_results/result_metrics",
                    help="Directorio de salida (.npy / .csv / .log)")
    ap.add_argument("--csv_name", default="colorectal_metrics.csv")
    ap.add_argument("--npy_name", default="colorectal_metrics.npy")
    ap.add_argument("--log_name", default="metrics_debug.log")
    args = ap.parse_args()

    # Resolver directorio de predicciones
    if os.path.isabs(args.pred_name):
        pred_dir = args.pred_name
    else:
        pred_dir = os.path.join(args.results_root, args.pred_name)
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"No existe el directorio de predicciones: {pred_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_name)

    # Listar predicciones válidas
    pred_files = sorted([f for f in os.listdir(pred_dir)
                         if f.endswith(".nii.gz") or f.endswith("_pred.npy")])
    if not pred_files:
        print(f"❌ No se encontraron predicciones en: {pred_dir}")
        return

    # Abrir log
    log_lines = ["# Log de métricas/depuración por caso\n"]

    result_list = []
    case_ids = []

    print("data length is", len(pred_files))
    print("Total casos test:", len(pred_files))

    for fname in tqdm(pred_files):
        if fname.endswith(".nii.gz"):
            case_id = fname.replace(".nii.gz", "")
        else:
            case_id = fname.replace("_pred.npy", "")

        pred_path = os.path.join(pred_dir, fname)
        gt_path = find_gt_file(args.data_dir, case_id)
        pkl_path = os.path.join(args.data_dir, f"{case_id}.pkl")

        if gt_path is None or not os.path.exists(pkl_path):
            log_lines.append(f"\n[CASE] {case_id}\n  SKIP: faltan GT o PKL\n")
            continue

        # Cargar GT
        gt = np.load(gt_path)
        if gt.ndim == 4 and gt.shape[0] == 1:
            gt = gt[0]
        elif gt.ndim == 4 and gt.shape[-1] == 1:
            gt = gt[..., 0]
        gt = gt.astype(np.uint8)
        gt[gt == 255] = 0
        gt = (gt == 1).astype(np.uint8)  # binario 0/1

        # Cargar pred
        try:
            pred = load_pred_any(pred_path)  # 0/1, usualmente (D,H,W)
        except Exception as e:
            log_lines.append(f"\n[CASE] {case_id}\n  ERROR cargando pred: {e}\n")
            continue

        # Asegurar 3D y tipado
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        assert pred.ndim == 3, f"Predicción no es 3D: {pred.shape}"
        assert gt.ndim == 3, f"GT no es 3D: {gt.shape}"

        # Redimensionar pred -> tamaño GT (3D)
        target_size = tuple(int(x) for x in gt.shape[-3:])
        if not (len(target_size) == 3):
            raise ValueError(f"target_size inválido ({target_size}) para GT {gt.shape}")
        pred_t = torch.as_tensor(pred, dtype=torch.float32)[None, ...]  # (1,D,H,W) como C,D,H,W
        resizer = Resize(spatial_size=target_size, mode="nearest")
        pred_rs = resizer(pred_t).squeeze(0).numpy().astype(np.uint8)

        # Booleanos
        gt_bool = safe_bool(gt)
        pred_bool = safe_bool(pred_rs)

        # Spacing
        info = load_pkl(pkl_path)
        spacing = info.get("spacing", [1, 1, 1])

        # Métricas
        m = cal_metric(gt_bool, pred_bool, spacing)
        result_list.append(m)
        case_ids.append(case_id)

        # Log por caso
        log_lines += [
            f"\n[CASE] {case_id}",
            f"  pred_file: {fname}",
            f"  pred.shape: {tuple(pred.shape)}  gt.shape: {tuple(gt.shape)}",
            f"  pred.sum: {int(pred.sum())}  gt.sum: {int(gt.sum())}",
            f"  resized_for_eval: True",
            f"  Dice: {m[0]:.4f}  HD95: {m[1]:.2f}",
        ]

    if not result_list:
        print("❌ No se calcularon métricas (0 casos válidos). Revisa rutas/nombres.")
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        print("🪵 Log:", log_path)
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

    # Guardar log
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    # Resumen
    print("✅ Métricas calculadas")
    print("Total casos:", len(result_array))
    print("Dice promedio:", float(result_array[:, 0].mean()))
    print("HD95 promedio:", float(result_array[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)
    print(" 🪵 Log de depuración:", log_path)


if __name__ == "__main__":
    main()
