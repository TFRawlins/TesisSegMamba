#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from medpy import metric

import torch
from monai.utils import set_determinism
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice as lt_dice  # la misma de tu validación

set_determinism(123)

def _to_3d(arr: np.ndarray, name: str, case_id: str):
    arr = np.asarray(arr)
    # Casos típicos: (1,D,H,W) o (D,H,W) o (D,H,W,1) o (1,1,D,H,W)
    if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] in (1, 2):
        arr = arr[0, 0]  # -> (D,H,W)
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]     # -> (D,H,W) o (H,W,1)
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]  # -> (D,H,W)
    # Si vino etiquetado con 255 como padding, llévalo a 0
    if name == "gt":
        arr = arr.astype(np.uint8)
        arr[arr == 255] = 0
        # binariza: asume 1 = foreground
        arr = (arr == 1).astype(np.uint8)
    # Verifica 3D
    if arr.ndim != 3:
        print(f"[SKIP] {case_id}: {name}.ndim={arr.ndim} forma={arr.shape} (esperado 3D). Se omite.")
        return None
    return arr

def load_pred_nii(pred_path: str) -> np.ndarray:
    """Carga NIfTI y devuelve máscara binaria (D,H,W) uint8 en {0,1}."""
    nii = nib.load(pred_path)
    arr = nii.get_fdata()
    # posibles formas: (D,H,W) o (1,D,H,W) o (C,D,H,W)
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            # si hubiera one-hot: CxDxHxW -> clase por argmax
            arr = np.argmax(arr, axis=0)
    # binariza: nuestras predicciones guardadas son 0/1 ya, pero esto es a prueba de balas
    if arr.max() > 1:
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = (arr > 0.5).astype(np.uint8)
    return arr

def safe_medpy_metrics(gt_bool: np.ndarray, pred_bool: np.ndarray, spacing=(1,1,1)):
    """Dice y HD95 (si ambas máscaras tienen positivos); si no, (0, 50)."""
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = float(metric.binary.dc(pred_bool, gt_bool))
        try:
            hd95 = float(metric.binary.hd95(pred_bool, gt_bool, voxelspacing=tuple(spacing)))
        except Exception:
            hd95 = 50.0
        return dsc, hd95
    else:
        return 0.0, 50.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_name", required=True, type=str,
                    help="Nombre de la carpeta de predicciones dentro de results_root, "
                         "o una ruta absoluta a esa carpeta.")
    ap.add_argument("--data_dir", required=True, type=str,
                    help="Ruta a data/fullres/<dataset> que usa get_train_val_test_loader_from_train")
    ap.add_argument("--results_root", default="prediction_results", type=str,
                    help="Raíz donde está la carpeta de predicciones si pred_name no es ruta absoluta.")
    ap.add_argument("--out_dir", default="prediction_results/result_metrics", type=str)
    ap.add_argument("--csv_name", default="colorectal_metrics_roi.csv")
    ap.add_argument("--npy_name", default="colorectal_metrics_roi.npy")
    ap.add_argument("--log_name", default="metrics_debug_roi.log")
    args = ap.parse_args()

    # Construye pred_dir
    pred_dir = args.pred_name if os.path.isabs(args.pred_name) else os.path.join(args.results_root, args.pred_name)
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"No existe el directorio de predicciones: {pred_dir}")

    # Carga el mismo test_ds que usaste en validación (mismo crop/espacio ROI)
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    print(f"Total casos test: {len(test_ds)}")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_name)
    log_lines = ["# Log de métricas ROI (mismo espacio que validación)\n"]

    results = []
    case_ids = []

    # Para debug: contaremos cuántos casos “cuadran” perfectos en shape 192³
    ok_shape = 0

    for i, batch in enumerate(tqdm(test_ds, total=len(test_ds))):
        props = batch["properties"]
        case = props["name"][0] if isinstance(props["name"], (list, tuple)) else props["name"]
        case = str(case)
        case_ids.append(case)

        # 1) GT ROI del loader: (B,1,D,H,W) -> (D,H,W) binaria
        gt = batch["seg"]
        if isinstance(gt, torch.Tensor):
            # (B,1,D,H,W) esperado
            if gt.ndim != 5 or gt.shape[1] != 1:
                log_lines.append(f"[{case}] ⚠️ shape GT inesperada: {tuple(gt.shape)}\n")
            gt = gt[0, 0].detach().cpu().numpy()
        gt = (gt == 1).astype(np.uint8)

        # 2) Pred ROI desde NIfTI guardado
        pred_path = os.path.join(pred_dir, f"{case}.nii.gz")
        if not os.path.exists(pred_path):
            log_lines.append(f"[{case}] ❌ pred no encontrada: {pred_path}\n")
            results.append([0.0, 50.0])
            continue

        pred = load_pred_nii(pred_path)  # (D,H,W) uint8
        gt = _to_3d(gt, "gt", case_id)
        pred = _to_3d(pred, "pred", case_id)
        if gt is None or pred is None:
            # salta este caso y continúa el loop
            continue
        # 3) Asegurar shapes (deberían ser 192³ ambos). Si difieren por 1–2 voxels, corrige con pad/crop seguro.
        if pred.shape != gt.shape:
            Dz, Dy, Dx = gt.shape
            Pz, Py, Px = pred.shape
            if (abs(Dz-Pz) <= 2) and (abs(Dy-Py) <= 2) and (abs(Dx-Px) <= 2):
                # pad/crop centrado para igualar
                def center_fit(a, target_shape):
                    out = np.zeros(target_shape, dtype=a.dtype)
                    tz, ty, tx = target_shape
                    sz = min(a.shape[0], tz)
                    sy = min(a.shape[1], ty)
                    sx = min(a.shape[2], tx)
                    z0 = (tz - sz)//2; y0 = (ty - sy)//2; x0 = (tx - sx)//2
                    az0 = (a.shape[0] - sz)//2; ay0 = (a.shape[1] - sy)//2; ax0 = (a.shape[2] - sx)//2
                    out[z0:z0+sz, y0:y0+sy, x0:x0+sx] = a[az0:az0+sz, ay0:ay0+sy, ax0:ax0+sx]
                    return out
                pred = center_fit(pred, gt.shape)
            else:
                log_lines.append(f"[{case}] ⚠️ shapes distintas (pred {pred.shape} vs gt {gt.shape}). Se intenta resize nearest.\n")
                # último recurso: resize con torch (mantener binario)
                import torch.nn.functional as F
                t = torch.from_numpy(pred.astype(np.float32))[None,None]
                t = F.interpolate(t, size=gt.shape, mode="nearest")
                pred = t[0,0].byte().numpy()
        else:
            ok_shape += 1

        # 4) Booleanos
        gt_bool   = gt.astype(bool)
        pred_bool = pred.astype(bool)

        # 5) Métricas (medpy) + Dice de light_training para contrastar
        dsc, hd95 = safe_medpy_metrics(gt_bool, pred_bool, spacing=(1,1,1))
        dsc_lt = float(lt_dice(pred.astype(np.uint8), gt.astype(np.uint8)))  # misma que en tu validación
        results.append([dsc, hd95])

        # 6) Debug por caso
        if i < 6:  # loguea unos cuantos casos
            up, cp = np.unique(pred, return_counts=True)
            ug, cg = np.unique(gt, return_counts=True)
            log_lines.append(
                f"[{case}] pred.sum={int(pred.sum())} gt.sum={int(gt.sum())} "
                f"unique_pred={list(zip(up.tolist(), cp.tolist()))} "
                f"unique_gt={list(zip(ug.tolist(), cg.tolist()))} "
                f"Dice_medpy={dsc:.4f} Dice_light={dsc_lt:.4f} HD95={hd95:.2f}\n"
            )

    results = np.asarray(results, dtype=np.float32)
    np.save(os.path.join(args.out_dir, args.npy_name), results)

    # CSV
    csv_path = os.path.join(args.out_dir, args.csv_name)
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice_medpy", "HD95"])
        for cid, (dsc, hd) in zip(case_ids, results):
            w.writerow([cid, float(dsc), float(hd)])

    # Resumen
    print("✅ Métricas ROI calculadas (comparables con inferencia)")
    print("Casos:", len(results))
    print("Dice promedio:", float(results[:,0].mean()))
    print("HD95 promedio:", float(results[:,1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)

    # Log
    log_lines.append(f"\nOK shapes (pred==gt): {ok_shape}/{len(results)}\n")
    with open(os.path.join(args.out_dir, args.log_name), "w") as f:
        f.writelines(log_lines)

if __name__ == "__main__":
    main()
