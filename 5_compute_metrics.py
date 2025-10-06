#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from medpy import metric
from monai.utils import set_determinism
from light_training.dataloading.dataset import get_train_val_test_loader_from_train

set_determinism(123)


def to_3d_uint8(arr, name="arr", case=""):
    """
    Normaliza un array a (D,H,W) uint8 si es posible.
    Elimina ejes de tamaño 1 por delante/atrás. Si no queda 3D, retorna None.
    """
    a = np.asarray(arr)
    # Quitar dims de tamaño 1 hasta acercarnos a 3D
    while a.ndim > 3 and (a.shape[0] == 1 or a.shape[-1] == 1):
        if a.shape[0] == 1:
            a = a[0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
        else:
            break

    if a.ndim != 3:
        print(f"[SKIP] {case}: {name}.ndim={a.ndim} forma={a.shape} (esperado 3D). Se omite.")
        return None

    return a.astype(np.uint8)


def load_pred_nii(path, case=""):
    """
    Carga NIfTI y devuelve máscara binaria (D,H,W) uint8.
    Soporta (D,H,W) o (C,D,H,W) (en cuyo caso toma argmax canal).
    """
    arr = nib.load(path).get_fdata()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 4:
        # Si estuviera one-hot/probs en (C,D,H,W)
        arr = np.argmax(arr, axis=0)

    arr = to_3d_uint8(arr, "pred_nii", case)
    if arr is None:
        return None

    # Binariza por seguridad
    if arr.max() > 1:
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = (arr > 0.5).astype(np.uint8)
    return arr


def dice_hd95(gt_bool, pr_bool, voxel_spacing=(1, 1, 1)):
    if pr_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pr_bool, gt_bool)
        hd = metric.binary.hd95(pr_bool, gt_bool, voxelspacing=voxel_spacing)
        return float(dsc), float(hd)
    return 0.0, 50.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Ruta a data/fullres/<dataset> (la misma que usaste para inferencia)",
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Carpeta con los .nii.gz de predicción (p.ej. .../prediction_results/segmamba)",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/trawlins/tesis/prediction_results/result_metrics",
    )
    parser.add_argument(
        "--csv_name",
        default="colorectal_metrics_roi_from_dataloader.csv",
    )
    parser.add_argument(
        "--npy_name",
        default="colorectal_metrics_roi_from_dataloader.npy",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Misma partición de test que en inferencia
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    print("data length is", len(test_ds))
    print("Total casos test:", len(test_ds))

    per_case = []
    case_ids = []
    skipped = 0

    for batch in tqdm(test_ds, total=len(test_ds)):
        props = batch["properties"]
        case = props["name"][0] if isinstance(props["name"], (list, tuple)) else props["name"]

        # GT ROI desde dataloader (exactamente como en inferencia)
        if "seg" not in batch or batch["seg"] is None:
            print(f"[SKIP] {case}: batch no tiene 'seg'.")
            skipped += 1
            continue

        gt_raw = batch["seg"][0, 0].detach().cpu().numpy()  # puede venir con 255
        gt = to_3d_uint8(gt_raw, "gt", case)
        if gt is None:
            skipped += 1
            continue

        gt[gt == 255] = 0
        gt = (gt == 1).astype(np.uint8)  # binario 0/1

        # Pred ROI desde NIfTI guardado
        pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")
        if not os.path.exists(pred_path):
            print(f"[SKIP] {case}: pred no encontrada en {pred_path}")
            skipped += 1
            continue

        pred = load_pred_nii(pred_path, case)
        if pred is None:
            skipped += 1
            continue

        # Igualar formas (si difieren, resize nearest a la forma del GT)
        if pred.shape != gt.shape:
            if pred.ndim == 3 and gt.ndim == 3:
                t = torch.from_numpy(pred.astype(np.float32))[None, None]  # (1,1,D,H,W)
                t = F.interpolate(t, size=gt.shape, mode="nearest")
                pred = t[0, 0].byte().numpy()
            else:
                print(f"[SKIP] {case}: pred.shape={pred.shape} gt.shape={gt.shape} (dims incompatible)")
                skipped += 1
                continue

        dsc, hd = dice_hd95(gt.astype(bool), pred.astype(bool), voxel_spacing=(1, 1, 1))
        per_case.append([dsc, hd])
        case_ids.append(case)

    if not per_case:
        print("❌ No se calcularon métricas (0 casos válidos). Revisa rutas/nombres.")
        return

    arr = np.asarray(per_case, dtype=np.float32)
    np.save(os.path.join(args.out_dir, args.npy_name), arr)

    # CSV
    import csv
    csv_path = os.path.join(args.out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for cid, (d, h) in zip(case_ids, arr):
            w.writerow([cid, float(d), float(h)])

    print("✅ Métricas ROI (desde dataloader) calculadas")
    print("Casos:", len(arr), f"(saltados: {skipped})")
    print("Dice promedio:", float(arr[:, 0].mean()))
    print("HD95 promedio:", float(arr[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)


if __name__ == "__main__":
    main()
