#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from medpy import metric
import nibabel as nib
import torch.nn.functional as F

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from monai.utils import set_determinism
set_determinism(123)

def to_3d_uint8(a):
    a = np.asarray(a)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 3:
        return None
    return a.astype(np.uint8)

def dice_hd95(gt_bool, pr_bool, voxel_spacing=(1,1,1)):
    if pr_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pr_bool, gt_bool)
        hd  = metric.binary.hd95(pr_bool, gt_bool, voxelspacing=voxel_spacing)
        return float(dsc), float(hd)
    # caso vacío: mantener estable el hd95
    return 0.0, 50.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Ruta a data/fullres/<dataset> (la misma que usaste para inferencia)")
    parser.add_argument("--pred_dir", required=True,
                        help="Carpeta donde están tus .nii.gz de predicción (p.ej. /home/.../prediction_results/segmamba)")
    parser.add_argument("--out_dir", default="/home/trawlins/tesis/prediction_results/result_metrics")
    parser.add_argument("--csv_name", default="colorectal_metrics_roi_from_dataloader.csv")
    parser.add_argument("--npy_name", default="colorectal_metrics_roi_from_dataloader.npy")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Misma partición de test que en inferencia
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    print("Total casos test:", len(test_ds))

    per_case = []
    case_ids = []
    skipped = 0

    for batch in tqdm(test_ds, total=len(test_ds)):
        props = batch["properties"]
        case = props["name"][0] if isinstance(props["name"], (list, tuple)) else props["name"]
        # GT ROI tal como se usó en inferencia
        gt = batch.get("seg", None)
        if gt is None:
            print(f"[SKIP] {case}: no viene 'seg' en el batch.")
            skipped += 1
            continue
        gt = gt[0,0].detach().cpu().numpy().astype(np.uint8)  # (Dz,Dy,Dx), valores 0/1/255
        gt[gt == 255] = 0
        gt = (gt == 1).astype(np.uint8)

        # 2) Pred ROI desde NIfTI guardado
        pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")
        if not os.path.exists(pred_path):
            print(f"[SKIP] {case}: pred no encontrada en {pred_path}")
            skipped += 1
            continue

        pred = nib.load(pred_path).get_fdata()
        # Si guardaste como etiqueta 0/1, esto ya es 3D; si fuera 4D (C,D,H,W) reducimos
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        elif pred.ndim == 4:
            pred = np.argmax(pred, axis=0)
        if pred.ndim != 3:
            print(f"[SKIP] {case}: pred ndim={pred.ndim}, esperaba 3D.")
            skipped += 1
            continue
        # Binariza por seguridad
        if pred.max() > 1:
            pred = (pred > 0).astype(np.uint8)
        else:
            pred = (pred > 0.5).astype(np.uint8)

        # 3) Igualar formas (deberían ser 192³ ambos; si difieren, resize nearest)
        if pred.shape != gt.shape:
            t = torch.from_numpy(pred.astype(np.float32))[None,None]  # (1,1,D,H,W)
            t = F.interpolate(t, size=gt.shape, mode="nearest")
            pred = t[0,0].byte().numpy()

        # 4) Métricas
        dsc, hd = dice_hd95(gt.astype(bool), pred.astype(bool), voxel_spacing=(1,1,1))
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
    print("Casos:", len(arr))
    print("Dice promedio:", float(arr[:,0].mean()))
    print("HD95 promedio:", float(arr[:,1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)
    if skipped:
        print(f"ℹ️ Casos saltados: {skipped}")

if __name__ == "__main__":
    main()
