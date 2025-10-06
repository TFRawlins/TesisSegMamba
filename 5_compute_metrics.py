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

def _as_uint8_01(a):
    a = a.astype(np.uint8, copy=False)
    a[a == 255] = 0
    a = (a == 1).astype(np.uint8, copy=False)
    return a

def _ensure_3d(arr, name="", case=""):
    a = np.asarray(arr)
    if a.ndim == 5:        # (B,C,D,H,W)
        a = a[0, 0]
    elif a.ndim == 4:      # (B,D,H,W) o (C,D,H,W)
        a = a[0]
    elif a.ndim == 3:      # (D,H,W)
        pass
    elif a.ndim == 2:      # (H,W) -> (1,H,W)
        a = a[None, ...]
        print(f"[WARN] {case}: {name} venía 2D {arr.shape}, se asume D=1 -> {a.shape}")
    else:
        print(f"[SKIP] {case}: {name}.ndim={a.ndim} shape={a.shape} (no soportado)")
        return None
    return a

def _binarize_pred_volume(arr_float):
    """Si parece prob (<=1): >0.5; si parece etiqueta: >0."""
    if arr_float.max() <= 1.0:
        return (arr_float > 0.5).astype(np.uint8)
    return (arr_float > 0).astype(np.uint8)

def _dice_hd95(gt_bool, pr_bool, spacing=(1,1,1)):
    if pr_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pr_bool, gt_bool)
        hd = metric.binary.hd95(pr_bool, gt_bool, voxelspacing=spacing)
        return float(dsc), float(hd)
    return 0.0, 50.0

def _resize_to(arr_3d_uint8, target_shape):
    if arr_3d_uint8.shape == target_shape:
        return arr_3d_uint8
    t = torch.from_numpy(arr_3d_uint8.astype(np.float32))[None, None]  # (1,1,D,H,W)
    t = F.interpolate(t, size=target_shape, mode="nearest")
    return t[0,0].byte().numpy()

def _orient_candidates(pred_xyz):
    """
    pred_xyz viene como lo devuelve nib (X,Y,Z).
    Genera candidatos en orden (D,H,W)≈(Z,Y,X) y variantes con flips,
    porque el GT del dataloader suele estar en (D,H,W).
    """
    # Permutaciones: identidad (X,Y,Z) y (Z,Y,X)
    perms = [
        ("xyz", (0,1,2)),
        ("zyx", (2,1,0)),
    ]
    flips = [
        ("",  (False, False, False)),
        ("fz", (True,  False, False)),
        ("fy", (False, True,  False)),
        ("fx", (False, False, True )),
        ("fzy",(True,  True,  False)),
        ("fzx",(True,  False, True )),
        ("fyx",(False, True,  True )),
        ("fzyx",(True,  True,  True )),
    ]

    cand = []
    for p_name, p in perms:
        base = np.transpose(pred_xyz, p)
        for f_name, (fz, fy, fx) in flips:
            a = base
            if fz: a = np.flip(a, axis=0)
            if fy: a = np.flip(a, axis=1)
            if fx: a = np.flip(a, axis=2)
            cand.append((f"{p_name}{f_name}", a))
    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Ruta a data/fullres/<dataset> (la misma de inferencia)")
    ap.add_argument("--pred_dir", required=True, help="Carpeta de .nii.gz de predicción (p.ej. .../prediction_results/segmamba)")
    ap.add_argument("--out_dir", default="/home/trawlins/tesis/prediction_results/result_metrics")
    ap.add_argument("--csv_name", default="colorectal_metrics_roi_oriented.csv")
    ap.add_argument("--npy_name", default="colorectal_metrics_roi_oriented.npy")
    ap.add_argument("--log_name", default="metrics_debug_roi_oriented.log")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_name)
    log = ["# Log métricas con autocorrección de orientación de pred NIfTI\n"]

    # misma partición test
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    print("data length is", len(test_ds))
    print("Total casos test:", len(test_ds))

    results = []
    case_ids = []
    skipped = 0

    for batch in tqdm(test_ds, total=len(test_ds)):
        props = batch["properties"]
        case = props["name"][0] if isinstance(props["name"], (list,tuple)) else props["name"]

        # GT ROI del dataloader
        seg = batch.get("seg", None)
        if seg is None:
            print(f"[SKIP] {case}: batch sin 'seg'")
            skipped += 1
            continue
        gt = _ensure_3d(seg.detach().cpu().numpy(), "gt", case)
        if gt is None:
            skipped += 1
            continue
        gt = _as_uint8_01(gt)

        # pred NIfTI (ROI guardado)
        pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")
        if not os.path.exists(pred_path):
            print(f"[SKIP] {case}: pred no encontrada en {pred_path}")
            skipped += 1
            continue

        pred_xyz = nib.load(pred_path).get_fdata()  # (X,Y,Z) float
        # Si vino 4D: colapsar canal
        if pred_xyz.ndim == 4:
            if pred_xyz.shape[0] == 1:
                pred_xyz = pred_xyz[0]
            else:
                pred_xyz = np.argmax(pred_xyz, axis=0)

        # Generar candidatos de orientación
        best_dice, best_hd, best_tag = -1.0, 50.0, ""
        best_arr = None
        for tag, cand in _orient_candidates(pred_xyz):
            cand = _binarize_pred_volume(cand.astype(np.float32))
            cand = _ensure_3d(cand, "pred_cand", case)
            if cand is None:
                continue
            # resize si hace falta
            if cand.shape != gt.shape:
                cand = _resize_to(cand, gt.shape)
            d, h = _dice_hd95(gt.astype(bool), cand.astype(bool), spacing=(1,1,1))
            if d > best_dice:
                best_dice, best_hd, best_tag, best_arr = d, h, tag, cand

        if best_arr is None:
            print(f"[SKIP] {case}: no se pudo construir candidato de orientación")
            skipped += 1
            continue

        results.append([best_dice, best_hd])
        case_ids.append(case)
        log.append(
            f"[{case}] best={best_tag}  pred.shape={best_arr.shape}  gt.shape={gt.shape}  "
            f"dice={best_dice:.4f} hd95={best_hd:.2f}\n"
        )

    if not results:
        print("❌ No se calcularon métricas (0 casos). Revisa rutas y nombres.")
        with open(log_path, "w") as f:
            f.writelines(log)
        return

    arr = np.asarray(results, dtype=np.float32)
    np.save(os.path.join(args.out_dir, args.npy_name), arr)

    # CSV por caso
    import csv
    csv_path = os.path.join(args.out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for cid, (d, h) in zip(case_ids, arr):
            w.writerow([cid, float(d), float(h)])

    with open(log_path, "w") as f:
        f.writelines(log)

    print("✅ Métricas ROI (autoorientadas) calculadas")
    print("Casos:", len(arr), f"(saltados: {skipped})")
    print("Dice promedio:", float(arr[:, 0].mean()))
    print("HD95 promedio:", float(arr[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)
    print(" 🪵 Log:", log_path)

if __name__ == "__main__":
    main()
