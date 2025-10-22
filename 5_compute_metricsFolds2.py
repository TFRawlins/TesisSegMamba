#!/usr/bin/env python3
import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch
import torch.nn.functional as F
from medpy import metric

# -------------------------------
# Utils de IO / spacing / formas
# -------------------------------
def load_itk(path):
    """
    Lee NIfTI con SimpleITK y devuelve:
      - arr en (Z,Y,X) 3D (si viene 4D colapsa el primer eje; si 2D, inserta Z=1)
      - voxelspacing en orden (Z,Y,X) para usar directo en medpy.hd95
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X) o (T,Z,Y,X)

    if arr.ndim == 4:
        # (T,Z,Y,X) o (C,Z,Y,X). Si T/C==1, toma 0; si no, toma el primer canal.
        arr = arr[0]
    elif arr.ndim == 2:
        arr = arr[None, :, :]  # (1,H,W) -> (Z=1,H,W)

    sp = img.GetSpacing()  # ITK: (sx, sy, sz)
    # convertir a (Z,Y,X)
    if len(sp) >= 3:
        voxelspacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    elif len(sp) == 2:
        voxelspacing_zyx = (1.0, float(sp[1]), float(sp[0]))
    elif len(sp) == 1:
        voxelspacing_zyx = (float(sp[0]), float(sp[0]), float(sp[0]))
    else:
        voxelspacing_zyx = (1.0, 1.0, 1.0)

    return arr.astype(np.uint8), voxelspacing_zyx


def ensure_3d_uint8(arr):
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0] if a.shape[0] == 1 else np.argmax(a, axis=0)
    elif a.ndim == 2:
        a = a[None, ...]
    return a.astype(np.uint8)


def resize_pred_to_gt(pred_zyx_uint8, gt_shape_zyx):
    """
    Remuestrea pred -> shape GT usando nearest (sin permutar ejes).
    pred y GT están en (Z,Y,X).
    """
    if tuple(pred_zyx_uint8.shape) == tuple(gt_shape_zyx):
        return pred_zyx_uint8

    t = torch.from_numpy(pred_zyx_uint8.astype(np.float32))[None, None]  # (1,1,Z,Y,X)
    t = F.interpolate(t, size=gt_shape_zyx, mode="nearest")
    return t[0, 0].byte().numpy()


def binarize_gt(arr):
    """GT robusta: >0 -> 1."""
    return (arr > 0).astype(np.uint8)


def binarize_pred(arr):
    """
    Si parece prob/clase: umbral 0.5; si ya es etiqueta, >0.
    Trabajamos con (Z,Y,X).
    """
    a = arr.astype(np.float32, copy=False)
    if a.max() <= 1.0:
        return (a > 0.5).astype(np.uint8)
    return (a > 0).astype(np.uint8)


def dice_hd95(gt_bool, pr_bool, spacing_zyx=(1, 1, 1)):
    if pr_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pr_bool, gt_bool)
        hd = metric.binary.hd95(pr_bool, gt_bool, voxelspacing=spacing_zyx)
        return float(dsc), float(hd)
    return 0.0, 50.0


def read_fold_list_ids(fold_lists_dir, fold, split="val"):
    p = os.path.join(fold_lists_dir, f"fold{fold}_{split}.txt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No existe: {p}")
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def find_label_path(label_dir, case_id):
    """
    Busca el label en labelsTr con nombres comunes.
    """
    cand = [
        os.path.join(label_dir, f"{case_id}.nii.gz"),
        os.path.join(label_dir, f"{case_id}_gt.nii.gz"),
        os.path.join(label_dir, f"{case_id}_seg.nii.gz"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Ruta a nnUNet_raw/Dataset001_Colorectal")
    ap.add_argument("--pred_dir", required=True,
                    help="Carpeta con predicciones NIfTI (p.ej. /.../prediction_results/colorectal_folds/fold0)")
    ap.add_argument("--out_dir", default=None,
                    help="Dónde guardar métricas (por defecto: <pred_dir>/../result_metrics)")
    # Folds (opcional)
    ap.add_argument("--fold_lists_dir", default=None,
                    help="Carpeta con fold{n}_{train,val}.txt (opcional)")
    ap.add_argument("--fold", type=int, default=None,
                    help="Índice de fold si usas fold_lists_dir (opcional)")
    args = ap.parse_args()

    # Out dir
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.pred_dir), "result_metrics")
    os.makedirs(args.out_dir, exist_ok=True)

    # Pred IDs presentes
    pred_ids = sorted([
        os.path.basename(p)[:-7]  # quita ".nii.gz"
        for p in sorted(os.listdir(args.pred_dir))
        if p.endswith(".nii.gz")
    ])

    # Si se pasa fold_list, filtramos por el hold-out del fold
    if args.fold_lists_dir is not None and args.fold is not None:
        holdout_ids = read_fold_list_ids(args.fold_lists_dir, args.fold, split="val")
        # Intersección: sólo evalúa lo que realmente predeciste
        case_ids = [cid for cid in holdout_ids if cid in pred_ids]
    else:
        case_ids = pred_ids

    print(f"Total casos a evaluar: {len(case_ids)}")

    label_dir = os.path.join(args.data_dir, "labelsTr")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"No existe labelsTr en {label_dir}")

    results = []
    csv_rows = []
    log_path = os.path.join(args.out_dir, "metrics_debug.log")

    with open(log_path, "w") as logf:
        logf.write("# Métricas en espacio de label (sin auto-orientación)\n")

        for case in tqdm(case_ids, total=len(case_ids)):
            # --- Paths ---
            lab_path = find_label_path(label_dir, case)
            if lab_path is None:
                print(f"[SKIP] {case}: no encontré label en {label_dir}")
                continue
            pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")
            if not os.path.exists(pred_path):
                print(f"[SKIP] {case}: no encontré pred en {pred_path}")
                continue

            # --- Cargar GT y pred ---
            gt_np, gt_spacing_zyx = load_itk(lab_path)   # (Z,Y,X), spacing (Z,Y,X)
            pred_np, _ = load_itk(pred_path)              # (Z,Y,X), spacing no lo usamos aquí

            gt_np = ensure_3d_uint8(gt_np)
            pred_np = ensure_3d_uint8(pred_np)

            # Binarizar
            gt_bin = binarize_gt(gt_np)
            pred_bin = binarize_pred(pred_np)

            # Remuestrea pred -> shape GT si difiere
            if pred_bin.shape != gt_bin.shape:
                pred_bin = resize_pred_to_gt(pred_bin, gt_bin.shape)

            # Métricas
            d, h = dice_hd95(gt_bin.astype(bool), pred_bin.astype(bool), spacing_zyx=gt_spacing_zyx)
            results.append([d, h])
            csv_rows.append((case, d, h))

            # Log por caso
            logf.write(
                f"{case}: dice={d:.4f} hd95={h:.2f} "
                f"shape(gt/pred)={gt_bin.shape}/{pred_bin.shape} "
                f"spacingZYX={tuple(float(s) for s in gt_spacing_zyx)} "
                f"pos(gt/pred)={int(gt_bin.sum())}/{int(pred_bin.sum())}\n"
            )

    if not results:
        print("❌ No se calcularon métricas (0 casos). Revisa rutas/IDs/preds.")
        print(" 🪵 Log:", log_path)
        return

    # Guardados
    arr = np.asarray(results, dtype=np.float32)
    np.save(os.path.join(args.out_dir, "colorectal_metrics.npy"), arr)

    import csv
    csv_path = os.path.join(args.out_dir, "colorectal_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "dice", "hd95"])
        for cid, d, h in csv_rows:
            w.writerow([cid, float(d), float(h)])

    print("✅ Métricas calculadas")
    print("Casos:", len(arr))
    print("Dice promedio:", float(arr[:, 0].mean()))
    print("HD95 promedio:", float(arr[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, "colorectal_metrics.npy"))
    print(" -", csv_path)
    print(" 🪵 Log:", log_path)


if __name__ == "__main__":
    main()
