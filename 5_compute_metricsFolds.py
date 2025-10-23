#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from tqdm import tqdm
from medpy import metric
from monai.utils import set_determinism

set_determinism(123)

def _as_uint8_01(a):
    a = np.asarray(a)
    return (a > 0).astype(np.uint8, copy=False)

def _binarize_pred(arr_float):
    """
    Si parece prob (<=1): >0.5; si parece etiqueta ya discreta: >0.
    """
    arr = np.asarray(arr_float)
    if arr.max() <= 1.0:
        return (arr > 0.5).astype(np.uint8, copy=False)
    return (arr > 0).astype(np.uint8, copy=False)

def _dice_hd95(gt_u8, pr_u8, spacing):
    if pr_u8.sum() > 0 and gt_u8.sum() > 0:
        dsc = float(metric.binary.dc(pr_u8, gt_u8))
        hd  = float(metric.binary.hd95(pr_u8, gt_u8, voxelspacing=spacing))
        return dsc, hd
    # caso degenerado
    return 0.0, 50.0

def dice_hd95_affine_aware(pred_path, gt_path):
    """
    1) Carga GT y Pred.
    2) Resamplea Pred -> rejilla del GT usando affine (nearest).
    3) Binariza robustamente.
    4) Calcula Dice y HD95 con 'spacing' real del GT.
    """
    gt_nii = nib.load(gt_path)
    pr_nii = nib.load(pred_path)

    # Re-muestrea predicción al espacio del GT (sin suavizado)
    pr_res = resample_from_to(pr_nii, gt_nii, order=0)

    gt = _as_uint8_01(gt_nii.get_fdata())
    pr = _binarize_pred(pr_res.get_fdata())

    zooms = gt_nii.header.get_zooms()[:3]
    return _dice_hd95(gt, pr, spacing=zooms), (gt.shape, pr.shape), (gt_nii.affine, pr_res.affine), (gt.sum(), pr.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Ruta a nnUNet_raw/DatasetXXX_*/ (debe contener labelsTr)")
    ap.add_argument("--pred_dir", required=True, help="Carpeta con .nii.gz de predicción")
    ap.add_argument("--out_dir", default="/home/trawlins/tesis/prediction_results/result_metrics")
    ap.add_argument("--csv_name", default="colorectal_metrics_affine.csv")
    ap.add_argument("--npy_name", default="colorectal_metrics_affine.npy")
    ap.add_argument("--log_name", default="metrics_debug_affine.log")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_name)

    from glob import glob
    pred_paths = sorted(glob(os.path.join(args.pred_dir, "*.nii.gz")))
    case_ids = [os.path.basename(p)[:-7] for p in pred_paths]  # quita ".nii.gz"
    print("Total preds encontradas:", len(case_ids))

    label_dir = os.path.join(args.data_dir, "labelsTr")
    assert os.path.isdir(label_dir), f"No existe labelsTr en {label_dir}"

    results = []
    skipped = 0

    with open(log_path, "w") as logf:
        logf.write("# Log métricas (resample affine-aware Pred -> GT)\n")

        for case in tqdm(case_ids, total=len(case_ids)):
            gt_path = os.path.join(label_dir, f"{case}.nii.gz")
            pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")

            if not os.path.exists(gt_path):
                print(f"[SKIP] {case}: GT no encontrada en {gt_path}")
                logf.write(f"[SKIP] {case}: GT no encontrada en {gt_path}\n")
                skipped += 1
                continue
            if not os.path.exists(pred_path):
                print(f"[SKIP] {case}: Pred no encontrada en {pred_path}")
                logf.write(f"[SKIP] {case}: Pred no encontrada en {pred_path}\n")
                skipped += 1
                continue

            # Logear shapes brutas antes del resample (útil para debug rápido)
            gt_nii = nib.load(gt_path)
            pr_nii = nib.load(pred_path)
            print("[PAIR]", os.path.basename(pred_path), "->", gt_path,
                  "| shapes", pr_nii.shape, "->", gt_nii.shape,
                  "| aff_eq?", np.allclose(pr_nii.affine, gt_nii.affine))

            (dice, hd95), (gt_shape, pr_shape), (gt_aff, pr_aff), (gt_pos, pr_pos) = dice_hd95_affine_aware(pred_path, gt_path)

            logf.write(
                f"{case}: dice={dice:.4f} hd95={hd95:.2f} "
                f"shape(gt/pred_res)={gt_shape}/{pr_shape} "
                f"pos(gt/pred_res)={gt_pos}/{pr_pos}\n"
            )

            results.append([case, float(dice), float(hd95)])

    if not results:
        print("❌ No se calcularon métricas (0 casos). Revisa rutas y nombres.")
        print(" 🪵 Log:", log_path)
        raise SystemExit

    # Guardar NPY (solo métricas numéricas)
    arr = np.asarray([[d, h] for _, d, h in results], dtype=np.float32)
    np.save(os.path.join(args.out_dir, args.npy_name), arr)

    # Guardar CSV por caso
    import csv
    csv_path = os.path.join(args.out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case ID", "Dice", "HD95"])
        for cid, d, h in results:
            w.writerow([cid, d, h])

    print("✅ Métricas calculadas (affine-aware)")
    print("Casos:", len(arr), f"(saltados: {skipped})")
    print("Dice promedio:", float(arr[:, 0].mean()))
    print("HD95 promedio:", float(arr[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)
    print(" 🪵 Log:", log_path)

if __name__ == "__main__":
    main()
