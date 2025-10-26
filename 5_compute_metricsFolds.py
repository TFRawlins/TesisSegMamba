#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from tqdm import tqdm
from medpy import metric
from monai.utils import set_determinism
import pickle
import warnings

set_determinism(123)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- utilidades ----------
def _as_uint8_01(a):
    a = np.asarray(a)
    return (a > 0).astype(np.uint8, copy=False)

def _binarize_pred(arr_float):
    """Si parece prob (<=1): >0.5; si parece etiqueta ya discreta: >0."""
    arr = np.asarray(arr_float)
    if arr.size == 0:
        return arr.astype(np.uint8, copy=False)
    return (arr > (0.5 if arr.max() <= 1.0 else 0)).astype(np.uint8, copy=False)

def _dice_hd95(gt_u8, pr_u8, spacing):
    if pr_u8.sum() > 0 and gt_u8.sum() > 0:
        dsc = float(metric.binary.dc(pr_u8, gt_u8))
        hd  = float(metric.binary.hd95(pr_u8, gt_u8, voxelspacing=spacing))
        return dsc, hd
    # casos degenerados (sin foreground)
    return (0.0 if gt_u8.sum() > 0 else 1.0), 50.0

def _read_spacing_from_meta(meta_path):
    """
    Lee spacing (Z,Y,X) desde {ID}.pkl o {ID}.npz del fullres si existe.
    Retorna (sz, sy, sx) o None.
    """
    try:
        if meta_path.endswith(".pkl") and os.path.isfile(meta_path):
            with open(meta_path, "rb") as f:
                d = pickle.load(f)
            for k in ["spacing", "target_spacing", "pixdim", "zoom", "zooms"]:
                if k in d:
                    v = d[k]
                    if isinstance(v, (list, tuple)) and len(v) >= 3:
                        return float(v[0]), float(v[1]), float(v[2])
        if meta_path.endswith(".npz") and os.path.isfile(meta_path):
            d = dict(np.load(meta_path, allow_pickle=True))
            for k in ["spacing", "target_spacing", "pixdim", "zoom", "zooms"]:
                if k in d:
                    v = d[k]
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v, (list, tuple)) and len(v) >= 3:
                        return float(v[0]), float(v[1]), float(v[2])
    except Exception:
        pass
    return None

def _find_meta_file(data_dir, case_id):
    pkl = os.path.join(data_dir, f"{case_id}.pkl")
    if os.path.isfile(pkl): return pkl
    npz = os.path.join(data_dir, f"{case_id}.npz")
    if os.path.isfile(npz): return npz
    return ""

def dice_hd95_affine_aware(pred_path, gt_path):
    """
    1) Carga GT y Pred como NIfTI.
    2) Resamplea Pred -> rejilla del GT usando affine (nearest).
    3) Binariza robustamente.
    4) Calcula Dice y HD95 con spacing real del GT.
    """
    gt_nii = nib.load(gt_path)
    pr_nii = nib.load(pred_path)
    pr_res = resample_from_to(pr_nii, gt_nii, order=0)

    gt = _as_uint8_01(gt_nii.get_fdata())
    pr = _binarize_pred(pr_res.get_fdata())
    zooms = gt_nii.header.get_zooms()[:3]
    return _dice_hd95(gt, pr, spacing=zooms), (gt.shape, pr.shape), (gt_nii.affine, pr_res.affine), (gt.sum(), pr.sum())

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Carpeta FULLRES con {ID}.npy / {ID}_seg.npy (+ opcional {ID}.pkl/.npz). Si existe labelsTr/ también se usa.")
    ap.add_argument("--pred_dir", required=True, help="Carpeta con predicciones: *.nii.gz y/o *_pred.npy")
    ap.add_argument("--out_dir", default="/home/trawlins/tesis/prediction_results/result_metrics")
    ap.add_argument("--csv_name", default="colorectal_metrics_affine.csv")
    ap.add_argument("--npy_name", default="colorectal_metrics_affine.npy")
    ap.add_argument("--log_name", default="metrics_debug_affine.log")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_name)

    # 1) Recolectar predicciones
    from glob import glob
    pred_nii = sorted(glob(os.path.join(args.pred_dir, "*.nii.gz")))
    pred_npy = sorted(glob(os.path.join(args.pred_dir, "*_pred.npy")))
    pred_map = {}  # case_id -> ("nii"|"npy", path)

    for p in pred_nii:
        cid = os.path.basename(p)[:-7]  # .nii.gz
        pred_map[cid] = ("nii", p)
    for p in pred_npy:
        base = os.path.basename(p)
        cid = base[:-9] if base.endswith("_pred.npy") else os.path.splitext(base)[0]
        # si ya había .nii.gz para el mismo caso, mantenemos NIfTI (affine-aware)
        pred_map.setdefault(cid, ("npy", p))

    case_ids = sorted(pred_map.keys())
    print("Total preds encontradas:", len(case_ids))
    if len(case_ids) == 0:
        print("❌ No hay predicciones en", args.pred_dir)
        raise SystemExit(1)

    # 2) Localizar ground truth
    fullres_dir = args.data_dir
    labelsTr_dir = os.path.join(args.data_dir, "labelsTr")
    has_nii_labels = os.path.isdir(labelsTr_dir)

    results = []
    skipped = 0

    with open(log_path, "w") as logf:
        logf.write("# Log métricas\n")

        for case in tqdm(case_ids, total=len(case_ids)):
            kind, pred_path = pred_map[case]
            gt_path_npy = os.path.join(fullres_dir, f"{case}_seg.npy")
            gt_path_nii = os.path.join(labelsTr_dir, f"{case}.nii.gz") if has_nii_labels else ""

            # Camino affine-aware: solo si PRED es NIfTI y existe GT NIfTI
            if kind == "nii" and os.path.isfile(gt_path_nii):
                try:
                    (dsc, hd95), (gt_shape, pr_shape), (_, _), (gt_pos, pr_pos) = dice_hd95_affine_aware(pred_path, gt_path_nii)
                    results.append([case, float(dsc), float(hd95)])
                    logf.write(f"{case}: [affine] dice={dsc:.4f} hd95={hd95:.2f} shape(gt/pred_res)={gt_shape}/{pr_shape} pos(gt/pred)={gt_pos}/{pr_pos}\n")
                except Exception as e:
                    logf.write(f"[ERR] {case}: fallo affine-aware ({e})\n")
                    skipped += 1
                continue

            # Camino fullres en .npy (GT preferida) o evaluación en la rejilla preprocesada
            if not os.path.isfile(gt_path_npy):
                logf.write(f"[SKIP] {case}: GT no encontrada ni en fullres ({gt_path_npy}) ni como NIfTI ({gt_path_nii})\n")
                skipped += 1
                continue

            try:
                gt = np.load(gt_path_npy)
                if gt.ndim == 4:  # (1,D,H,W)
                    gt = gt[0]
                gt = _as_uint8_01(gt)

                if kind == "npy":
                    pr = np.load(pred_path)
                    if pr.ndim == 4:
                        pr = pr[0]
                    pr = _binarize_pred(pr)
                    # shapes deben coincidir normalmente (misma rejilla preprocesada)
                    if pr.shape != gt.shape:
                        # fallback: recorte/pad centrado simple (evitar dependencias)
                        D, H, W = gt.shape
                        d, h, w = pr.shape
                        # pad or crop a la forma GT
                        def _fit(a, target):
                            z = np.zeros(target, dtype=a.dtype)
                            s = tuple(max(0, (t - a.shape[i]) // 2) for i, t in enumerate(target))
                            e = tuple(min(a.shape[i], s[i] + target[i]) for i in range(3))
                            zs = tuple(0 for _ in range(3))
                            ze = tuple(zs[i] + (e[i] - (s[i])) for i in range(3))
                            z[zs[0]:ze[0], zs[1]:ze[1], zs[2]:ze[2]] = a[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
                            return z
                        pr = _fit(pr, gt.shape)
                else:
                    # pred es NIfTI pero GT no lo es (raro); convertir data del NIfTI y adaptar shape sin affine
                    pr_nii = nib.load(pred_path)
                    pr = _binarize_pred(pr_nii.get_fdata())
                    if pr.shape != gt.shape:
                        # mismo fallback de pad/crop centrado
                        def _fit(a, target):
                            z = np.zeros(target, dtype=a.dtype)
                            s = tuple(max(0, (t - a.shape[i]) // 2) for i, t in enumerate(target))
                            e = tuple(min(a.shape[i], s[i] + target[i]) for i in range(3))
                            zs = tuple(0 for _ in range(3))
                            ze = tuple(zs[i] + (e[i] - (s[i])) for i in range(3))
                            z[zs[0]:ze[0], zs[1]:ze[1], zs[2]:ze[2]] = a[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
                            return z
                        pr = _fit(pr, gt.shape)

                # spacing desde meta si existe, si no (1,1,1)
                meta = _find_meta_file(fullres_dir, case)
                spacing = _read_spacing_from_meta(meta) or (1.0, 1.0, 1.0)

                dsc, hd95 = _dice_hd95(gt, pr, spacing=spacing)
                results.append([case, float(dsc), float(hd95)])
                logf.write(f"{case}: [fullres] dice={dsc:.4f} hd95={hd95:.2f} shape(gt/pred)={gt.shape}/{pr.shape} spacing={spacing}\n")

            except Exception as e:
                logf.write(f"[ERR] {case}: fallo fullres ({e})\n")
                skipped += 1

    if not results:
        print("❌ No se calcularon métricas (0 casos). Revisa rutas y nombres.")
        print(" 🪵 Log:", log_path)
        raise SystemExit(1)

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

    print("✅ Métricas calculadas")
    print("Casos:", len(arr), f"(saltados: {skipped})")
    print("Dice promedio:", float(arr[:, 0].mean()))
    print("HD95 promedio:", float(arr[:, 1].mean()))
    print("📄 Guardado en:")
    print(" -", os.path.join(args.out_dir, args.npy_name))
    print(" -", csv_path)
    print(" 🪵 Log:", log_path)

if __name__ == "__main__":
    main()
