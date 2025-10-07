import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch
import torch.nn.functional as F
from monai.utils import set_determinism
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from medpy import metric

set_determinism(123)

def load_itk(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # SimpleITK: array en (Z,Y,X) o (T,Z,Y,X)

    # Asegurar que quede en 3D (Z,Y,X)
    if arr.ndim == 4:
        # casos: (T,Z,Y,X) o (C,Z,Y,X). Si T/C==1, squeeze; si no, toma el primer canal/tiempo.
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            # Si guardaste one-hot/probas en canales, normalmente querrás argmax antes de guardar.
            # Aquí tomamos el canal 0 por robustez:
            arr = arr[0]
    elif arr.ndim == 2:
        # Imagen 2D → empaqueta una dimensión Z ficticia
        arr = arr[None, :, :]

    # Spacing puede tener longitud != 3 en algunos NIfTI “minimalistas”
    spacing = img.GetSpacing()  # ITK: (sx, sy, sz) para 3D; puede ser de largo 0/1/2/4
    if len(spacing) >= 3:
        voxelspacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    elif len(spacing) == 2:
        voxelspacing_zyx = (1.0, float(spacing[1]), float(spacing[0]))
    elif len(spacing) == 1:
        voxelspacing_zyx = (float(spacing[0]), float(spacing[0]), float(spacing[0]))
    else:
        voxelspacing_zyx = (1.0, 1.0, 1.0)  # fallback razonable

    return arr.astype(np.uint8), voxelspacing_zyx


def to_numpy_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.astype(np.uint8)

def ensure_same_shape(pred_np, gt_np):
    import torch, numpy as np, torch.nn.functional as F

    def to_3d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 2:  arr = arr[None, ...]
        elif arr.ndim == 4:
            # colapsa canales si los hay
            arr = np.argmax(arr, axis=0) if arr.shape[0] > 1 else arr[0]
        return arr.astype(np.uint8)

    pred_np = to_3d(pred_np)
    gt_np   = to_3d(gt_np)

    if pred_np.shape == gt_np.shape:
        return pred_np, gt_np

    # Re-muestrea PRED a la forma del GT (nearest)
    pred_t = torch.from_numpy(pred_np)[None, None].float()
    pred_t = F.interpolate(pred_t, size=gt_np.shape, mode="nearest")
    pred_np2 = pred_t.squeeze(0)..squeeze(0).byte().numpy()
    return pred_np2, gt_np


def cal_metric_binary(gt, pred, voxel_spacing):
    # gt/pred: (D,H,W) con {0,1}
    if pred.sum() > 0 and gt.sum() > 0:
        dc = metric.binary.dc(pred, gt)
        hd = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dc, hd], dtype=np.float32)
    else:
        # si alguna está vacía, definimos Dice=0, HD95=50 (o nan/valor centinela)
        return np.array([0.0, 50.0], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Tu carpeta fullres (p.ej. /home/.../data/colorectal/fullres/colorectal)")
    ap.add_argument("--pred_dir", required=True, help="Carpeta con predicciones NIfTI (p.ej. /home/.../prediction_results/segmamba)")
    ap.add_argument("--out_dir", default=None, help="Dónde guardar el .npy de métricas (por defecto: <pred_dir>/../result_metrics)")
    ap.add_argument("--use_seg_path", action="store_true",
                    help="Usa properties['seg_path'] si está disponible para leer GT desde disco; si no, usa batch['seg']")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.pred_dir), "result_metrics")
    os.makedirs(args.out_dir, exist_ok=True)

    # Cargamos split desde tu loader (así obtenemos 'properties' y opcionalmente batch['seg'])
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)

    results = []
    case_names = []

    for batch in tqdm(test_ds, total=len(test_ds)):
        props = batch["properties"]
        case_name = props["name"][0] if isinstance(props["name"], (list, tuple)) else props["name"]
        pred_path = os.path.join(args.pred_dir, f"{case_name}.nii.gz")
        if not os.path.exists(pred_path):
            # intenta sin .gz
            pred_path = os.path.join(args.pred_dir, f"{case_name}.nii")
            if not os.path.exists(pred_path):
                print(f"[WARN] No se encontró pred para {case_name}")
                continue

        pred_np, pred_vox = load_itk(pred_path)
        # Si tu save_to_nii guardó clase única, pred_np será (D,H,W) con {0,1} (ideal).
        # Si guardaste probas o one-hot, conviértelo a binario aquí.
        if pred_np.ndim == 4:  # e.g. (C,D,H,W) o (D,H,W,C)
            # asumimos canal en 0 ⇒ (C,D,H,W)
            if pred_np.shape[0] > 1:
                pred_np = np.argmax(pred_np, axis=0)
            else:
                pred_np = (pred_np[0] > 0.5).astype(np.uint8)

        pred_np = pred_np.astype(np.uint8)

        # Ground truth
        if args.use_seg_path and "seg_path" in props:
            gt_path = props["seg_path"][0] if isinstance(props["seg_path"], (list, tuple)) else props["seg_path"]
            gt_np, gt_vox = load_itk(gt_path)
            gt_voxspacing = gt_vox
        else:
            gt_t = batch["seg"][0, 0]
            gt_np = to_numpy_uint8(gt_t)
            gt_voxspacing = pred_vox

        if pred_np.ndim > 3:
            if pred_np.shape[0] > 1:
                pred_np = np.argmax(pred_np, axis=0)
            else:
                pred_np = np.squeeze(pred_np, axis=0)
        elif pred_np.ndim == 2:
            pred_np = pred_np[None, :, :]
        
        pred_np = (pred_np > 0).astype(np.uint8)
        if gt_np.ndim > 3:
            if gt_np.shape[0] > 1:
                gt_np = np.argmax(gt_np, axis=0)
            else:
                gt_np = np.squeeze(gt_np, axis=0)
        elif gt_np.ndim == 2:
            gt_np = gt_np[None, :, :]
        
        gt_np = (gt_np > 0).astype(np.uint8)
        pred_np, gt_np = ensure_same_shape(pred_np, gt_np)

        # Métricas binario
        props_spacing = None
        if isinstance(props, dict):
            s = props.get("spacing", None)  # típicamente [sx, sy, sz] o [z,y,x]
            if s is not None:
                # asegúrate de usar orden Z,Y,X
                if len(s) >= 3:
                    props_spacing = (float(s[2]), float(s[1]), float(s[0]))
        
        voxelspacing = props_spacing or gt_voxspacing or pred_vox
        m = cal_metric_binary(gt_np, pred_np, voxelspacing)
        results.append(m)
        case_names.append(case_name)

    if len(results) == 0:
        print("No se calcularon métricas. ¿Rutas correctas?")
        return

    results = np.stack(results, axis=0)  # [N, 2] con (dice, hd95)
    out_npy = os.path.join(args.out_dir, "colorectal_metrics.npy")
    np.save(out_npy, results)
    
    import csv
    csv_path = os.path.join(args.out_dir, "colorectal_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "dice", "hd95"])
        for name, (dice, hd) in zip(case_names, results):
            w.writerow([name, float(dice), float(hd)])
    print(f"Guardado CSV: {csv_path}")
    
    mean = results.mean(axis=0)
    std = results.std(axis=0)
    print(f"Guardado: {out_npy}")
    print(f"Mean  [Dice, HD95]: {mean}")
    print(f"Std   [Dice, HD95]: {std}")

if __name__ == "__main__":
    main()
