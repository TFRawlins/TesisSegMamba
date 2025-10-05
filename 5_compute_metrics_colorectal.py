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
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = img.GetSpacing()         # x,y,z en ITK; ojo orden
    # reordenamos a z,y,x para voxelspacing de medpy (voxelspacing=D,H,W)
    voxelspacing_zyx = (spacing[2], spacing[1], spacing[0])
    return arr, voxelspacing_zyx

def to_numpy_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.astype(np.uint8)

def ensure_same_shape(pred_np, gt_np):
    if pred_np.shape == gt_np.shape:
        return pred_np, gt_np
    # re-muestrea GT a la forma del pred (nearest para máscaras)
    gt_t = torch.from_numpy(gt_np)[None, None].float()  # [1,1,D,H,W]
    gt_t = F.interpolate(gt_t, size=pred_np.shape, mode="nearest")
    gt_np2 = gt_t.squeeze().byte().numpy()
    return pred_np, gt_np2

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
            # usamos el GT del batch (más simple y consistente)
            gt_t = batch["seg"][0, 0]  # (B,1,D,H,W) → (D,H,W)
            gt_np = to_numpy_uint8(gt_t)
            # usamos el spacing del pred (suficiente para HD)
            gt_voxspacing = pred_vox

        # Asegurar misma forma
        pred_np, gt_np = ensure_same_shape(pred_np, gt_np)

        # Métricas binario
        m = cal_metric_binary(gt_np, pred_np, pred_vox)  # usa spacing del pred
        results.append(m)
        case_names.append(case_name)

    if len(results) == 0:
        print("No se calcularon métricas. ¿Rutas correctas?")
        return

    results = np.stack(results, axis=0)  # [N, 2] con (dice, hd95)
    out_npy = os.path.join(args.out_dir, "colorectal_metrics.npy")
    np.save(out_npy, results)

    mean = results.mean(axis=0)
    std = results.std(axis=0)
    print(f"Guardado: {out_npy}")
    print(f"Mean  [Dice, HD95]: {mean}")
    print(f"Std   [Dice, HD95]: {std}")

if __name__ == "__main__":
    main()
