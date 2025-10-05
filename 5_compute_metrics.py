#!/usr/bin/env python3
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from monai.utils import set_determinism
import torch
import os
import numpy as np
import SimpleITK as sitk
from medpy import metric
import argparse
from tqdm import tqdm
from monai.transforms import Resize

set_determinism(123)
torch.set_num_threads(max(1, os.cpu_count() // 2))

def cal_metric(gt_bool: np.ndarray, pred_bool: np.ndarray, voxel_spacing):
    """
    gt_bool, pred_bool: booleanos (D, H, W)
    voxel_spacing: lista/tupla de 3 floats (z, y, x)
    """
    if pred_bool.sum() > 0 and gt_bool.sum() > 0:
        dsc = metric.binary.dc(pred_bool, gt_bool)
        hd95 = metric.binary.hd95(pred_bool, gt_bool, voxelspacing=tuple(voxel_spacing))
        return np.array([dsc, hd95], dtype=np.float32)
    else:
        return np.array([0.0, 50.0], dtype=np.float32)

def load_prediction(pred_dir: str, case_name: str) -> np.ndarray:
    """
    Carga la predicción del caso como máscara binaria (D,H,W) np.uint8 {0,1}.
    Busca primero NIfTI y luego .npy.
    """
    nii_path = os.path.join(pred_dir, f"{case_name}.nii.gz")
    npy_path = os.path.join(pred_dir, f"{case_name}_pred.npy")

    if os.path.exists(nii_path):
        img = sitk.ReadImage(nii_path)
        arr = sitk.GetArrayFromImage(img)  # (D,H,W) o (C,D,H,W)
        # Si viniera one-hot en NIfTI (poco probable), colapsar
        if arr.ndim == 4:
            arr = np.argmax(arr, axis=0)
        arr = arr.astype(np.uint8)
        if arr.max() > 1:
            arr = (arr > 0).astype(np.uint8)
        return arr

    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        if arr.ndim == 4:
            # Soportar (1,D,H,W), (C,D,H,W) o (D,H,W,1)
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = np.argmax(arr, axis=0)
        arr = arr.astype(np.uint8)
        if arr.max() > 1:
            arr = (arr > 0).astype(np.uint8)
        return arr

    raise FileNotFoundError(f"No se encontró predicción para {case_name} en {pred_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", required=True, type=str,
                        help="Nombre de la carpeta dentro de prediction_results con las predicciones")
    parser.add_argument("--data_dir", default="./data/fullres/train",
                        help="Ruta a data/fullres/<dataset> usada por el loader")
    parser.add_argument("--results_root", default="prediction_results",
                        help="Raíz donde se escriben y leen predicciones y métricas")
    args = parser.parse_args()

    pred_name = args.pred_name
    pred_dir = os.path.join(args.results_root, pred_name)
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"No existe el directorio de predicciones: {pred_dir}")

    # Misma partición/split que el pipeline
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    print("Total casos test:", len(test_ds))

    # Resultados: (N, 1, 2) -> 1 clase (lesión), 2 métricas (Dice, HD95)
    all_results = np.zeros((len(test_ds), 1, 2), dtype=np.float32)

    # Redimensionador (por si hay mismatch leve por I/O), nearest para máscaras
    resizer = Resize(spatial_size=None, mode="nearest")

    for i, batch in enumerate(tqdm(test_ds, total=len(test_ds))):
        properties = batch["properties"]
        case_name = properties["name"][0] if isinstance(properties["name"], (list, tuple)) else properties["name"]
        spacing = properties.get("spacing", [1, 1, 1])
        try:
            pred = load_prediction(pred_dir, case_name)  # (D,H,W) uint8 {0,1}
        except Exception as e:
            print(f"⚠️ {case_name}: {e} -> se omite")
            continue

        # GT desde el loader (ROI 192³)
        if "seg" not in batch or batch["seg"] is None:
            print(f"⚠️ {case_name}: batch no trae 'seg' -> se omite")
            continue
        gt = batch["seg"][0, 0].detach().cpu().numpy()  # (D,H,W)
        gt = gt.astype(np.uint8)
        gt[gt == 255] = 0
        gt = (gt == 1).astype(np.uint8)

        # Asegurar shapes iguales (en teoría ya están en 192³, pero por si acaso)
        if tuple(pred.shape) != tuple(gt.shape):
            # usar MONAI Resize con canal ficticio
            pred_t = torch.as_tensor(pred, dtype=torch.float32)[None, ...]      # (1,D,H,W)
            resizer.spatial_size = gt.shape                                     # set target
            pred_rs = resizer(pred_t).squeeze(0).numpy().astype(np.uint8)       # (D,H,W)
        else:
            pred_rs = pred

        # Booleanos
        pred_bool = pred_rs.astype(bool)
        gt_bool = gt.astype(bool)

        m = cal_metric(gt_bool, pred_bool, spacing)
        all_results[i, 0, :] = m

    # Guardado y estadísticas
    os.makedirs(f"./{args.results_root}/result_metrics/", exist_ok=True)
    out_npy = f"./{args.results_root}/result_metrics/{pred_name}.npy"
    np.save(out_npy, all_results)

    # Reporte
    valid_rows = (all_results.sum(axis=(1, 2)) != 0)  # casos con algo calculado
    used = all_results[valid_rows]
    print("Shape resultados:", all_results.shape)
    if used.size > 0:
        print("Media [Dice, HD95]:", used.mean(axis=0)[0])
        print("Std   [Dice, HD95]:", used.std(axis=0)[0])
    else:
        print("⚠️ No hubo casos válidos.")

if __name__ == "__main__":
    main()
