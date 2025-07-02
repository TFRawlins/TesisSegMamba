import os
import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from monai.transforms import Resize
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from model.segmentation_model import SegMambaModel  # <-- Ajusta esto si tu clase tiene otro nombre

def main():
    # =================== Configuración ===================
    model_path = "./ckpts_seg/best_model.pt"
    data_dir = "./data/fullres/train"
    save_path = "./prediction_results/segmamba"
    os.makedirs(save_path, exist_ok=True)

    # ============ Cargar modelo =============
    model = SegMambaModel(
        depths=[2, 2, 2, 2],
        dims=[24, 48, 96, 192]
    )
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.eval()
    model.cuda()

    # ============ Obtener datos =============
    _, _, test_ds = get_train_val_test_loader_from_train(
        data_dir=data_dir,
        train_rate=0.7,
        val_rate=0.1,
        test_rate=0.2,
        seed=42
    )

    # ============ Inferencia ============
    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5)

    with torch.no_grad():
        for i in tqdm(range(len(test_ds))):
            batch = test_ds[i]
            image = batch["data"].unsqueeze(0).cuda()  # (1, C, D, H, W)
            props = batch["properties"]

            pred = inferer(inputs=image, network=model)
            pred = torch.argmax(pred, dim=1).cpu().numpy()[0]  # (D, H, W)

            # Guardar como .npy (o puedes reconstruir a .nii.gz usando SimpleITK si quieres)
            name = props.get("name", f"case_{i}")
            np.save(os.path.join(save_path, f"{name}_pred.npy"), pred)

    print(f"✅ Inferencia completada. Resultados guardados en: {save_path}")

if __name__ == "__main__":
    main()
