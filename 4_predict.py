import os
import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from model_segmamba.segmamba import SegMamba

def main():
    # ==== Configuración general ====
    model_path = "./ckpts_seg/best_model.pt"
    data_dir = "./data/fullres/train"
    save_path = "./prediction_results/segmamba"
    os.makedirs(save_path, exist_ok=True)

    # ==== Crear el modelo y cargar pesos entrenados ====
    model = SegMamba(
        depths=[2, 2, 2, 2],
        feat_size=[24, 48, 96, 192]
    )
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.eval().cuda()

    # ==== Cargar datos de test ====
    _, _, test_dataset = get_train_val_test_loader_from_train(
        data_dir=data_dir,
        train_rate=0.7,
        val_rate=0.1,
        test_rate=0.2,
        seed=42
    )

    # ==== Configurar inferencia ====
    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5)

    # ==== Ejecutar inferencia ====
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            batch = test_dataset[i]
            image = batch["data"].unsqueeze(0).cuda()  # (1, C, D, H, W)
            props = batch["properties"]
            pred = inferer(inputs=image, network=model)
            pred = torch.argmax(pred, dim=1).cpu().numpy()[0]  # (D, H, W)

            case_name = props.get("name", f"case_{i}")
            save_file = os.path.join(save_path, f"{case_name}_pred.npy")
            np.save(save_file, pred)

    print(f"\n✅ Inferencia completada. Resultados guardados en {save_path}")

if __name__ == "__main__":
    main()
