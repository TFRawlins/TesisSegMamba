import os
import torch
import numpy as np
from monai.inferers import SlidingWindowInferer
from light_training.prediction import Predictor
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from monai.utils import set_determinism

set_determinism(123)

data_dir = "/workspace/data/content/data/fullres/train"
patch_size = [64, 64, 64]
save_path = "./prediction_results/segmamba_liver"
model_path = "./ckpts_seg/best_model.pt"
device = "cuda:0"

os.makedirs(save_path, exist_ok=True)

# 1. Cargar modelo
from model_segmamba.segmamba import SegMamba

model = SegMamba(
    in_chans=1,
    out_chans=2,
    depths=[1, 1, 2, 2],
    feat_size=[16, 32, 64, 128]
)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
model.to(device)

# 2. Inferencia sliding window
inferer = SlidingWindowInferer(
    roi_size=patch_size,
    sw_batch_size=1,
    overlap=0.5,
    progress=True,
    mode="gaussian"
)

predictor = Predictor(window_infer=inferer, mirror_axes=[0,1,2])

# 3. Obtener dataset test
_, _, test_ds = get_train_val_test_loader_from_train(data_dir)

# 4. Inferencia por muestra
for batch in test_ds:
    data = batch["data"].unsqueeze(0).to(device)
    properties = batch["properties"]

    with torch.no_grad():
        pred = predictor.maybe_mirror_and_predict(data, model, device=device)
        pred = predictor.predict_raw_probability(pred, properties)
        pred = torch.argmax(pred, dim=0).unsqueeze(0)  # (1, D, H, W)
        pred = predictor.predict_noncrop_probability(pred, properties)
        predictor.save_to_nii(pred, raw_spacing=[1,1,1], case_name=properties["name"][0], save_dir=save_path)

print("✅ Inferencia completada. Resultados guardados en:", save_path)
