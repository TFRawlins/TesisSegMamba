import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from model_segmamba.segmamba import SegMamba

torch.backends.cudnn.benchmark = True

def best_case_name(props, i):
    for k in ("name", "case_id", "id", "basename"):
        if isinstance(props, dict) and k in props and props[k]:
            return str(props[k])
    return f"case_{i:04d}"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="/home/trawlins/tesis/ckpts_seg/best_model.pt")
    p.add_argument("--data_dir",  type=str, default="/home/trawlins/tesis/data/fullres/train")
    p.add_argument("--save_path", type=str, default="/home/trawlins/tesis/prediction_results/segmamba")
    p.add_argument("--roi", nargs=3, type=int, default=[128,128,128], help="ROI D H W (sliding window)")
    p.add_argument("--sw_batch_size", type=int, default=2, help="ventanas por paso en inferencia")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--train_rate", type=float, default=0.7)
    p.add_argument("--val_rate",   type=float, default=0.1)
    p.add_argument("--test_rate",  type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_dataparallel", action="store_true", help="usar todas las GPUs disponibles")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"🟩 Device: {device} | GPUs: {n_gpus}")

    # === Modelo (mismos canales que en train) ===
    model = SegMamba(
        in_chans=1,
        out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[24, 48, 96, 192]
    )

    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    if args.use_dataparallel and n_gpus > 1:
        print("✅ DataParallel activado para inferencia")
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))

    # === Dataset de test (usar mismos ratios/seed que en train) ===
    _, _, test_dataset = get_train_val_test_loader_from_train(
        data_dir=args.data_dir,
        train_rate=args.train_rate,
        val_rate=args.val_rate,
        test_rate=args.test_rate,
        seed=args.seed
    )

    inferer = SlidingWindowInferer(
        roi_size=tuple(args.roi), sw_batch_size=args.sw_batch_size, overlap=args.overlap
    )

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Inferencia"):
            sample = test_dataset[i]
            image = sample["data"].unsqueeze(0).to(device, non_blocking=True)  # (1, C, D, H, W)
            props = sample.get("properties", {})
            logits = inferer(inputs=image, network=model)              # [1, C, D, H, W]
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu()     # [C, D, H, W] (torch)
            pred   = torch.argmax(probs, dim=0).cpu().numpy()          # [D, H, W]   (numpy)
            
            case_name = best_case_name(props, i)                       # <-- usa tu helper actual
            os.makedirs(args.save_path, exist_ok=True)
            
            # Probs: si quieres ahorrar espacio, podrías castear a float16
            np.save(os.path.join(args.save_path, f"{case_name}_probs.npy"), probs.numpy())
            np.save(os.path.join(args.save_path, f"{case_name}_pred.npy"),  pred)
            print(f"\n✅ Inferencia completada. Resultados en: {args.save_path}")

if __name__ == "__main__":
    main()
