import os
import argparse
import torch
import numpy as np
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.prediction import Predictor
from light_training.evaluation.metric import dice

set_determinism(123)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Ruta a data/fullres/<dataset>")
parser.add_argument("--ckpt", required=True, help="Ruta al best_model.pt")
parser.add_argument("--save_dir", default="/home/trawlins/tesis/prediction_results/segmamba")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--roi", type=int, nargs=3, default=[128,128,128])
parser.add_argument("--sw_batch_size", type=int, default=1)
parser.add_argument("--overlap", type=float, default=0.5)
parser.add_argument("--mirror_axes", type=int, nargs="*", default=[0,1,2])
args = parser.parse_args()

class ColorectalPredict(Trainer):
    def __init__(self, device="cuda:0"):
        super().__init__(
            env_type="pytorch", max_epochs=1, batch_size=1, device=device,
            val_every=1, num_gpus=1, logdir="", master_ip="localhost",
            master_port=17750, training_script=__file__
        )
        self.patch_size = args.roi
        self.augmentation = False

    def get_input(self, batch):
        # batch["data"] -> (B, 1, D, H, W); batch["seg"] puede venir o no en test
        return batch["data"], batch.get("seg", None), batch["properties"]

    def define_model(self):
        from model_segmamba.segmamba import SegMamba

        model = SegMamba(
            in_chans=1,      # CT de 1 canal
            out_chans=2,     # binario (fondo/lesión). Si fue 1-logit, cambia lógica abajo.
            depths=[2,2,2,2],
            feat_size=[48, 96, 192, 384]
        )
        sd = torch.load(args.ckpt, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "module" in sd:
            sd = sd["module"]
        sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
        model.load_state_dict(sd, strict=False)
        model.eval()

        inferer = SlidingWindowInferer(
            roi_size=args.roi,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            progress=True,
            mode="gaussian",
        )
        predictor = Predictor(window_infer=inferer, mirror_axes=args.mirror_axes)

        os.makedirs(args.save_dir, exist_ok=True)
        return model, predictor

    @torch.no_grad()
    def validation_step(self, batch):
        import torch.nn.functional as F

        image, label, properties = self.get_input(batch)
        model, predictor = self.define_model()

        with torch.cuda.amp.autocast(enabled=True):
            logits = predictor.maybe_mirror_and_predict(image, model, device=args.device)
            probs = predictor.predict_raw_probability(logits, properties=properties)

        pred_roi = probs.argmax(dim=0, keepdim=True)

        if label is not None:
            # label: (B,1,D,H,W) → (D,H,W)
            gt_roi = label[0, 0].detach().cpu().numpy().astype(np.uint8)
            pr_roi = pred_roi[0].detach().cpu().numpy().astype(np.uint8)


            if pr_roi.shape != gt_roi.shape:
                pr_t = torch.from_numpy(pr_roi)[None, None].float()
                pr_t = F.interpolate(pr_t, size=gt_roi.shape, mode="nearest")
                pr_roi = pr_t.squeeze().byte().numpy()

            print(f"[ROI] Dice clase 1: {dice(pr_roi, gt_roi):.4f}")

        fullres = predictor.predict_noncrop_probability(pred_roi, properties)
        predictor.save_to_nii(
            fullres,
            raw_spacing=[1,1,1],
            case_name=properties["name"][0],
            save_dir=args.save_dir,
        )

        return 0

if __name__ == "__main__":
    trainer = ColorectalPredict(device=args.device)
    _, _, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    trainer.validation_single_gpu(test_ds)
