from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor
import numpy as np
import pickle
import json
from pathlib import Path
import shutil

data_filename = ["image.nii.gz"]
seg_filename = "seg.nii.gz"

base_dir = "./data/raw_data/"
image_dir = "ColorectalVessels"

NNUNET_BASE = Path.home() / "tesis" / "data_nnUnet" / "nnUNet_raw" / "Dataset001_Colorectal"
IMAGES_TR = NNUNET_BASE / "imagesTr"
LABELS_TR = NNUNET_BASE / "labelsTr"

def prepare_from_nnunet_to_rawdata():
    out_root = Path(base_dir) / image_dir
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMAGES_TR.glob("*_0000.nii.gz"))
    lbls = sorted(LABELS_TR.glob("*.nii.gz"))

    def img_id(p): return p.name.split("_")[0]
    def lbl_id(p): return p.stem

    imgs_by = {img_id(p): p for p in imgs}
    lbls_by = {lbl_id(p): p for p in lbls}
    common = sorted(set(imgs_by).intersection(lbls_by))

    if not common:
        raise RuntimeError(f"No se encontraron pares en {IMAGES_TR} y {LABELS_TR}")

    for cid in common:
        case_dir = out_root / cid
        case_dir.mkdir(parents=True, exist_ok=True)

        img_dst = case_dir / data_filename[0]
        seg_dst = case_dir / seg_filename
        if img_dst.exists(): img_dst.unlink()
        if seg_dst.exists(): seg_dst.unlink()

        shutil.copy2(imgs_by[cid], img_dst)
        shutil.copy2(lbls_by[cid], seg_dst)

    print(f"[staging] Casos preparados: {len(common)} en {out_root}")


def process_train():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "./data/fullres/train/"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3],
    )

def plan():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    
    preprocessor.run_plan()


if __name__ == "__main__":
    prepare_from_nnunet_to_rawdata()
    plan()
    process_train()
