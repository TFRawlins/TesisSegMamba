from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor
import numpy as np
import pickle
import json
import os, shutil, re
from pathlib import Path

NNUNET_BASE = Path("/home/trawlins/tesis/data_nnUnet/nnUNet_raw/Dataset001_Colorectal")
IMAGES_TR = NNUNET_BASE / "imagesTr"
LABELS_TR = NNUNET_BASE / "labelsTr"

base_dir  = "/home/trawlins/tesis/data/colorectal/raw_data/"
image_dir = "ColorectalVessels"

data_filename = ["image.nii.gz"]
seg_filename  = "seg.nii.gz"

OUTPUT_FULLRES_ROOT = "/home/trawlins/tesis/data/colorectal/fullres/colorectal"
OUT_ROOT     = Path(OUTPUT_FULLRES_ROOT)
OUT_TRAIN_IMG = OUT_ROOT / "train" / "images"
OUT_TRAIN_LBL = OUT_ROOT / "train" / "labels"
OUT_VAL_IMG   = OUT_ROOT / "val" / "images"
OUT_VAL_LBL   = OUT_ROOT / "val" / "labels"

def prepare_from_nnunet_to_rawdata(dryrun=False):

    out_root = Path(base_dir) / image_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if not IMAGES_TR.exists() or not LABELS_TR.exists():
        raise RuntimeError(f"Rutas no existen:\n  IMAGES_TR={IMAGES_TR}\n  LABELS_TR={LABELS_TR}")

    imgs = sorted(IMAGES_TR.glob("*_0000.nii.gz"))
    if not imgs:
        imgs = sorted(IMAGES_TR.glob("*.nii.gz"))
    lbls = sorted(LABELS_TR.glob("*.nii.gz"))

    num_re = re.compile(r"(\d+)")

    def id_from_img(p: Path) -> str:
        m = num_re.search(p.name)
        return m.group(1) if m else p.stem.split("_")[0]

    def id_from_lbl(p: Path) -> str:
        m = num_re.search(p.stem)
        return m.group(1) if m else p.stem

    imgs_by = {}
    for p in imgs:
        cid = id_from_img(p)
        if cid in imgs_by:
            if p.name.endswith("_0000.nii.gz"):
                imgs_by[cid] = p
        else:
            imgs_by[cid] = p

    lbls_by = {id_from_lbl(p): p for p in lbls}
    common = sorted(set(imgs_by).intersection(lbls_by))

    if not common:
        print("[debug] ejemplos imagesTr:", [p.name for p in imgs[:5]])
        print("[debug] ejemplos labelsTr:", [p.name for p in lbls[:5]])
        print("[debug] ids_img:", sorted(list(imgs_by.keys()))[:10])
        print("[debug] ids_lbl:", sorted(list(lbls_by.keys()))[:10])
        raise RuntimeError(f"No se encontraron pares en {IMAGES_TR} y {LABELS_TR}")

    print(f"[staging] Encontrados {len(common)} pares. dryrun={dryrun}")
    if dryrun:
        for cid in common[:10]:
            print(f"  {cid}: IMG={imgs_by[cid].name}  LBL={lbls_by[cid].name}")
        return

    for cid in common:
        case_dir = out_root / cid
        case_dir.mkdir(parents=True, exist_ok=True)

        img_dst = case_dir / data_filename[0]   # "image.nii.gz"
        seg_dst = case_dir / seg_filename       # "seg.nii.gz"

        if img_dst.exists(): img_dst.unlink()
        if seg_dst.exists(): seg_dst.unlink()

        shutil.copy2(imgs_by[cid], img_dst)
        shutil.copy2(lbls_by[cid], seg_dst)

    print(f"[staging] Casos preparados: {len(common)} en {out_root}")


def plan():
    preprocessor = MultiModalityPreprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=data_filename,
        seg_filename=seg_filename
    )
    preprocessor.run_plan()


def process_train():
    preprocessor = MultiModalityPreprocessor(
        base_dir=base_dir,
        image_dir=image_dir,
        data_filenames=data_filename,
        seg_filename=seg_filename
    )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = OUTPUT_FULLRES_ROOT

    preprocessor.run(
        output_spacing=out_spacing,
        output_dir=output_dir,
        all_labels=[1],
    )


if __name__ == "__main__":
    prepare_from_nnunet_to_rawdata(dryrun=False)
    plan()
    process_train()
