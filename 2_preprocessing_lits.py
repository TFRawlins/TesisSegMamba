# 2_preprocessing_colorectal.py  (versión alineada)
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor
import numpy as np, json, os, shutil, re
from pathlib import Path
import nibabel as nib

# === RUTAS DE ENTRADA (nnU-Net RAW) ===
NNUNET_BASE = Path("/home/trawlins/tesis/data/LITS/raw")
IMAGES_TR = NNUNET_BASE / "images"
LABELS_TR = NNUNET_BASE / "masks"

# === STAGING (crudos por-caso estilo repo original) ===
base_dir  = "/home/trawlins/tesis/data/LITS/raw_data"
image_dir = "LITS"
data_filename = ["image.nii.gz"]
seg_filename  = "seg.nii.gz"

# === SALIDA FULLRES (arrays + properties) ===
OUTPUT_FULLRES_ROOT = Path("/home/trawlins/tesis/data/LITS/fullres/lits")
# (Opcional) lugar donde están tus listas de folds paciente-wise:
FOLD_LISTS_DIR = Path("/home/trawlins/tesis/data/LITS/fold_lists")  # fold{0..4}_{train,val}.txt

def prepare_from_nnunet_to_rawdata(dryrun=False):
    out_root = Path(base_dir) / image_dir
    out_root.mkdir(parents=True, exist_ok=True)
    if not IMAGES_TR.exists() or not LABELS_TR.exists():
        raise RuntimeError(f"Rutas no existen:\n  IMAGES_TR={IMAGES_TR}\n  LABELS_TR={LABELS_TR}")

        # Buscar imágenes en varios formatos posibles
    imgs = sorted(IMAGES_TR.glob("*_0000.nii.gz"))
    if not imgs:
        imgs = sorted(IMAGES_TR.glob("*.nii.gz"))
    if not imgs:
        imgs = sorted(IMAGES_TR.glob("*_0000.nii"))
    if not imgs:
        imgs = sorted(IMAGES_TR.glob("*.nii"))

    lbls = sorted(LABELS_TR.glob("*.nii.gz"))
    if not lbls:
        lbls = sorted(LABELS_TR.glob("*.nii"))


    num_re = re.compile(r"(\d+)")
    def id_from_img(p: Path) -> str:
        m = num_re.search(p.name);  return m.group(1) if m else p.stem.split("_")[0]
    def id_from_lbl(p: Path) -> str:
        m = num_re.search(p.stem);  return m.group(1) if m else p.stem

    imgs_by, lbls_by = {}, {id_from_lbl(p): p for p in lbls}
    for p in imgs:
        cid = id_from_img(p)
        if cid in imgs_by:
            if p.name.endswith("_0000.nii.gz"):
                imgs_by[cid] = p
        else:
            imgs_by[cid] = p
    common = sorted(set(imgs_by).intersection(lbls_by))
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

def median_spacing_from_nnunet_raw() -> tuple[float, float, float]:
    """Lee spacings de imagesTr y devuelve mediana (Z,Y,X)."""
    imgs = sorted(IMAGES_TR.glob("*_0000.nii.gz")) or sorted(IMAGES_TR.glob("*.nii.gz"))
    spacings = []
    for p in imgs:
        hdr = nib.load(str(p)).header
        # nibabel da pixdim en orden (x,y,z); devolvemos (Z,Y,X) para MONAI/nuestro uso
        sx, sy, sz = map(float, hdr.get_zooms()[:3])
        spacings.append([sz, sy, sx])
    med = np.median(np.asarray(spacings), axis=0).tolist()
    return tuple(float(x) for x in med)

def plan():
    pre = MultiModalityPreprocessor(
        base_dir=base_dir, image_dir=image_dir,
        data_filenames=data_filename, seg_filename=seg_filename
    )
    preprocessor_plan = pre.run_plan()
    # (Se puede guardar/inspeccionar si quieres)

def process_train(target_spacing: tuple[float,float,float]):
    pre = MultiModalityPreprocessor(
        base_dir=base_dir, image_dir=image_dir,
        data_filenames=data_filename, seg_filename=seg_filename
    )
    OUTPUT_FULLRES_ROOT.mkdir(parents=True, exist_ok=True)
    pre.run(
        output_spacing=list(target_spacing),   # ej (1.5,1.5,3.0)
        output_dir=str(OUTPUT_FULLRES_ROOT),
        all_labels=[1],
    )

def export_manifest_and_folds():
    """Escribe manifest.json y duplica tus fold lists apuntando al fullres."""
    # Manifest simple: asume estructura de salida del preprocesador
    manifest = {}
    for case_dir in sorted((Path(base_dir)/image_dir).glob("*")):
        if not case_dir.is_dir(): 
            continue
        cid = case_dir.name
        manifest[cid] = {
            "raw_image": str(case_dir / data_filename[0]),
            "raw_label": str(case_dir / seg_filename),
            # Punteros típicos de salida (ajusta si tu preprocesador usa otra convención)
            "fullres_dir": str(OUTPUT_FULLRES_ROOT),
        }
    (OUTPUT_FULLRES_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Re-escribir folds → listas en fullres (si ya las tienes paciente-wise)
    if FOLD_LISTS_DIR.exists():
        out_lists = OUTPUT_FULLRES_ROOT / "fold_lists"
        out_lists.mkdir(parents=True, exist_ok=True)
        for p in sorted(FOLD_LISTS_DIR.glob("fold*_*.txt")):
            lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            # Deja solo IDs; el loader por npy/npz resolverá rutas desde OUTPUT_FULLRES_ROOT
            (out_lists / p.name).write_text("\n".join(lines) + "\n")
        print(f"[folds] Exportadas listas a: {out_lists}")

if __name__ == "__main__":
    # 1) staging desde nnU-Net raw
    prepare_from_nnunet_to_rawdata(dryrun=False)

    # 2) plan (stats)
    plan()

    # 3) escoger spacing objetivo desde mediana del dataset
    med_spacing = median_spacing_from_nnunet_raw()
    # Ajuste pragmatico: no fuerces Z muy fino; redondea a 1–2 mm si Z<1, y a 2–3 mm si Z>3
    z, y, x = med_spacing
    target_spacing = (max(1.2, min(z, 3.0)), max(1.0, y), max(1.0, x))
    print(f"[spacing] mediana (Z,Y,X)={med_spacing} → target={target_spacing}")

    # 4) preprocesar a fullres
    process_train(target_spacing=target_spacing)

    # 5) manifest + fold lists apuntando a fullres
    export_manifest_and_folds()
