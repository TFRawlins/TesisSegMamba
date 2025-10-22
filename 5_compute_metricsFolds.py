from glob import glob

label_dir = os.path.join(args.data_dir, "labelsTr")
assert os.path.isdir(label_dir), f"No existe labelsTr en {label_dir}"

label_paths = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
print("Total casos test:", len(label_paths))

results = []
case_ids = []
skipped = 0

for lab_path in tqdm(label_paths, total=len(label_paths)):
    case = os.path.basename(lab_path).replace(".nii.gz", "")
    gt_xyz = nib.load(lab_path).get_fdata()
    gt = _ensure_3d(gt_xyz, "gt", case)
    if gt is None:
        skipped += 1
        continue
    gt = _as_uint8_01(gt)

    pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")
    if not os.path.exists(pred_path):
        print(f"[SKIP] {case}: pred no encontrada en {pred_path}")
        skipped += 1
        continue

    pred_xyz = nib.load(pred_path).get_fdata()  # (X,Y,Z) float
    if pred_xyz.ndim == 4:
        pred_xyz = pred_xyz[0] if pred_xyz.shape[0] == 1 else np.argmax(pred_xyz, axis=0)

    best_dice, best_hd, best_tag, best_arr = -1.0, 50.0, "", None
    for tag, cand in _orient_candidates(pred_xyz):
        cand = _binarize_pred_volume(cand.astype(np.float32))
        cand = _ensure_3d(cand, "pred_cand", case)
        if cand is None:
            continue
        if cand.shape != gt.shape:
            cand = _resize_to(cand, gt.shape)
        d, h = _dice_hd95(gt.astype(bool), cand.astype(bool), spacing=(1,1,1))
        if d > best_dice:
            best_dice, best_hd, best_tag, best_arr = d, h, tag, cand

    if best_arr is None:
        print(f"[SKIP] {case}: no se pudo construir candidato de orientación")
        skipped += 1
        continue

    results.append([best_dice, best_hd])
    case_ids.append(case)
    # si quieres log detallado:
    # log.append(f"[{case}] best={best_tag} dice={best_dice:.4f} hd95={best_hd:.2f}\n")
