import nibabel as nib
import numpy as np
from pathlib import Path

case = "1041"
pred_p = Path("/home/trawlins/tesis/prediction_results/colorectal_folds/fold1")/f"{case}.nii.gz"
gt_p   = Path("/home/trawlins/tesis/data_nnUnet/nnUNet_raw/Dataset001_Colorectal/labelsTr")/f"{case}.nii.gz"

pred = nib.load(str(pred_p)); gt = nib.load(str(gt_p))
pred_arr = np.asarray(pred.dataobj)
gt_arr   = np.asarray(gt.dataobj)

print("Shapes", pred_arr.shape, gt_arr.shape)
print("Affines equal?", np.allclose(pred.affine, gt.affine))

# binariza de forma robusta: >0
pred_bin = (pred_arr > 0).astype(np.uint8)
gt_bin   = (gt_arr   > 0).astype(np.uint8)

inter = (pred_bin & gt_bin).sum()
dice  = 2*inter / (pred_bin.sum() + gt_bin.sum() + 1e-8)
print("Dice simple (voxel a voxel, mismo espacio):", dice)
