import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

case = "1041"
pred_p = f"/home/trawlins/tesis/prediction_results/colorectal_folds/fold1/{case}.nii.gz"
gt_p   = f"/home/trawlins/tesis/data_nnUnet/nnUNet_raw/Dataset001_Colorectal/labelsTr/{case}.nii.gz"

pred_nii = nib.load(pred_p)
gt_nii   = nib.load(gt_p)

# Resamplea la predicción A LA REJILLA DEL GT usando affine (nearest, sin suavizado)
pred_res = resample_from_to(pred_nii, gt_nii, order=0)

pred_bin = (pred_res.get_fdata() > 0).astype(np.uint8)
gt_bin   = (gt_nii.get_fdata()   > 0).astype(np.uint8)

inter = (pred_bin & gt_bin).sum()
dice  = 2*inter / (pred_bin.sum() + gt_bin.sum() + 1e-8)
print("Shapes (pred_res, gt):", pred_bin.shape, gt_bin.shape)
print("Dice affine-aware:", dice)
