# Utilities for SJF_anUNetv3 training pipeline.
import os
from typing import Any, Tuple, cast
import numpy as np
import nibabel as nib
from nibabel.dataobj_images import DataobjImage
from nibabel.filebasedimages import FileBasedImage
from scipy import ndimage
from scipy.spatial import distance


def load_nifti(path: str) -> Tuple[np.ndarray, FileBasedImage]:
    img = cast(Any, nib).load(path)
    if not isinstance(img, DataobjImage):
        raise TypeError(f"Unsupported image type from nib.load: {type(img)!r}")
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    return data, img


def zscore_normalize(volume):
    mask = volume > 0
    if mask.sum() == 0:
        return (volume - volume.mean()) / (volume.std() + 1e-8)
    m = volume[mask]
    mean = m.mean()
    std = m.std() + 1e-8
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / std
    return out


def map_brats_labels(lbl):
    # Input label mapping as used in BraTS: {0,1,2,4} -> map to {0,1,2,3}
    out = np.zeros_like(lbl, dtype=np.uint8)
    out[lbl == 1] = 1
    out[lbl == 2] = 2
    out[lbl == 4] = 3
    return out


def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / (denom + eps)


def hd95(pred_mask, gt_mask, voxelspacing=(1.0, 1.0, 1.0)):
    # Compute Hausdorff distance 95 between binary volumes using surface point distances
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return np.nan

    def surface_voxels(mask):
        eroded = ndimage.binary_erosion(mask)
        return mask ^ eroded

    s_pred = surface_voxels(pred_mask)
    s_gt = surface_voxels(gt_mask)
    pred_pts = np.array(np.nonzero(s_pred)).T * voxelspacing
    gt_pts = np.array(np.nonzero(s_gt)).T * voxelspacing
    if pred_pts.size == 0 or gt_pts.size == 0:
        return np.nan
    d1 = distance.cdist(pred_pts, gt_pts)
    d2 = distance.cdist(gt_pts, pred_pts)
    hd1 = np.percentile(d1.min(axis=1), 95)
    hd2 = np.percentile(d2.min(axis=1), 95)
    return float(max(hd1, hd2))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
