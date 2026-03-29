# BraTSDataset copied into SJF_NOSAv3 for training.
import os
import random
from typing import Any, cast
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandSpatialCropd
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, RandRotate90d
from utils import zscore_normalize, map_brats_labels


class BraTSDataset(Dataset):
    def __init__(self, data_dir, keys=None, patch_size=(128,128,128), mode="train", transforms=None):
        self.data_dir = data_dir
        self.mode = mode
        self.patch_size = patch_size
        self.ids = self._collect_ids()
        self.transforms = transforms or self.default_transforms()

    def _collect_ids(self):
        # Expecting folders each containing *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz and *_seg.nii.gz
        files = os.listdir(self.data_dir)
        ids = []
        # If dataset is arranged as case folders
        for entry in files:
            p = os.path.join(self.data_dir, entry)
            if os.path.isdir(p):
                ids.append(p)
        # fallback: look for nii files
        if len(ids) == 0:
            # group by prefix
            prefixes = {}
            for f in files:
                if f.endswith('.nii.gz'):
                    prefix = f.split('.')[0].rsplit('_',1)[0]
                    prefixes.setdefault(prefix, []).append(os.path.join(self.data_dir, f))
            ids = [v for v in prefixes.values() if len(v) >= 4]
        return sorted(ids)

    def default_transforms(self):
        keys = ["image", "label"]
        tr = Compose([
            # LoadImaged will be handled in __getitem__ to allow custom normalization
            # Random spatial augmentations
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.3, max_k=3),
            RandAffined(keys=keys, prob=0.3, scale_range=(0.9,1.1), rotate_range=(0.1,0.1,0.1)),
            RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_center=True, random_size=False),
        ])
        return tr

    def __len__(self):
        return len(self.ids)

    def _load_case(self, case_path):
        # case_path may be folder or list of file paths
        if isinstance(case_path, list):
            files = case_path
        else:
            files = os.listdir(case_path)
            files = [os.path.join(case_path, f) for f in files]

        # find modalities
        # Supports both classic naming (t1,t1ce,t2,flair) and BraTS2023 naming (t1n,t1c,t2w,t2f).
        mod_map: dict[str, str | None] = {'t1':None,'t1ce':None,'t2':None,'flair':None,'seg':None}
        for f in files:
            fname = os.path.basename(f).lower()
            if 't1ce' in fname or '_t1ce' in fname or '-t1ce.' in fname or '-t1c.' in fname:
                mod_map['t1ce'] = f
            elif '_t1.' in fname or fname.endswith('t1.nii.gz') or '-t1n.' in fname:
                mod_map['t1'] = f
            elif '_t2' in fname or '-t2w.' in fname:
                mod_map['t2'] = f
            elif 'flair' in fname or '-t2f.' in fname:
                mod_map['flair'] = f
            elif 'seg' in fname or 'segmentation' in fname:
                mod_map['seg'] = f

        imgs = []
        for k in ['t1','t1ce','t2','flair']:
            if mod_map[k] is None:
                imgs.append(np.zeros(self.patch_size, dtype=np.float32))
            else:
                file_path: str | None = mod_map[k]
                if file_path is None:
                    imgs.append(np.zeros(self.patch_size, dtype=np.float32))
                    continue
                nii_img = cast(Any, nib).load(file_path)
                arr = nii_img.get_fdata().astype(np.float32)
                # standardize layout to CZYX -> take center crop later
                arr = np.transpose(arr, (2,1,0))
                arr = zscore_normalize(arr)
                imgs.append(arr)

        image = np.stack(imgs, axis=0)
        if mod_map['seg'] is not None:
            seg_path: str | None = mod_map['seg']
            if seg_path is None:
                seg = np.zeros_like(image[0], dtype=np.uint8)
            else:
                seg_nii = cast(Any, nib).load(seg_path)
                seg = seg_nii.get_fdata().astype(np.uint8)
            seg = np.transpose(seg, (2,1,0))
            seg = map_brats_labels(seg)
        else:
            seg = np.zeros_like(image[0], dtype=np.uint8)

        return image, seg

    def __getitem__(self, idx):
        case = self.ids[idx]
        image, seg = self._load_case(case)
        sample: Any = {"image": image, "label": np.expand_dims(seg, 0)}
        if self.transforms is not None:
            sample = self.transforms(sample)

        image_data: Any
        label_data: Any
        if isinstance(sample, dict):
            image_data = sample.get("image")
            label_data = sample.get("label")
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image_data, label_data = sample[0], sample[1]
        else:
            raise TypeError("Transforms must return a dict with 'image'/'label' or a 2-item tuple/list")

        if image_data is None or label_data is None:
            raise ValueError("Missing image/label in transformed sample")

        if isinstance(image_data, torch.Tensor):
            img = image_data.float()
        else:
            img = torch.from_numpy(np.asarray(image_data, dtype=np.float32))

        if isinstance(label_data, torch.Tensor):
            lbl = label_data.long()
        else:
            lbl = torch.from_numpy(np.asarray(label_data, dtype=np.int64))

        return img, lbl.squeeze(0).long()
