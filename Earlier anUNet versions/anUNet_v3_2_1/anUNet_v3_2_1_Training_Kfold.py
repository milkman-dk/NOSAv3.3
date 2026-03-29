# anUNet_v3.2.1: WT-priority training strategy.
# Goal: maximize Whole Tumor (WT) IoU.
# Key differences vs anUNet_v3_Training_optimized.py:
# - Single output head (WT only)
# - WT-only loss (Dice + Focal)
# - WT-focused sampling wrapper (bias toward patches containing tumor)
# - Validation threshold sweep to log best WT IoU and best threshold

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import json
import os
import random
import sys
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from monai.losses.dice import DiceLoss
from monai.losses.focal_loss import FocalLoss
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CenterSpatialCropd, RandSpatialCropd
from monai.transforms.intensity.dictionary import RandScaleIntensityd, RandShiftIntensityd
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, RandRotate90d

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import BraTSDataset
try:
    from .anUNet_v3_2_1_model import UNet3D
except ImportError:
    from anUNet_v3_2_1.anUNet_v3_2_1_model import UNet3D


# DATA / RUN PATHS
DATA_DIR = r"C:/"
MAPPING_XLSX = r"C:/"
MODEL_SAVE_DIR = os.path.abspath(os.path.dirname(__file__))

# OPTIMIZATION HYPERPARAMETERS
BATCH_SIZE = 2
ACCUMULATE_GRAD_BATCHES = 8
PATCH_SIZE = (128, 128, 128)
NUM_WORKERS = 10
LR = 4e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
MIN_LR = 1e-6
MAX_EPOCHS = 240
PRECISION = "16-mixed"
GRAD_CLIP = 1.0
SEED = 44

# WT-priority knobs
# Softer WT bias: fewer resampling retries and lower positive-voxel threshold.
WT_SAMPLE_ATTEMPTS = 3
MIN_WT_VOXELS = 12
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

# KFold settings
N_SPLITS = 5
FOLD_INDEX = 0  # which fold is used for val
CASE_ID_COL = "BraTS2023"
GROUP_ID_COL = "Local ID "


class SizedSampleDataset(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class WTLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, focal_weight: float = 0.3, gamma: float = 1.5):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.focal = FocalLoss(gamma=gamma, to_onehot_y=False)
        self.dice_w = dice_weight
        self.focal_w = focal_weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Binary WT supervision for the single segmentation head.
        y_wt = (y > 0).unsqueeze(1).float()
        return self.dice_w * self.dice(logits, y_wt) + self.focal_w * self.focal(logits, y_wt)


class WTPrioritizedDataset(Dataset):
    # Wrap BraTSDataset and bias training samples toward WT-positive patches.

    def __init__(self, base_dataset: SizedSampleDataset, attempts: int = 6, min_wt_voxels: int = 32):
        self.base_dataset = base_dataset
        self.attempts = attempts
        self.min_wt_voxels = min_wt_voxels
        self._len = len(base_dataset)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if int((label > 0).sum().item()) >= self.min_wt_voxels:
            return image, label

        for _ in range(self.attempts - 1):
            ridx = random.randrange(self._len)
            image_alt, label_alt = self.base_dataset[ridx]
            if int((label_alt > 0).sum().item()) >= self.min_wt_voxels:
                return image_alt, label_alt

        return image, label


def build_train_transforms(patch_size):
    # Augmentation recipe for training patches.
    keys = ["image", "label"]
    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=keys, prob=0.3, max_k=3),
        RandAffined(
            keys=keys,
            prob=0.3,
            rotate_range=(0.08, 0.08, 0.08),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.25),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.25),
        RandSpatialCropd(keys=keys, roi_size=patch_size, random_center=True, random_size=False),
    ])


def build_val_transforms(patch_size):
    # Validation preprocessing stays deterministic.
    return Compose([
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
    ])


def iou_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return tp / (tp + fp + fn + eps)


def dice_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (2.0 * tp) / (2.0 * tp + fp + fn + eps)


def recall_from_counts(tp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return tp / (tp + fn + eps)


class NOSA31Lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Main WT head + auxiliary deep-supervision heads from decoder stages.
        self.net = UNet3D(n_channels=4, n_classes=1, base_filters=32)
        self.loss_fn = WTLoss()
        self.aux2_head = nn.Conv3d(self.net.d2, 1, kernel_size=1)
        self.aux3_head = nn.Conv3d(self.net.d3, 1, kernel_size=1)
        self.aux2_weight = 0.4
        self.aux3_weight = 0.2
        self.thresholds: List[float] = list(THRESHOLDS)
        self._val_stats: Dict[float, Dict[str, torch.Tensor]] = {}
        self.save_hyperparameters(ignore=["loss_fn"])

    def forward(self, x):
        return self.net(x)

    def on_validation_epoch_start(self):
        # Reset threshold statistics at epoch start.
        self._val_stats = {
            thr: {
                "tp": torch.tensor(0.0, device=self.device),
                "fp": torch.tensor(0.0, device=self.device),
                "fn": torch.tensor(0.0, device=self.device),
            }
            for thr in self.thresholds
        }

    def training_step(self, batch, batch_idx):
        # Training pass with deep supervision losses.
        x, y = batch
        logits, y2, y3 = self.net(x, return_decoder_features=True)
        aux2_logits = F.interpolate(self.aux2_head(y2), size=logits.shape[2:], mode="trilinear", align_corners=False)
        aux3_logits = F.interpolate(self.aux3_head(y3), size=logits.shape[2:], mode="trilinear", align_corners=False)

        main_loss = self.loss_fn(logits, y)
        aux2_loss = self.loss_fn(aux2_logits, y)
        aux3_loss = self.loss_fn(aux3_logits, y)
        loss = main_loss + self.aux2_weight * aux2_loss + self.aux3_weight * aux3_loss

        self.log("train/loss_main", main_loss, on_step=True, on_epoch=True)
        self.log("train/loss_aux2", aux2_loss, on_step=True, on_epoch=True)
        self.log("train/loss_aux3", aux3_loss, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation pass with deep supervision and threshold sweep stats.
        x, y = batch
        logits, y2, y3 = self.net(x, return_decoder_features=True)
        aux2_logits = F.interpolate(self.aux2_head(y2), size=logits.shape[2:], mode="trilinear", align_corners=False)
        aux3_logits = F.interpolate(self.aux3_head(y3), size=logits.shape[2:], mode="trilinear", align_corners=False)
        main_loss = self.loss_fn(logits, y)
        aux2_loss = self.loss_fn(aux2_logits, y)
        aux3_loss = self.loss_fn(aux3_logits, y)
        loss = main_loss + self.aux2_weight * aux2_loss + self.aux3_weight * aux3_loss

        probs = torch.sigmoid(logits[:, 0])
        gt_wt = y > 0

        for thr in self.thresholds:
            pred = probs > thr
            tp = (pred & gt_wt).float().sum()
            fp = (pred & (~gt_wt)).float().sum()
            fn = ((~pred) & gt_wt).float().sum()
            self._val_stats[thr]["tp"] += tp
            self._val_stats[thr]["fp"] += fp
            self._val_stats[thr]["fn"] += fn

        self.log("val/loss_main", main_loss, on_epoch=True)
        self.log("val/loss", loss, on_epoch=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Finalize best WT threshold and summary metrics.
        best_thr = self.thresholds[0]
        best_iou = torch.tensor(-1.0, device=self.device)
        best_dice = torch.tensor(0.0, device=self.device)
        best_recall = torch.tensor(0.0, device=self.device)

        for thr in self.thresholds:
            tp = self._val_stats[thr]["tp"]
            fp = self._val_stats[thr]["fp"]
            fn = self._val_stats[thr]["fn"]
            iou = iou_from_counts(tp, fp, fn)
            dice = dice_from_counts(tp, fp, fn)
            recall = recall_from_counts(tp, fn)
            self.log(f"val/wt_iou@{thr:.2f}", iou, on_epoch=True)
            self.log(f"val/wt_recall@{thr:.2f}", recall, on_epoch=True)

            if iou > best_iou:
                best_iou = iou
                best_dice = dice
                best_recall = recall
                best_thr = thr

        tp = self._val_stats[0.50]["tp"]
        fp = self._val_stats[0.50]["fp"]
        fn = self._val_stats[0.50]["fn"]
        iou_050 = iou_from_counts(tp, fp, fn)
        dice_050 = dice_from_counts(tp, fp, fn)
        recall_050 = recall_from_counts(tp, fn)

        self.log("val/wt_iou", best_iou, prog_bar=True, on_epoch=True)
        self.log("val/wt_dice", best_dice, prog_bar=True, on_epoch=True)
        self.log("val/wt_recall", best_recall, prog_bar=True, on_epoch=True)
        self.log("val/wt_best_thr", torch.tensor(float(best_thr), device=self.device), on_epoch=True)
        self.log("val/wt_iou@0.50", iou_050, on_epoch=True)
        self.log("val/wt_dice@0.50", dice_050, on_epoch=True)
        self.log("val/wt_recall@0.50", recall_050, on_epoch=True)

    def configure_optimizers(self) -> Any:
        # Linear warmup followed by cosine annealing.
        optimizer = optim.AdamW(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, MAX_EPOCHS - WARMUP_EPOCHS),
            eta_min=MIN_LR,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[WARMUP_EPOCHS],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class SaveBestWTThreshold(Callback):
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = output_path
        self.best_iou = float("-inf")

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        iou_val = self._to_float(metrics.get("val/wt_iou"))
        thr_val = self._to_float(metrics.get("val/wt_best_thr"))
        if iou_val is None or thr_val is None:
            return

        if iou_val > self.best_iou:
            self.best_iou = iou_val
            payload = {
                "model": "anUNet_v3.2.1",
                "best_wt_iou": iou_val,
                "best_wt_threshold": thr_val,
                "epoch": int(trainer.current_epoch),
            }
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)


def _collect_case_dirs(data_dir: str) -> List[str]:
    return sorted(
        [
            name
            for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
        ]
    )


def build_first_fold_case_split(
    data_dir: str,
    mapping_xlsx: str,
    n_splits: int = 5,
    fold_index: int = 0,
    seed: int = 44,
) -> Tuple[List[str], List[str]]:
    # Fold generation uses group-aware split where mapping is available.
    if not os.path.isfile(mapping_xlsx):
        raise FileNotFoundError(f"Mapping file not found: {mapping_xlsx}")

    all_cases = _collect_case_dirs(data_dir)
    if len(all_cases) < n_splits:
        raise ValueError(f"Not enough cases ({len(all_cases)}) for {n_splits}-fold split.")
    if not (0 <= fold_index < n_splits):
        raise ValueError(f"fold_index must be in [0, {n_splits - 1}], got {fold_index}.")

    case_set = set(all_cases)
    df = pd.read_excel(mapping_xlsx)
    for col in (CASE_ID_COL, GROUP_ID_COL):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in mapping file.")

    df_cases = df[df[CASE_ID_COL].isin(case_set)].copy()
    if df_cases.empty:
        raise ValueError("No BraTS2023 IDs from mapping file match case folders in DATA_DIR.")

    mapped_df = df_cases[df_cases[GROUP_ID_COL].notna()].copy()
    unmapped_df = df_cases[df_cases[GROUP_ID_COL].isna()].copy()

    train_cases: List[str] = []
    val_cases: List[str] = []

    if not mapped_df.empty:
        mapped_groups = mapped_df[GROUP_ID_COL].astype(str).values
        gkf = GroupKFold(n_splits=n_splits)
        mapped_splits = list(gkf.split(mapped_df, groups=mapped_groups))
        tr_idx, va_idx = mapped_splits[fold_index]
        train_cases.extend(mapped_df.iloc[tr_idx][CASE_ID_COL].astype(str).tolist())
        val_cases.extend(mapped_df.iloc[va_idx][CASE_ID_COL].astype(str).tolist())

    if not unmapped_df.empty:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        unmapped_splits = list(kf.split(unmapped_df))
        tr_idx, va_idx = unmapped_splits[fold_index]
        train_cases.extend(unmapped_df.iloc[tr_idx][CASE_ID_COL].astype(str).tolist())
        val_cases.extend(unmapped_df.iloc[va_idx][CASE_ID_COL].astype(str).tolist())

    mapped_case_names = set(df_cases[CASE_ID_COL].astype(str).tolist())
    missing_from_mapping = sorted(case_set - mapped_case_names)
    if missing_from_mapping:
        # Keep unmatched cases for training so data is not silently dropped.
        train_cases.extend(missing_from_mapping)

    train_cases = sorted(set(train_cases))
    val_cases = sorted(set(val_cases))

    overlap = set(train_cases).intersection(val_cases)
    if overlap:
        raise RuntimeError(f"Fold split produced overlap between train/val sets: {len(overlap)} cases")

    covered = set(train_cases).union(val_cases)
    if covered != case_set:
        missing = sorted(case_set - covered)
        raise RuntimeError(f"Fold split does not cover all cases. Missing count: {len(missing)}")

    print(
        f"[FOLD] Using fold {fold_index}/{n_splits - 1} | "
        f"train={len(train_cases)} val={len(val_cases)} total={len(case_set)}"
    )
    print(
        f"[FOLD] mapped_with_group={len(mapped_df)} | "
        f"unmapped_no_group={len(unmapped_df)} | "
        f"missing_in_mapping={len(missing_from_mapping)}"
    )

    return train_cases, val_cases


def main():
    # Global setup.
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for anUNet_v3.2.1 WT-priority training.")

    enable_tf32()

    os.environ["WANDB_MODE"] = "online"

    # Generate train/val case IDs for current fold.
    train_case_ids, val_case_ids = build_first_fold_case_split(
        data_dir=DATA_DIR,
        mapping_xlsx=MAPPING_XLSX,
        n_splits=N_SPLITS,
        fold_index=FOLD_INDEX,
        seed=SEED,
    )

    # Build datasets and bind selected case lists.
    train_base = BraTSDataset(
        DATA_DIR,
        mode="train",
        patch_size=PATCH_SIZE,
        transforms=build_train_transforms(PATCH_SIZE),
    )
    train_base.ids = [os.path.join(DATA_DIR, case_id) for case_id in train_case_ids]
    train_ds = WTPrioritizedDataset(
        train_base,
        attempts=WT_SAMPLE_ATTEMPTS,
        min_wt_voxels=MIN_WT_VOXELS,
    )
    val_ds = BraTSDataset(
        DATA_DIR,
        mode="val",
        patch_size=PATCH_SIZE,
        transforms=build_val_transforms(PATCH_SIZE),
    )
    val_ds.ids = [os.path.join(DATA_DIR, case_id) for case_id in val_case_ids]

    # Construct dataloaders.
    use_persistent = NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    # Create model, logger, and callbacks.
    model = NOSA31Lightning()

    wandb_logger = WandbLogger(project="anUNet_v3.2.1", log_model=True, save_dir=MODEL_SAVE_DIR)
    best_thr_path = os.path.join(MODEL_SAVE_DIR, "anUNet_v3_2_1_best_threshold.json")
    best_thr_callback = SaveBestWTThreshold(best_thr_path)

    ckpt_callback = ModelCheckpoint(
        monitor="val/wt_iou",
        mode="max",
        save_top_k=3,
        dirpath=MODEL_SAVE_DIR,
        filename=f"nosa-v3_2_1-fold{FOLD_INDEX}" + "-{epoch:02d}-{val_wt_iou:.4f}",
        auto_insert_metric_name=False,
    )

    early_stop = EarlyStopping(
        monitor="val/wt_iou",
        mode="max",
        patience=40,
        min_delta=1e-4,
    )

    # Start trainer.
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision=PRECISION,
        deterministic=True,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=GRAD_CLIP,
        callbacks=[ckpt_callback, early_stop, best_thr_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
    )


if __name__ == "__main__":
    main()
