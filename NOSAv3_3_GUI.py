"""
NOSAv3.3 GUI

Threading:
- InferenceWorker: Runs model.predict() in background
- MetricsWorker: Computes Dice/IoU/Precision/Recall/HD95 progressively
- GUI never freezes (ideally); all long ops use QThreadPool
"""

import json
import importlib.util
import logging
import os
import sys
import ctypes
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import nibabel as nib
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch
from PyQt6.QtCore import (
    Qt,
    QEvent,
    QObject,
    pyqtSignal,
    QRunnable,
    QThreadPool,
    QTimer,
)
from PyQt6.QtGui import QFont, QIcon, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QPushButton,
    QLabel,
    QGroupBox,
    QStatusBar,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)

# VTK imports
from vtkmodules.vtkRenderingOpenGL2 import vtkGenericOpenGLRenderWindow
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
from vtkmodules.vtkRenderingVolume import (
    vtkVolumeMapper,
    vtkGPUVolumeRayCastMapper,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkVolume,
    vtkVolumeProperty,
    vtkColorTransferFunction,
    vtkRenderer,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Constants & Configuration

NOSA_MODEL_DIR = Path(__file__).parent #model_directory
DEFAULT_APP_ICON = NOSA_MODEL_DIR.parent / "various" / "NOSALogo.jpeg" #app_icon
WINDOWS_APP_ID = "NOSA.v3_3.GUI" #windows_identifier
DEFAULT_CHECKPOINT = NOSA_MODEL_DIR / "anunet-v3_3-fold1-230-0.0000.ckpt" #checkpoint_path
DEFAULT_THRESHOLD_JSON = NOSA_MODEL_DIR / "anUNet_v3_3_fold1_best_threshold.json" #threshold_json_path
BRATS_DATA_ROOT = Path(os.environ.get("BRATS_DATA_ROOT", str(NOSA_MODEL_DIR.parent / "BraTS"))) #brats_data_root
BRATS_TRAINING_DIR = Path( #brats_source_dir
    os.environ.get(
        "BRATS_TRAINING_DIR",
        str(BRATS_DATA_ROOT / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
    )
)
DEFAULT_LOAD_IMAGE_DIR = Path( #default_load_dir
    os.environ.get("BRATS_VALIDATION_DIR", str(BRATS_DATA_ROOT / "Fold3ValidationData"))
)
PREDICTION_CACHE_DIR = NOSA_MODEL_DIR / ".tmp_predictions" #temp_cache_dir

BRAIN_OPACITY = 0.85 #brain_transparency
TUMOR_OPACITY = 0.8 #tumor_transparency
TUMOR_THRESHOLD = 0.5 #prediction_threshold
TUMOR_VOXEL_THRESHOLD = 0.01  # 1% of volume #voxel_count_min

# Automatic post-processing knobs (crater sealing + enclosed-hole filling).
POSTPROCESS_MIN_COMPONENT_VOXELS = 64 #min_component_size
POSTPROCESS_MAX_CLOSING_ITERS = 7 #max_morphology_iterations
POSTPROCESS_ENVELOPE_DILATION_ITERS = 6 #envelope_dilation_iters
POSTPROCESS_MAX_ADDED_RATIO = 40.0 #max_growth_ratio
POSTPROCESS_MAX_OUTSIDE_ADDED_RATIO = 0.20 #max_outside_growth_ratio

# VTK Colors (RGB normalized to 0-1)
COLOR_BRAIN = (0.6, 0.6, 0.6)  # Gray #color_brain
COLOR_TUMOR = (1.0, 0.35, 0.35)  # Light red #color_tumor_gt
COLOR_PRED_TUMOR = (0.0, 0.831, 1.0)  # Cyan/bright blue matching UI boundaries and buttons (#00d4ff) #color_pred_tumor
COLOR_TP = (0.1, 0.95, 0.2)  # Green (true positives) #color_tp
COLOR_FP = (1.0, 0.92, 0.2)  # Yellow (false positives) #color_fp
COLOR_FN = (1.0, 0.2, 0.2)   # Red (false negatives) #color_fn
COLOR_BACKGROUND = (0.2, 0.2, 0.2)  # Dark gray #color_background

PREDICTION_SMOOTHING_SIGMA = 1.0 #gaussian_smoothing_sigma

DIAGNOSIS_COLOR_NEUTRAL = "#edf3f8" #neutral_color_hex
DIAGNOSIS_COLOR_POSITIVE = "#ff6b6b" #positive_color_hex
DIAGNOSIS_COLOR_NEGATIVE = "#6bff6b" #negative_color_hex

# Rendering backend policy: default to safe mapper to avoid driver-specific crashes.
USE_GPU_MAPPER_BY_DEFAULT = False #gpu_mapper_disabled


# AppState: Dataclass to hold all state


@dataclass
class AppState:
    """Thread-safe application state container."""

    brain_nii: Optional[object] = None  # nibabel Nifti1Image #nifti_header
    brain_data: Optional[np.ndarray] = None  # (D, H, W) float32 #brain_mri_volume
    tumor_nii: Optional[object] = None #tumor_nifti_header
    raw_tumor_data: Optional[np.ndarray] = None  # Raw binary prediction (D, H, W) uint8 #raw_prediction
    postprocessed_tumor_data: Optional[np.ndarray] = None  # Postprocessed prediction (D, H, W) uint8 #filtered_prediction
    tumor_data: Optional[np.ndarray] = None  # Binary (D, H, W) uint8 #binary_tumor
    ground_truth: Optional[np.ndarray] = None  # Binary (D, H, W) uint8 #gt_mask
    case_id: Optional[str] = None #case_identifier
    case_dir: Optional[str] = None #case_folder_path
    voxel_volume_mm3: Optional[float] = None #voxel_spacing_volume

    slice_pos: Tuple[int, int, int] = (64, 64, 64) #current_slice_position
    zoom_level: float = 1.0 #viewport_zoom

    metrics: Dict[str, float] = field(default_factory=dict) #computed_metrics
    is_loading: bool = False #flag_loading
    is_inferring: bool = False #flag_inferring
    is_postprocessing: bool = False #flag_postprocessing

    tumor_voxel_count: int = 0 #tumor_volume_voxels
    confidence: float = 0.0 #mean_prediction_confidence
    postprocessing_enabled: bool = False #postprocess_active
    gt_comparison_visible: bool = False #gt_overlay_active

    def get_bounds(self) -> Optional[Tuple[int, int, int]]:
        """Return (D, H, W) bounds if brain loaded."""
        if self.brain_data is None:
            return None
        s = self.brain_data.shape
        if len(s) != 3:
            return None
        return (int(s[0]), int(s[1]), int(s[2]))


# ModelLoader: Load NOSA v3.3 checkpoint


class ModelLoader:
    """Load NOSA v3.3 model checkpoint (pattern from NOSA_v3_3_eval_BraTS.py)."""

    def __init__(self, checkpoint_path: Path, threshold_json: Path):
        self.checkpoint_path = checkpoint_path
        self.threshold_json = threshold_json
        self.model = None
        self.device = None
        self.threshold = TUMOR_THRESHOLD

    def _choose_device(self) -> torch.device:
        """Select CUDA when supported by the current PyTorch build, else CPU."""
        # Avoid noisy startup warning for unsupported new GPUs (e.g., sm_120).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*CUDA capability .* is not compatible with the current PyTorch installation.*",
                category=UserWarning,
            )
            cuda_available = torch.cuda.is_available()

        if not cuda_available:
            logger.warning("CUDA not available or unsupported by current PyTorch build; using CPU")
            return torch.device("cpu")

        # Validate that this wheel includes kernels for the active GPU architecture.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*CUDA capability .* is not compatible with the current PyTorch installation.*",
                    category=UserWarning,
                )
                major, minor = torch.cuda.get_device_capability(0)
            target_arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())

            if target_arch not in supported_arches:
                logger.warning(
                    "GPU architecture %s not supported by this PyTorch install (%s); using CPU",
                    target_arch,
                    ", ".join(sorted(supported_arches)) if supported_arches else "unknown",
                )
                return torch.device("cpu")
        except Exception as e:
            logger.warning("Could not validate CUDA architecture (%s); using CPU", e)
            return torch.device("cpu")

        return torch.device("cuda:0")

    def load(self) -> Tuple[object, float]:
        """Load model and threshold from checkpoint.
        
        Returns:
            (model, threshold) — both ready for inference
        """
        self.device = self._choose_device()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        logger.info(f"Loading checkpoint: {self.checkpoint_path}")

        # Import model class
        sys.path.insert(0, str(NOSA_MODEL_DIR))
        from anUNet_v3_3_model import UNet3D

        self.model = UNet3D(n_channels=4, n_classes=1, base_filters=32)
        model_keys = set(self.model.state_dict().keys())

        # Load checkpoint with state_dict cleanup (from NOSA_v3_3_eval_BraTS.py)
        raw = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=False
        )
        state = raw.get("state_dict", raw)

        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint does not contain valid state_dict.")

        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("net.", "model.", "module."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break
            if new_key in model_keys:
                cleaned[new_key] = value

        missing = sorted(model_keys - set(cleaned.keys()))
        if missing:
            raise RuntimeError(f"Missing model keys: {missing[:5]}")

        self.model.load_state_dict(cleaned, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Load threshold from JSON
        if self.threshold_json.exists():
            try:
                with open(self.threshold_json, "r") as f:
                    payload = json.load(f)
                    self.threshold = float(payload.get("best_wt_threshold", 0.5))
                    logger.info(f"Threshold loaded: {self.threshold:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load threshold: {e}; using default 0.5")
                self.threshold = 0.5

        logger.info(
            f"Model loaded on device: {self.device}; threshold: {self.threshold:.4f}"
        )
        return self.model, self.threshold


# ImageIOManager: Load .nii.gz files and GT discovery


class ImageIOManager:
    """Handle loading .nii.gz files and auto-discovery of ground truth."""

    @staticmethod
    def load_nifti(path: str) -> Tuple[np.ndarray, object]:
        """Load .nii.gz file.
        
        Returns:
            (data, nii_image) — data is float32, nii_image is nibabel Nifti1Image
        """
        img: Any = nib_load(path)
        data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
        logger.info(f"Loaded: {path}, shape: {data.shape}")
        return data, img

    @staticmethod
    def load_case_modalities(case_path: str) -> Tuple[np.ndarray, Optional[str]]:
        """Load 4-channel brain MRI (T1, T1ce, T2, FLAIR) from BraTS case folder.
        
        Follows pattern from NOSA_v3_3_eval_BraTS.py build_model_input()
        
        Returns:
            (image_4ch, case_id) — image_4ch is (4, D, H, W) float32
        """
        modality_names = ["t1", "t1ce", "t2", "flair"]
        modality_tokens = {
            "t1": {"t1", "t1n"},
            "t1ce": {"t1ce", "t1c"},
            "t2": {"t2", "t2w"},
            "flair": {"flair", "t2f"},
        }

        def _strip_nii_extensions(file_name: str) -> Optional[str]:
            name = file_name.lower()
            if name.endswith(".nii.gz"):
                return name[:-7]
            if name.endswith(".nii"):
                return name[:-4]
            return None

        def _extract_modality_token(file_name: str) -> Optional[str]:
            stem = _strip_nii_extensions(file_name)
            if stem is None:
                return None
            # Handles patterns like BraTS-...-t1n and BraTS_..._t1ce.
            tail_dash = stem.split("-")[-1]
            tail_underscore = tail_dash.split("_")[-1]
            return tail_underscore

        if os.path.isfile(case_path):
            case_dir = os.path.dirname(case_path)
        else:
            case_dir = case_path

        case_id = os.path.basename(case_dir)
        files = [f for f in os.listdir(case_dir)]

        volumes = {}
        ref_shape = None

        for name in modality_names:
            found = None
            expected_tokens = modality_tokens[name]

            for fname_real in files:
                token = _extract_modality_token(fname_real)
                if token in expected_tokens:
                    found = os.path.join(case_dir, fname_real)
                    break

            if found:
                data, _ = ImageIOManager.load_nifti(found)
                volumes[name] = data
                if ref_shape is None:
                    ref_shape = data.shape
            else:
                logger.warning(f"Missing modality {name} in {case_dir}")
                if ref_shape is None:
                    continue
                volumes[name] = np.zeros(ref_shape, dtype=np.float32)

        if ref_shape is None:
            raise RuntimeError(f"No valid MRI modalities found in: {case_dir}")

        for name in modality_names:
            if name not in volumes:
                volumes[name] = np.zeros(ref_shape, dtype=np.float32)

        # Stack into 4-channel (4, D, H, W)
        image_4ch = np.stack(
            [volumes.get(name, np.zeros_like(next(iter(volumes.values()))))
             for name in modality_names],
            axis=0,
        ).astype(np.float32)

        logger.info(f"Loaded 4-channel image for {case_id}: {image_4ch.shape}")
        return image_4ch, case_id

    @staticmethod
    def get_case_voxel_volume_mm3(case_path: str) -> Optional[float]:
        """Return voxel volume (mm^3) from one modality header in a case folder."""
        modality_tokens = {
            "t1", "t1n", "t1ce", "t1c", "t2", "t2w", "flair", "t2f"
        }

        def _strip_nii_extensions(file_name: str) -> Optional[str]:
            name = file_name.lower()
            if name.endswith(".nii.gz"):
                return name[:-7]
            if name.endswith(".nii"):
                return name[:-4]
            return None

        def _extract_modality_token(file_name: str) -> Optional[str]:
            stem = _strip_nii_extensions(file_name)
            if stem is None:
                return None
            tail_dash = stem.split("-")[-1]
            tail_underscore = tail_dash.split("_")[-1]
            return tail_underscore

        if os.path.isfile(case_path):
            case_dir = os.path.dirname(case_path)
        else:
            case_dir = case_path

        try:
            for file_name in os.listdir(case_dir):
                token = _extract_modality_token(file_name)
                if token not in modality_tokens:
                    continue

                image_path = os.path.join(case_dir, file_name)
                nii: Any = nib_load(image_path)
                zooms = nii.header.get_zooms()[:3]
                if len(zooms) != 3:
                    continue

                voxel_volume = float(zooms[0] * zooms[1] * zooms[2])
                logger.info(
                    "Voxel spacing for %s: %.3f x %.3f x %.3f mm (%.3f mm^3/voxel)",
                    os.path.basename(case_dir),
                    float(zooms[0]),
                    float(zooms[1]),
                    float(zooms[2]),
                    voxel_volume,
                )
                return voxel_volume
        except Exception as e:
            logger.warning("Could not read voxel spacing for %s (%s)", case_dir, e)

        return None

    @staticmethod
    def load_case_modalities_aligned(case_path: str) -> Tuple[np.ndarray, Optional[str]]:
        """Load modalities for visualization with orientation aligned to model output.

        Raw NIfTI arrays are typically XYZ; inference path uses ZYX. This helper
        keeps raw intensities but transposes to (C, Z, Y, X) so brain and tumor
        overlays share the same coordinate frame.
        """
        image_4ch, case_id = ImageIOManager.load_case_modalities(case_path)
        image_4ch = np.transpose(image_4ch, (0, 3, 2, 1)).astype(np.float32)
        logger.info(
            f"Loaded orientation-aligned 4-channel visualization input for {case_id}: {image_4ch.shape}"
        )
        return image_4ch, case_id

    @staticmethod
    def load_case_for_inference(case_path: str) -> Tuple[np.ndarray, Optional[str]]:
        """Load model input exactly like evaluation pipeline.

        Uses `BraTSDataset._load_case` from `SJF_NOSAv3/dataset.py`, which applies:
        - BraTS naming resolution
        - axis transpose to model layout (C, Z, Y, X)
        - per-modality z-score normalization
        """
        if os.path.isfile(case_path):
            case_dir = os.path.dirname(case_path)
        else:
            case_dir = case_path

        case_id = os.path.basename(case_dir)

        try:
            # dataset.py lives at repository root in this workspace.
            repo_root = str(NOSA_MODEL_DIR.parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)

            from dataset import BraTSDataset  # type: ignore

            ds = BraTSDataset(case_dir, mode="val")
            image_4ch, _ = ds._load_case(case_dir)
            image_4ch = np.asarray(image_4ch, dtype=np.float32)
            logger.info(f"Loaded preprocessed 4-channel inference input for {case_id}: {image_4ch.shape}")
            return image_4ch, case_id
        except Exception as e:
            logger.warning(
                "Failed to load evaluation-style preprocessing (%s). Falling back to raw modality loader.",
                e,
            )
            image_4ch, fallback_case_id = ImageIOManager.load_case_modalities(case_dir)
            image_4ch = np.transpose(image_4ch, (0, 3, 2, 1)).astype(np.float32)
            return image_4ch, fallback_case_id

    @staticmethod
    def find_ground_truth(case_id: str) -> Optional[np.ndarray]:
        """Auto-discover ground truth from BraTS training directory.
        
        Returns:
            Binary mask (D, H, W) uint8, or None if not found
        """
        if not BRATS_TRAINING_DIR.exists():
            logger.warning(f"BraTS training dir not found: {BRATS_TRAINING_DIR}")
            return None

        # Search for matching case folder
        for folder in BRATS_TRAINING_DIR.iterdir():
            if folder.is_dir() and case_id.lower() in folder.name.lower():
                seg_file = list(folder.glob("*seg.nii.gz"))
                if seg_file:
                    seg_data, _ = ImageIOManager.load_nifti(str(seg_file[0]))
                    # Create binary mask: whole tumor = any nonzero label
                    binary_mask = np.transpose((seg_data > 0).astype(np.uint8), (2, 1, 0))
                    logger.info(f"Found GT for {case_id}: {seg_file[0]}")
                    return binary_mask

        logger.warning(f"No GT found for case: {case_id}")
        return None


# SignalEmitter: Custom signals for thread-safe GUI updates


class SignalEmitter(QObject):
    """Thread-safe signal emitter for background workers."""

    inference_done = pyqtSignal(np.ndarray, float, int, float)  # tumor_data, confidence, voxel_count, inference_time_s
    inference_error = pyqtSignal(str)
    postprocess_done = pyqtSignal(np.ndarray)  # postprocessed_tumor_data
    postprocess_error = pyqtSignal(str)

    metric_ready = pyqtSignal(str, float)  # metric_name, value
    metrics_done = pyqtSignal(bool)  # is_postprocessed

    status_update = pyqtSignal(str)  # status message


# Workers: Inference & Metrics computation (QRunnable)


class InferenceWorker(QRunnable):
    """Run NOSA v3.3 inference in background thread."""

    def __init__(self, model, device, threshold, case_dir, state, signals):
        super().__init__()
        self.model = model
        self.device = device
        self.threshold = threshold
        self.case_dir = case_dir
        self.state = state
        self.signals = signals

    def run(self):
        """Inference worker main loop (pattern from NOSA_v3_3_eval_BraTS.py)."""
        try:
            self.signals.status_update.emit("Running inference...")
            logger.info("Inference started")

            # Load full 4-channel input in worker thread to keep UI responsive.
            image_4ch, _ = ImageIOManager.load_case_for_inference(self.case_dir)

            # Pad to multiple of 16 (from NOSA_v3_3_eval_BraTS.py)
            _, depth, height, width = image_4ch.shape

            def _pad_size(size, multiple=16):
                extra = (-size) % multiple
                before = extra // 2
                return before, extra - before

            pd0, pd1 = _pad_size(depth)
            ph0, ph1 = _pad_size(height)
            pw0, pw1 = _pad_size(width)

            padded = np.pad(
                image_4ch,
                ((0, 0), (pd0, pd1), (ph0, ph1), (pw0, pw1)),
                mode="constant",
            )
            crop = (slice(pd0, pd0 + depth), slice(ph0, ph0 + height), slice(pw0, pw0 + width))

            def _predict_on_device(device: torch.device) -> np.ndarray:
                input_tensor = torch.as_tensor(padded, dtype=torch.float32, device=device)
                input_tensor = input_tensor.unsqueeze(0)

                with torch.inference_mode():
                    logits = self.model(input_tensor)
                    probability = torch.sigmoid(logits[0, 0])
                    prediction = (probability > self.threshold).to(torch.uint8)

                dsl, hsl, wsl = crop
                return prediction.cpu().numpy()[dsl, hsl, wsl].astype(np.uint8)

            # Run inference with automatic CUDA -> CPU fallback for unsupported GPUs.
            t0 = time.perf_counter()
            try:
                pred_zyx = _predict_on_device(self.device)
            except RuntimeError as e:
                err = str(e).lower()
                cuda_kernel_issue = (
                    "no kernel image is available for execution on the device" in err
                    or "cuda error" in err
                )
                if self.device.type == "cuda" and cuda_kernel_issue:
                    logger.warning(
                        "CUDA inference failed due to unsupported kernel image; "
                        "falling back to CPU inference."
                    )
                    self.signals.status_update.emit(
                        "CUDA unsupported by this PyTorch build. Falling back to CPU inference..."
                    )
                    self.model.to(torch.device("cpu"))
                    self.model.eval()
                    self.device = torch.device("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pred_zyx = _predict_on_device(self.device)
                else:
                    raise
            inference_time_s = time.perf_counter() - t0

            # Compute confidence and voxel count
            tumor_voxel_count = int(pred_zyx.sum())
            total_voxels = pred_zyx.size
            confidence = tumor_voxel_count / max(1, total_voxels)

            logger.info(
                f"Inference done: {tumor_voxel_count} tumor voxels, "
                f"confidence: {confidence:.4f}, time: {inference_time_s:.3f}s"
            )

            self.signals.inference_done.emit(pred_zyx, confidence, tumor_voxel_count, float(inference_time_s))

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            self.signals.inference_error.emit(str(e))


class MetricsWorker(QRunnable):
    """Compute Dice/IoU/Precision/Recall/HD95 in background."""

    def __init__(self, tumor_data, ground_truth, state, signals, is_postprocessed: bool = False):
        super().__init__()
        self.tumor_data = tumor_data
        self.ground_truth = ground_truth
        self.state = state
        self.signals = signals
        self.is_postprocessed = is_postprocessed

    @staticmethod
    def _fallback_dice(pred: np.ndarray, gt: np.ndarray) -> float:
        pred_bin = pred.astype(bool)
        gt_bin = gt.astype(bool)
        inter = np.logical_and(pred_bin, gt_bin).sum()
        denom = pred_bin.sum() + gt_bin.sum()
        return float((2.0 * inter) / max(1, denom))

    @staticmethod
    def _fallback_hd95(_: np.ndarray, __: np.ndarray) -> float:
        # Fallback when utils.py cannot be imported in the current environment.
        return float("nan")

    @staticmethod
    def _load_metric_functions() -> Tuple[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray, np.ndarray], float]]:
        utils_path = NOSA_MODEL_DIR.parent / "SJF_NOSAv3" / "utils.py"
        if not utils_path.exists():
            return MetricsWorker._fallback_dice, MetricsWorker._fallback_hd95

        try:
            spec = importlib.util.spec_from_file_location("sjf_utils", str(utils_path))
            if spec is None or spec.loader is None:
                return MetricsWorker._fallback_dice, MetricsWorker._fallback_hd95

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            dice_fn = getattr(module, "dice_coefficient", MetricsWorker._fallback_dice)
            hd95_fn = getattr(module, "hd95", MetricsWorker._fallback_hd95)
            return dice_fn, hd95_fn
        except Exception:
            return MetricsWorker._fallback_dice, MetricsWorker._fallback_hd95

    def run(self):
        """Metrics computation (uses utils.py functions)."""
        try:
            dice_coefficient, hd95 = self._load_metric_functions()

            if self.ground_truth is None:
                logger.warning("No ground truth; skipping metrics")
                self.signals.metrics_done.emit(self.is_postprocessed)
                return

            # Compute metrics sequentially
            logger.info("Computing metrics...")

            # Dice
            self.signals.status_update.emit("Computing Dice...")
            dice = float(dice_coefficient(self.tumor_data, self.ground_truth))
            self.signals.metric_ready.emit("Dice", dice)

            # IoU
            self.signals.status_update.emit("Computing IoU...")
            intersection = (self.tumor_data & self.ground_truth).sum()
            union = (self.tumor_data | self.ground_truth).sum()
            iou = float(intersection / max(1, union))
            self.signals.metric_ready.emit("IoU", iou)

            # Precision
            self.signals.status_update.emit("Computing Precision...")
            tp = (self.tumor_data & self.ground_truth).sum()
            fp = (self.tumor_data & ~self.ground_truth).sum()
            precision = float(tp / max(1, tp + fp))
            self.signals.metric_ready.emit("Precision", precision)

            # Recall
            self.signals.status_update.emit("Computing Recall...")
            fn = (~self.tumor_data & self.ground_truth).sum()
            recall = float(tp / max(1, tp + fn))
            self.signals.metric_ready.emit("Recall", recall)

            # HD95 (slowest)
            self.signals.status_update.emit("Computing Hausdorff Distance...")
            hd95_val = float(hd95(self.tumor_data.astype(bool), 
                                   self.ground_truth.astype(bool)))
            self.signals.metric_ready.emit("Hausdorff95", hd95_val)

            logger.info(f"Metrics computation done (postprocessed={self.is_postprocessed})")
            self.signals.metrics_done.emit(self.is_postprocessed)

        except Exception as e:
            logger.exception(f"Metrics error: {e}")
            self.signals.status_update.emit(f"Metrics error: {e}")


class PostprocessWorker(QRunnable):
    """Compute postprocessed mask in background thread to avoid GUI stalls."""

    def __init__(self, raw_tumor_data: np.ndarray, signals: SignalEmitter):
        super().__init__()
        self.raw_tumor_data = raw_tumor_data
        self.signals = signals

    def run(self):
        try:
            self.signals.status_update.emit("Preparing post-processing mask...")
            postprocessed = MainWindow._fill_prediction_holes(self.raw_tumor_data)
            self.signals.postprocess_done.emit(postprocessed)
        except Exception as e:
            logger.exception("Post-processing error: %s", e)
            self.signals.postprocess_error.emit(str(e))


# VTKVolumeRenderer: GPU-based 3D volume rendering


class VTKVolumeRenderer:
    """Render brain + tumor volumes using VTK GPU raycasting.
    
    CRITICAL DESIGN:
    - Brain volume always visible (grayscale, 30% opacity)
    - Tumor overlay on top (red, 80% opacity)
    - Both use vtkGPUVolumeRayCastMapper for real-time performance
    """

    def __init__(self):
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(*COLOR_BACKGROUND)

        self.brain_volume = None
        self.tumor_volume = None
        self.brain_mapper = None
        self.tumor_mapper = None
        self.volume_center = (0.0, 0.0, 0.0)
        self.volume_dims = (1.0, 1.0, 1.0)
        self.tumor_centroid = None
        self.default_camera_state = None
        self.comparison_volume = None

        self.interactor_style = vtkInteractorStyleTrackballCamera()

    def set_render_window(self, render_window):
        """Attach to VTK render window."""
        self.renderer.ResetCamera()
        render_window.AddRenderer(self.renderer)

    def clear_scene(self):
        """Remove all currently displayed volumes to avoid stale overlays."""
        if self.comparison_volume is not None:
            self.renderer.RemoveViewProp(self.comparison_volume)
            self.comparison_volume = None
        if self.tumor_volume is not None:
            self.renderer.RemoveViewProp(self.tumor_volume)
            self.tumor_volume = None
        if self.brain_volume is not None:
            self.renderer.RemoveViewProp(self.brain_volume)
            self.brain_volume = None

    def _build_volume_mapper(self, mapper_name: str) -> vtkVolumeMapper:
        """Create a working volume mapper on this machine.

        Prefers GPU mapper, but falls back to SmartVolumeMapper when the
        current VTK build has no concrete GPU override.
        """
        use_gpu = os.environ.get("NOSA_USE_GPU_MAPPER", "0") == "1"
        if not use_gpu and not USE_GPU_MAPPER_BY_DEFAULT:
            mapper = vtkSmartVolumeMapper()
            mapper.SetRequestedRenderModeToDefault()
            logger.info(f"Using vtkSmartVolumeMapper (safe default) for {mapper_name}")
            return mapper

        try:
            mapper = vtkGPUVolumeRayCastMapper()
            logger.info(f"Using vtkGPUVolumeRayCastMapper for {mapper_name}")
            return mapper
        except Exception as e:
            logger.warning(
                "GPU volume mapper unavailable for %s (%s). Falling back to vtkSmartVolumeMapper.",
                mapper_name,
                e,
            )

        mapper = vtkSmartVolumeMapper()
        mapper.SetRequestedRenderModeToDefault()
        logger.info(f"Using vtkSmartVolumeMapper fallback for {mapper_name}")
        return mapper

    def _apply_default_view(self):
        """Apply neutral default camera framing with no custom rotation."""
        camera = self.renderer.GetActiveCamera()
        camera.Roll(90.0)
        camera.Dolly(2.0)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()

    def _capture_default_camera_state(self):
        """Persist current camera pose so Reset View is perfectly repeatable."""
        camera = self.renderer.GetActiveCamera()
        self.default_camera_state = {
            "position": tuple(camera.GetPosition()),
            "focal_point": tuple(camera.GetFocalPoint()),
            "view_up": tuple(camera.GetViewUp()),
            "parallel_scale": float(camera.GetParallelScale()),
            "view_angle": float(camera.GetViewAngle()),
            "clipping_range": tuple(camera.GetClippingRange()),
        }

    def _restore_default_camera_state(self) -> bool:
        """Restore saved default camera pose. Returns False if no state exists."""
        if not self.default_camera_state:
            return False

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*self.default_camera_state["position"])
        camera.SetFocalPoint(*self.default_camera_state["focal_point"])
        camera.SetViewUp(*self.default_camera_state["view_up"])
        camera.SetParallelScale(self.default_camera_state["parallel_scale"])
        camera.SetViewAngle(self.default_camera_state["view_angle"])
        camera.SetClippingRange(*self.default_camera_state["clipping_range"])
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        return True

    def load_brain_volume(self, brain_data: np.ndarray):
        """Load brain MRI volume (always visible as context).
        
        Args:
            brain_data: (D, H, W) float32 numpy array
        """
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.util import numpy_support

        from scipy.ndimage import gaussian_filter

        logger.info(f"Loading brain volume: {brain_data.shape}")

        # Ensure previous case volumes are fully removed before adding new data.
        self.clear_scene()

        # Smooth the volume to remove blocky voxel-boundary artifacts
        brain_data = gaussian_filter(brain_data.astype(np.float32), sigma=1.0)

        # Create VTK image data
        vtkdata = vtkImageData()
        depth, height, width = brain_data.shape
        vtkdata.SetDimensions(width, height, depth)
        self.volume_center = (width / 2.0, height / 2.0, depth / 2.0)
        self.volume_dims = (float(width), float(height), float(depth))
        self.tumor_centroid = None

        # Convert numpy to VTK array
        vtk_array = numpy_support.numpy_to_vtk(
            num_array=brain_data.ravel(), deep=True, array_type=None
        )
        vtk_array.SetNumberOfComponents(1)
        vtkdata.GetPointData().SetScalars(vtk_array)

        # Create mapper
        self.brain_mapper = self._build_volume_mapper("brain")
        self.brain_mapper.SetInputData(vtkdata)

        # Create volume property (grayscale, 30% opacity)
        prop = vtkVolumeProperty()
        prop.ShadeOn()
        prop.IndependentComponentsOn()

        # Grayscale transfer function
        ctf = vtkColorTransferFunction()
        ctf.AddRGBPoint(0, 0, 0, 0)
        ctf.AddRGBPoint(128, 0.5, 0.5, 0.5)
        ctf.AddRGBPoint(255, 1, 1, 1)

        # Opacity function — skip air/background, then quickly ramp to near-opaque
        otf = vtkPiecewiseFunction()
        otf.AddPoint(0,   0.0)
        otf.AddPoint(25,  0.0)            # skip dark background / air
        otf.AddPoint(60,  0.55)           # tissue starts: jump to visible
        otf.AddPoint(120, BRAIN_OPACITY)  # brain parenchyma: nearly opaque
        otf.AddPoint(255, BRAIN_OPACITY)

        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)

        # Create volume
        self.brain_volume = vtkVolume()
        self.brain_volume.SetMapper(self.brain_mapper)
        self.brain_volume.SetProperty(prop)

        self.renderer.AddViewProp(self.brain_volume)
        self.renderer.ResetCamera()
        self._apply_default_view()
        self._capture_default_camera_state()

        logger.info("Brain volume added to renderer")

    def add_tumor_volume(self, tumor_data: np.ndarray):
        """Add tumor overlay on top of brain (red, 80% opacity).
        
        Args:
            tumor_data: Binary (D, H, W) uint8 numpy array
        """
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.util import numpy_support
        from scipy.ndimage import gaussian_filter

        logger.info(f"Loading tumor volume: {tumor_data.shape}")

        self._remove_comparison_volumes()
        if self.tumor_volume is not None:
            self.renderer.RemoveViewProp(self.tumor_volume)
            self.tumor_volume = None

        # Smooth slightly to reduce voxel stair-stepping while preserving sharper borders.
        smooth_data = gaussian_filter(
            tumor_data.astype(np.float32),
            sigma=PREDICTION_SMOOTHING_SIGMA,
        )
        smooth_data = np.clip(smooth_data, 0.0, 1.0)

        # Create VTK image data
        vtkdata = vtkImageData()
        depth, height, width = smooth_data.shape
        vtkdata.SetDimensions(width, height, depth)

        # Convert float32 smooth array to VTK
        vtk_array = numpy_support.numpy_to_vtk(
            num_array=smooth_data.ravel(), deep=True, array_type=None
        )
        vtk_array.SetNumberOfComponents(1)
        vtkdata.GetPointData().SetScalars(vtk_array)

        # Create mapper
        self.tumor_mapper = self._build_volume_mapper("tumor")
        self.tumor_mapper.SetInputData(vtkdata)

        # Create volume property (red, 80% opacity)
        prop = vtkVolumeProperty()
        prop.ShadeOn()
        prop.IndependentComponentsOn()

        # Red transfer function — color starts at 0.15 so faint edges are tinted
        ctf = vtkColorTransferFunction()
        ctf.AddRGBPoint(0.00, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(0.15, *COLOR_PRED_TUMOR)
        ctf.AddRGBPoint(1.00, *COLOR_PRED_TUMOR)

        # Opacity: smooth ramp from transparent edge → full opacity interior
        otf = vtkPiecewiseFunction()
        otf.AddPoint(0.00, 0.0)
        otf.AddPoint(0.10, 0.0)           # ignore noise below 10%
        otf.AddPoint(0.40, TUMOR_OPACITY)  # ramp to full by 40%
        otf.AddPoint(1.00, TUMOR_OPACITY)

        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)

        # Create volume
        self.tumor_volume = vtkVolume()
        self.tumor_volume.SetMapper(self.tumor_mapper)
        self.tumor_volume.SetProperty(prop)

        self.renderer.AddViewProp(self.tumor_volume)
        self.renderer.ResetCameraClippingRange()

        logger.info("Tumor volume added to renderer (overlaid on brain)")

    def _remove_comparison_volumes(self):
        """Remove combined TP/FP/FN comparison overlay from scene."""
        if self.comparison_volume is not None:
            self.renderer.RemoveViewProp(self.comparison_volume)
            self.comparison_volume = None

    def show_gt_comparison(
        self,
        prediction: Optional[np.ndarray],
        ground_truth: np.ndarray,
        tp_opacity: float = 0.35,
        fp_opacity: float = 0.35,
        fn_opacity: float = 0.35,
    ):
        """Visualize TP/FP/FN in one labeled volume with per-class colors/opacities."""
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.util import numpy_support

        self._remove_comparison_volumes()

        if self.tumor_volume is not None:
            self.renderer.RemoveViewProp(self.tumor_volume)
            self.tumor_volume = None

        gt_bin = np.asarray(ground_truth > 0, dtype=bool)
        if prediction is None:
            pred_bin = np.zeros_like(gt_bin, dtype=bool)
        else:
            pred_bin = np.asarray(prediction > 0, dtype=bool)

        tp = np.logical_and(pred_bin, gt_bin).astype(np.uint8)
        fp = np.logical_and(pred_bin, ~gt_bin).astype(np.uint8)
        fn = np.logical_and(~pred_bin, gt_bin).astype(np.uint8)

        label_data = np.zeros_like(tp, dtype=np.uint8)
        label_data[tp > 0] = 1
        label_data[fp > 0] = 2
        label_data[fn > 0] = 3

        if int(label_data.sum()) == 0:
            self.renderer.ResetCameraClippingRange()
            return

        vtkdata = vtkImageData()
        depth, height, width = label_data.shape
        vtkdata.SetDimensions(width, height, depth)

        vtk_array = numpy_support.numpy_to_vtk(
            num_array=label_data.ravel(), deep=True, array_type=None
        )
        vtk_array.SetNumberOfComponents(1)
        vtkdata.GetPointData().SetScalars(vtk_array)

        mapper = self._build_volume_mapper("gt-comparison")
        mapper.SetInputData(vtkdata)
        if hasattr(mapper, "SetBlendModeToComposite"):
            try:
                mapper.SetBlendModeToComposite()
            except Exception:
                pass

        prop = vtkVolumeProperty()
        prop.ShadeOff()
        prop.SetInterpolationTypeToNearest()
        prop.IndependentComponentsOn()

        eps = 0.49  # Keep plateaus almost to the next integer label boundary.
        ctf = vtkColorTransferFunction()
        ctf.AddRGBPoint(0.00, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(1.00 - eps, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(1.00, *COLOR_TP)
        ctf.AddRGBPoint(2.00 - eps, *COLOR_TP)
        ctf.AddRGBPoint(2.00, *COLOR_FP)
        ctf.AddRGBPoint(3.00 - eps, *COLOR_FP)
        ctf.AddRGBPoint(3.00, *COLOR_FN)

        otf = vtkPiecewiseFunction()
        otf.AddPoint(0.00, 0.0)
        otf.AddPoint(1.00 - eps, 0.0)
        otf.AddPoint(1.00, float(tp_opacity))
        otf.AddPoint(2.00 - eps, float(tp_opacity))
        otf.AddPoint(2.00, float(fp_opacity))
        otf.AddPoint(3.00 - eps, float(fp_opacity))
        otf.AddPoint(3.00, float(fn_opacity))

        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)

        self.comparison_volume = vtkVolume()
        self.comparison_volume.SetMapper(mapper)
        self.comparison_volume.SetProperty(prop)

        self.renderer.AddViewProp(self.comparison_volume)

        self.renderer.ResetCameraClippingRange()

    def zoom_to_tumor_centroid(self, tumor_data: np.ndarray):
        """Animate camera to tumor centroid.
        
        Args:
            tumor_data: Binary (D, H, W) uint8 array
        """
        coords = np.where(tumor_data > 0)
        if len(coords[0]) == 0:
            logger.warning("No tumor voxels to zoom to")
            return

        centroid = [float(c.mean()) for c in coords]
        logger.info(f"Zooming to tumor centroid: {centroid}")

        # Compute camera position to focus on centroid
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*centroid)
        # Position camera slightly offset
        camera.SetPosition(
            centroid[0] + 50, centroid[1] + 50, centroid[2] + 50
        )
        self.renderer.ResetCameraClippingRange()

    def reset_camera(self):
        """Reset camera to fit entire volume."""
        if self._restore_default_camera_state():
            return
        self.renderer.ResetCamera()
        self._apply_default_view()

    def _get_rotation_origin(self) -> Tuple[float, float, float]:
        # Keep navigation centered on full brain volume even after tumor overlay.
        return self.volume_center

    def rotate_horizontal(self, degrees: float):
        """Rotate around z-axis direction (left/right) around current origin."""
        camera = self.renderer.GetActiveCamera()
        focal = self._get_rotation_origin()
        camera.SetFocalPoint(*focal)
        camera.Azimuth(degrees)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()

    def rotate_vertical(self, degrees: float):
        """Rotate around x/y image axes depending on current camera orientation."""
        camera = self.renderer.GetActiveCamera()
        focal = self._get_rotation_origin()
        camera.SetFocalPoint(*focal)
        camera.Elevation(degrees)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()

    def zoom(self, factor: float):
        """Zoom camera by dolly factor (>1 zoom in, <1 zoom out)."""
        camera = self.renderer.GetActiveCamera()
        camera.Dolly(factor)
        self.renderer.ResetCameraClippingRange()

    def render(self):
        """Trigger render (called by VTKWidget)."""
        self.renderer.Render()


# VTKWidget: PyQt6 widget wrapping VTK renderer


class VTKWidget(QVTKRenderWindowInteractor):
    """PyQt6 widget embedding VTK render window."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.renderer = None
        self.iren = self.GetRenderWindow().GetInteractor()

        # Create renderer
        from vtkmodules.vtkRenderingCore import vtkRenderWindow

        self.render_window = self.GetRenderWindow()
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def event(self, event):
        """Block mouse/wheel interactions in the viewport."""
        blocked = {
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
            QEvent.Type.MouseMove,
            QEvent.Type.Wheel,
        }
        if event.type() in blocked:
            event.accept()
            return True
        return super().event(event)

    def initialize_renderer(self, renderer: vtkRenderer):
        """Initialize with VTKVolumeRenderer."""
        self.renderer = renderer
        self.render_window.AddRenderer(renderer)

    def render(self):
        """Request render update."""
        if self.render_window:
            self.render_window.Render()


# MainWindow: PyQt6 main application window


class MainWindow(QMainWindow):
    """Main GUI window with VTK viewer, controls, and metrics panel."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NOSA v3.3")
        if DEFAULT_APP_ICON.exists():
            self.setWindowIcon(QIcon(str(DEFAULT_APP_ICON)))
        self.setGeometry(100, 100, 2240, 1280)

        # Application state
        self.state = AppState()

        # Worker signals
        self.signals = SignalEmitter()
        self.signals.inference_done.connect(self.on_inference_done)
        self.signals.inference_error.connect(self.on_inference_error)
        self.signals.postprocess_done.connect(self.on_postprocess_done)
        self.signals.postprocess_error.connect(self.on_postprocess_error)
        self.signals.metric_ready.connect(self.on_metric_ready)
        self.signals.metrics_done.connect(self.on_metrics_done)
        self.signals.status_update.connect(self.on_status_update)

        # Thread pool for background workers
        self.thread_pool = QThreadPool()
        logger.info(f"Thread pool initialized with {self.thread_pool.maxThreadCount()} threads")

        # Model & threshold
        self.model = None
        self.device = None
        self.threshold = TUMOR_THRESHOLD

        # VTK setup
        self.vtk_renderer = VTKVolumeRenderer()
        self.vtk_widget = VTKWidget(self)
        self.vtk_widget.setObjectName("visualizationViewport")
        self.vtk_widget.initialize_renderer(self.vtk_renderer.renderer)

        # Build UI
        self.init_ui()

        # Load model on startup
        self.load_model()

    def _set_status(self, message: str):
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message)

    def _set_diagnosis_text_color(self, color_hex: str):
        """Update diagnosis text color only; keep textbox background/border from global theme."""
        self.text_diagnosis.setStyleSheet(f"color: {color_hex};")

    def _set_comparison_legend_visible(self, visible: bool):
        if hasattr(self, "legend_widget"):
            self.legend_widget.setVisible(visible)
        if hasattr(self, "legend_spacer"):
            self.legend_spacer.setVisible(visible)

    def init_ui(self):
        """Initialize UI components."""
        # Central widget and layouts
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)

        # LEFT PANEL: VTK viewer + controls
        left_panel = QVBoxLayout()

        # VTK viewer with right-side up/down rotation controls
        viewer_row = QHBoxLayout()
        self.viewport_frame = QWidget()
        self.viewport_frame.setObjectName("visualizationViewportFrame")
        self.viewport_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.vtk_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        viewport_layout = QVBoxLayout(self.viewport_frame)
        viewport_layout.setContentsMargins(1, 1, 1, 1)
        viewport_layout.setSpacing(0)
        viewport_layout.addWidget(self.vtk_widget)

        self.inference_overlay = QWidget(self.viewport_frame)
        self.inference_overlay.setObjectName("inferenceOverlay")
        self.inference_overlay_layout = QVBoxLayout(self.inference_overlay)
        self.inference_overlay_layout.setContentsMargins(0, 0, 0, 0)
        self.inference_overlay_layout.setSpacing(0)
        self.inference_overlay_layout.addStretch(1)
        self.inference_overlay_label = QLabel("Running inference")
        self.inference_overlay_label.setObjectName("inferenceOverlayText")
        self.inference_overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inference_overlay_layout.addWidget(self.inference_overlay_label)
        self.inference_overlay_layout.addStretch(1)
        self.inference_overlay.hide()

        self._inference_overlay_phase = 0
        self._inference_overlay_timer = QTimer(self)
        self._inference_overlay_timer.timeout.connect(self._tick_inference_overlay)

        viewer_row.addWidget(self.viewport_frame, stretch=20)

        rotate_vertical_box = QVBoxLayout()
        rotate_vertical_box.setSpacing(4)
        rotate_vertical_box.addStretch(1)
        self.btn_rotate_up = QPushButton("↑")
        self.btn_rotate_up.clicked.connect(self.on_rotate_up)
        self.btn_rotate_up.setFixedSize(20, 100)
        self.btn_rotate_up.setAutoRepeat(True)
        self.btn_rotate_up.setAutoRepeatDelay(150)
        self.btn_rotate_up.setAutoRepeatInterval(40)
        rotate_vertical_box.addWidget(self.btn_rotate_up)

        self.btn_rotate_down = QPushButton("↓")
        self.btn_rotate_down.clicked.connect(self.on_rotate_down)
        self.btn_rotate_down.setFixedSize(20, 100)
        self.btn_rotate_down.setAutoRepeat(True)
        self.btn_rotate_down.setAutoRepeatDelay(150)
        self.btn_rotate_down.setAutoRepeatInterval(40)
        rotate_vertical_box.addWidget(self.btn_rotate_down)
        rotate_vertical_box.addStretch(1)

        viewer_row.addLayout(rotate_vertical_box, stretch=2)
        left_panel.addLayout(viewer_row, stretch=12)

        # Horizontal rotation controls directly under viewer (4:1 width:height)
        self.rotate_row_widget = QWidget()
        self.rotate_row_widget.setMinimumWidth(440)
        self.rotate_row_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        rotate_horizontal_row = QHBoxLayout(self.rotate_row_widget)
        rotate_horizontal_row.setSpacing(8)
        rotate_horizontal_row.setContentsMargins(0, 0, 0, 0)
        self.btn_rotate_left = QPushButton("←")
        self.btn_rotate_left.clicked.connect(self.on_rotate_left)
        self.btn_rotate_left.setMinimumHeight(30)
        self.btn_rotate_left.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.btn_rotate_left.setAutoRepeat(True)
        self.btn_rotate_left.setAutoRepeatDelay(150)
        self.btn_rotate_left.setAutoRepeatInterval(40)
        rotate_horizontal_row.addWidget(self.btn_rotate_left, stretch=3)

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.clicked.connect(self.on_zoom_out_button)
        self.btn_zoom_out.setMinimumHeight(30)
        self.btn_zoom_out.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.btn_zoom_out.setAutoRepeat(True)
        self.btn_zoom_out.setAutoRepeatDelay(150)
        self.btn_zoom_out.setAutoRepeatInterval(40)
        rotate_horizontal_row.addWidget(self.btn_zoom_out, stretch=1)

        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.on_reset_view)
        self.btn_reset.setMinimumHeight(30)
        self.btn_reset.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        rotate_horizontal_row.addWidget(self.btn_reset, stretch=3)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.clicked.connect(self.on_zoom_in_button)
        self.btn_zoom_in.setMinimumHeight(30)
        self.btn_zoom_in.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.btn_zoom_in.setAutoRepeat(True)
        self.btn_zoom_in.setAutoRepeatDelay(150)
        self.btn_zoom_in.setAutoRepeatInterval(40)
        rotate_horizontal_row.addWidget(self.btn_zoom_in, stretch=1)

        self.btn_rotate_right = QPushButton("→")
        self.btn_rotate_right.clicked.connect(self.on_rotate_right)
        self.btn_rotate_right.setMinimumHeight(30)
        self.btn_rotate_right.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.btn_rotate_right.setAutoRepeat(True)
        self.btn_rotate_right.setAutoRepeatDelay(150)
        self.btn_rotate_right.setAutoRepeatInterval(40)
        rotate_horizontal_row.addWidget(self.btn_rotate_right, stretch=3)

        def _make_legend_entry(color: Tuple[float, float, float], text: str) -> QWidget:
            entry = QWidget()
            entry_layout = QHBoxLayout(entry)
            entry_layout.setContentsMargins(0, 0, 0, 0)
            entry_layout.setSpacing(8)

            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {QColor.fromRgbF(*color).name()}; "
                "border: 1px solid #d8d8d8; border-radius: 3px;"
            )
            entry_layout.addWidget(swatch)

            text_label = QLabel(text)
            text_label.setObjectName("legendText")
            entry_layout.addWidget(text_label)
            entry_layout.addStretch(1)
            return entry

        self.legend_widget = QWidget()
        self.legend_widget.setObjectName("comparisonLegend")
        self.legend_widget.setFixedWidth(180)
        legend_layout = QVBoxLayout(self.legend_widget)
        legend_layout.setContentsMargins(10, 6, 10, 6)
        legend_layout.setSpacing(3)

        legend_title = QLabel("GT Key")
        legend_title.setObjectName("legendTitle")
        legend_layout.addWidget(legend_title)
        legend_layout.addWidget(_make_legend_entry(COLOR_TP, "Green = TP"))
        legend_layout.addWidget(_make_legend_entry(COLOR_FP, "Yellow = FP"))
        legend_layout.addWidget(_make_legend_entry(COLOR_FN, "Red = FN"))

        def _make_opacity_slider_row(label_text: str, slider_attr: str, value_attr: str):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            label = QLabel(label_text)
            label.setObjectName("legendText")
            label.setFixedWidth(18)
            row_layout.addWidget(label)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(35)
            slider.setFixedWidth(90)
            slider.valueChanged.connect(self.on_gt_opacity_slider_changed)
            setattr(self, slider_attr, slider)
            row_layout.addWidget(slider)

            value_label = QLabel("35%")
            value_label.setObjectName("legendText")
            value_label.setFixedWidth(36)
            setattr(self, value_attr, value_label)
            row_layout.addWidget(value_label)

            return row

        legend_layout.addSpacing(4)
        legend_layout.addWidget(_make_opacity_slider_row("TP", "slider_tp_opacity", "label_tp_opacity"))
        legend_layout.addWidget(_make_opacity_slider_row("FP", "slider_fp_opacity", "label_fp_opacity"))
        legend_layout.addWidget(_make_opacity_slider_row("FN", "slider_fn_opacity", "label_fn_opacity"))

        self.button_group = QGroupBox("Controls")
        self.button_group.setMinimumWidth(440)
        self.button_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        button_layout = QVBoxLayout()

        self.btn_infer = QPushButton("Find Tumors")
        self.btn_infer.clicked.connect(self.on_find_tumors)
        self.btn_infer.setEnabled(False)
        button_layout.addWidget(self.btn_infer)

        self.btn_gt = QPushButton("Show Ground Truth")
        self.btn_gt.clicked.connect(self.on_load_ground_truth)
        self.btn_gt.setEnabled(False)
        button_layout.addWidget(self.btn_gt)

        self.btn_hole_fill = QPushButton("Apply post-processing")
        self.btn_hole_fill.clicked.connect(self.on_apply_hole_filling)
        self.btn_hole_fill.setEnabled(False)
        button_layout.addWidget(self.btn_hole_fill)

        self.button_group.setLayout(button_layout)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.on_load_image)
        self.btn_load.setMinimumWidth(440)
        self.btn_load.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        button_column_widget = QWidget()
        button_column_layout = QVBoxLayout(button_column_widget)
        button_column_layout.setContentsMargins(0, 0, 0, 0)
        button_column_layout.setSpacing(9)
        button_column_layout.addWidget(self.rotate_row_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        button_column_layout.addWidget(self.btn_load, alignment=Qt.AlignmentFlag.AlignHCenter)
        button_column_layout.addWidget(self.button_group, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.legend_spacer = QWidget()
        self.legend_spacer.setMinimumWidth(180)

        controls_legend_row = QWidget()
        controls_legend_layout = QHBoxLayout(controls_legend_row)
        controls_legend_layout.setContentsMargins(0, 0, 0, 0)
        controls_legend_layout.setSpacing(12)
        controls_legend_layout.addStretch(1)
        controls_legend_layout.addWidget(self.legend_spacer)
        controls_legend_layout.addWidget(button_column_widget)
        controls_legend_layout.addWidget(self.legend_widget, alignment=Qt.AlignmentFlag.AlignTop)
        controls_legend_layout.addStretch(1)

        self.legend_widget.hide()
        self.legend_spacer.hide()
        left_panel.addWidget(controls_legend_row)

        # RIGHT PANEL: Metrics + Diagnosis
        right_panel = QVBoxLayout()

        # Metrics display
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout()

        self.label_dice = QLabel("Dice: —")
        self.label_iou = QLabel("IoU: —")
        self.label_precision = QLabel("Precision: —")
        self.label_recall = QLabel("Recall: —")
        self.label_hd95 = QLabel("Hausdorff95: —")
        self.label_inference_time = QLabel("Inference Time: —")

        for label in [self.label_dice, self.label_iou, self.label_precision,
                 self.label_recall, self.label_hd95, self.label_inference_time]:
            label.setObjectName("metricLabel")
            label.setFont(QFont("Courier", 22, QFont.Weight.Bold))
            metrics_layout.addWidget(label)

        metrics_group.setLayout(metrics_layout)
        right_panel.addWidget(metrics_group, stretch=1)

        # Diagnosis display
        diagnosis_group = QGroupBox("Diagnosis")
        diagnosis_layout = QVBoxLayout()

        self.text_diagnosis = QTextEdit()
        self.text_diagnosis.setReadOnly(True)
        self.text_diagnosis.setFont(QFont("Verdana", 11))
        self._set_diagnosis_text_color(DIAGNOSIS_COLOR_NEUTRAL)
        self.text_diagnosis.setPlainText("Load image and run inference to see diagnosis.")
        diagnosis_layout.addWidget(self.text_diagnosis)

        diagnosis_group.setLayout(diagnosis_layout)
        right_panel.addWidget(diagnosis_group, stretch=1)

        # Add panels to main layout
        layout.addLayout(left_panel, stretch=8)
        layout.addLayout(right_panel, stretch=3)

        central.setLayout(layout)

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self._set_status("Ready | Load an image to begin")

        # Apply professional styling
        self.apply_stylesheet()
        QTimer.singleShot(0, self._update_responsive_layout)
        QTimer.singleShot(0, self._update_inference_overlay_geometry)

    def _update_responsive_layout(self):
        """Scale viewport-adjacent controls with window size for fullscreen usability."""
        if not hasattr(self, "viewport_frame"):
            return

        viewport_w = max(1, self.viewport_frame.width())
        viewport_h = max(1, self.viewport_frame.height())

        control_w = max(440, int(viewport_w * 0.58))
        self.rotate_row_widget.setFixedWidth(control_w)
        self.button_group.setFixedWidth(control_w)
        self.btn_load.setFixedWidth(control_w)

        row_h = max(30, min(46, int(viewport_h * 0.055)))
        self.btn_rotate_left.setFixedHeight(row_h)
        self.btn_zoom_out.setFixedHeight(row_h)
        self.btn_reset.setFixedHeight(row_h)
        self.btn_zoom_in.setFixedHeight(row_h)
        self.btn_rotate_right.setFixedHeight(row_h)

        v_w = max(20, min(34, int(viewport_w * 0.025)))
        v_h = max(100, min(180, int(viewport_h * 0.2)))
        self.btn_rotate_up.setFixedSize(v_w, v_h)
        self.btn_rotate_down.setFixedSize(v_w, v_h)

    def _update_inference_overlay_geometry(self):
        if hasattr(self, "inference_overlay") and hasattr(self, "viewport_frame"):
            self.inference_overlay.setGeometry(self.viewport_frame.rect())

    def _tick_inference_overlay(self):
        self._inference_overlay_phase = (self._inference_overlay_phase + 1) % 4
        dots = "." * self._inference_overlay_phase
        self.inference_overlay_label.setText(f"Running inference{dots}")

    def _show_inference_overlay(self):
        self._inference_overlay_phase = 0
        self.inference_overlay_label.setText("Running inference")
        self._update_inference_overlay_geometry()
        self.inference_overlay.show()
        self.inference_overlay.raise_()
        self._inference_overlay_timer.start(220)

    def _hide_inference_overlay(self):
        self._inference_overlay_timer.stop()
        self.inference_overlay.hide()

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        self._update_responsive_layout()
        self._update_inference_overlay_geometry()

    def apply_stylesheet(self):
        """Apply professional dark theme styling."""
        stylesheet = """
        QMainWindow {
            background-color: #39424a;
        }
        QGroupBox {
            color: #edf3f8;
            background-color: #3f4852;
            border: 1px solid #00d4ff;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 2px 0 2px;
        }
        QPushButton {
            background-color: #00d4ff;
            color: #000;
            border: none;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #00f0ff;
        }
        QPushButton:pressed {
            background-color: #0099cc;
        }
        QPushButton:disabled {
            background-color: #6f7c89;
            color: #c1cad3;
        }
        QSlider::groove:horizontal {
            border: 1px solid #00d4ff;
            height: 8px;
            margin: 2px 0px;
            background: #2f3842;
        }
        QSlider::handle:horizontal {
            background: #00d4ff;
            border: 1px solid #00a0cc;
            width: 18px;
            margin: -5px 0px;
            border-radius: 9px;
        }
        QLabel {
            color: #edf3f8;
            font-size: 11px;
        }
        QLabel#metricLabel {
            color: #f5f7ff;
            font-size: 24px;
            font-weight: 700;
        }
        QWidget#comparisonLegend {
            background-color: #3b4450;
            border: 1px solid #00d4ff;
            border-radius: 4px;
        }
        QWidget#visualizationViewportFrame {
            background-color: #11161c;
            border: 1px solid #00d4ff;
            border-radius: 4px;
        }
        QWidget#inferenceOverlay {
            background-color: rgba(14, 20, 28, 170);
            border: none;
            border-radius: 3px;
        }
        QLabel#inferenceOverlayText {
            color: #eaf7ff;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        QLabel#legendTitle {
            color: #f5f7ff;
            font-size: 11px;
            font-weight: 700;
        }
        QLabel#legendText {
            color: #edf3f8;
            font-size: 10px;
        }
        QWidget#visualizationViewport {
            background-color: #11161c;
            border: 1px solid #00d4ff;
            border-radius: 4px;
        }
        QTextEdit {
            background-color: #303944;
            color: #edf3f8;
            border: 1px solid #00d4ff;
            border-radius: 4px;
        }
        QStatusBar {
            background-color: #323b46;
            color: #edf3f8;
            border-top: 1px solid #00d4ff;
        }
        """
        self.setStyleSheet(stylesheet)

    def load_model(self):
        """Load NOSA v3.3 model on startup."""
        try:
            logger.info("Loading NOSA v3.3 model...")
            loader = ModelLoader(DEFAULT_CHECKPOINT, DEFAULT_THRESHOLD_JSON)
            self.model, self.threshold = loader.load()
            self.device = loader.device
            logger.info("Model loaded successfully")
            self._set_status("Model loaded | Ready for image loading")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    @staticmethod
    def _fill_prediction_holes(mask: np.ndarray) -> np.ndarray:
        """Automatically seal surface gaps and fill enclosed cavities per component."""
        from scipy.ndimage import (
            binary_closing,
            binary_dilation,
            binary_fill_holes,
            generate_binary_structure,
            label,
        )

        mask_bool = np.asarray(mask > 0, dtype=bool)
        if not mask_bool.any():
            return mask.astype(np.uint8, copy=True)

        structure = generate_binary_structure(3, 1)
        closing_structure = generate_binary_structure(3, 2)
        labeled = np.zeros_like(mask_bool, dtype=np.int32)
        label_result = label(mask_bool, structure=structure, output=labeled)
        if isinstance(label_result, tuple):
            num_features = int(label_result[1])
        else:
            num_features = int(label_result)
        postprocessed = np.zeros_like(mask_bool, dtype=bool)

        for component_idx in range(1, num_features + 1):
            component_mask = labeled == component_idx

            component_size = int(component_mask.sum())
            if component_size < POSTPROCESS_MIN_COMPONENT_VOXELS:
                postprocessed |= component_mask
                continue

            # Allow fills only near the original shape envelope to avoid growth spikes.
            envelope = np.asarray(
                binary_dilation(
                    component_mask,
                    structure=structure,
                    iterations=POSTPROCESS_ENVELOPE_DILATION_ITERS,
                ),
                dtype=bool,
            )

            best_candidate = np.asarray(binary_fill_holes(component_mask), dtype=bool)
            best_added = int(
                np.logical_and(best_candidate, np.logical_not(component_mask)).sum()
            )

            for closing_iters in range(1, POSTPROCESS_MAX_CLOSING_ITERS + 1):
                closed_component = np.asarray(
                    binary_closing(
                        component_mask,
                        structure=closing_structure,
                        iterations=closing_iters,
                        border_value=0,
                    ),
                    dtype=bool,
                )
                candidate = np.asarray(binary_fill_holes(closed_component), dtype=bool)

                added_voxels = np.asarray(
                    np.logical_and(candidate, np.logical_not(component_mask)),
                    dtype=bool,
                )
                added_count = int(added_voxels.sum())
                if added_count == 0:
                    continue

                outside_added = np.logical_and(added_voxels, np.logical_not(envelope))
                outside_added_ratio = int(outside_added.sum()) / max(1, component_size)
                if outside_added_ratio > POSTPROCESS_MAX_OUTSIDE_ADDED_RATIO:
                    continue

                added_ratio = added_count / max(1, component_size)
                if added_ratio > POSTPROCESS_MAX_ADDED_RATIO:
                    continue

                if added_count > best_added:
                    best_candidate = candidate
                    best_added = added_count

            postprocessed |= best_candidate

        return postprocessed.astype(np.uint8)

    def _prediction_cache_paths(self) -> Tuple[Path, Path]:
        case_id = self.state.case_id or "unknown_case"
        safe_case_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)
        return (
            PREDICTION_CACHE_DIR / f"{safe_case_id}_raw.npy",
            PREDICTION_CACHE_DIR / f"{safe_case_id}_post.npy",
        )

    def _save_prediction_cache(self):
        if self.state.raw_tumor_data is None or self.state.postprocessed_tumor_data is None:
            return
        try:
            PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            raw_path, post_path = self._prediction_cache_paths()
            np.save(raw_path, self.state.raw_tumor_data.astype(np.uint8, copy=False))
            np.save(post_path, self.state.postprocessed_tumor_data.astype(np.uint8, copy=False))
            logger.info("Saved prediction cache: %s | %s", raw_path, post_path)
        except Exception as e:
            logger.warning("Could not save prediction cache: %s", e)
    def _delete_prediction_cache(self, case_id: Optional[str]):
        """Delete cached prediction files for a given case (privacy)."""
        if case_id is None:
            return
        try:
            safe_case_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)
            raw_path = PREDICTION_CACHE_DIR / f"{safe_case_id}_raw.npy"
            post_path = PREDICTION_CACHE_DIR / f"{safe_case_id}_post.npy"
            if raw_path.exists():
                raw_path.unlink()
                logger.info(f"Deleted cached prediction: {raw_path}")
            if post_path.exists():
                post_path.unlink()
                logger.info(f"Deleted cached prediction: {post_path}")
        except Exception as e:
            logger.warning(f"Could not delete prediction cache: {e}")

    def _delete_metrics_cache(self, case_id: Optional[str]):
        """Delete cached metrics files for a given case (privacy)."""
        if case_id is None:
            return
        try:
            safe_case_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)
            normal_path = PREDICTION_CACHE_DIR / f"{safe_case_id}_normal_metrics.json"
            post_path = PREDICTION_CACHE_DIR / f"{safe_case_id}_postprocessed_metrics.json"
            if normal_path.exists():
                normal_path.unlink()
                logger.info(f"Deleted cached metrics: {normal_path}")
            if post_path.exists():
                post_path.unlink()
                logger.info(f"Deleted cached metrics: {post_path}")
        except Exception as e:
            logger.warning(f"Could not delete metrics cache: {e}")

    def _metric_cache_paths(self) -> Tuple[Path, Path]:
        """Returns (normal_metrics_path, postprocessed_metrics_path) for current case."""
        case_id = self.state.case_id or "unknown_case"
        safe_case_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_id)
        return (
            PREDICTION_CACHE_DIR / f"{safe_case_id}_normal_metrics.json",
            PREDICTION_CACHE_DIR / f"{safe_case_id}_postprocessed_metrics.json",
        )

    def _save_metrics_to_cache(self, metrics_dict: Dict[str, float], is_postprocessed: bool):
        """Save metrics to JSON cache file."""
        if not metrics_dict:
            return
        try:
            PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            normal_path, post_path = self._metric_cache_paths()
            cache_path = post_path if is_postprocessed else normal_path
            cache_data = {"metrics": metrics_dict, "timestamp": time.time()}
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            logger.info(f"Saved metrics cache ({'postprocessed' if is_postprocessed else 'normal'}): {cache_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics cache: {e}")

    def _load_metrics_from_cache(self, is_postprocessed: bool) -> Optional[Dict[str, float]]:
        """Load metrics from JSON cache file. Returns None if not found or invalid."""
        try:
            normal_path, post_path = self._metric_cache_paths()
            cache_path = post_path if is_postprocessed else normal_path
            if not cache_path.exists():
                return None
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            logger.info(f"Loaded metrics cache ({'postprocessed' if is_postprocessed else 'normal'}): {cache_path}")
            return cache_data.get("metrics", None)
        except Exception as e:
            logger.warning(f"Could not load metrics cache: {e}")
            return None

    def _start_metrics_worker(self, is_postprocessed: bool = False):
        """Start metrics computation worker for either normal or postprocessed prediction."""
        if self.state.tumor_data is None or self.state.ground_truth is None:
            return

        worker = MetricsWorker(
            self.state.tumor_data, self.state.ground_truth, self.state, self.signals, is_postprocessed=is_postprocessed
        )
        self.thread_pool.start(worker)

    def _start_postprocess_worker(self):
        if self.state.raw_tumor_data is None:
            return
        worker = PostprocessWorker(
            self.state.raw_tumor_data.astype(np.uint8, copy=True),
            self.signals,
        )
        self.thread_pool.start(worker)

    def _get_gt_overlay_opacities(self) -> Tuple[float, float, float]:
        tp = getattr(self, "slider_tp_opacity", None)
        fp = getattr(self, "slider_fp_opacity", None)
        fn = getattr(self, "slider_fn_opacity", None)
        if tp is None or fp is None or fn is None:
            return 0.35, 0.35, 0.35
        return tp.value() / 100.0, fp.value() / 100.0, fn.value() / 100.0

    def on_gt_opacity_slider_changed(self):
        tp, fp, fn = self._get_gt_overlay_opacities()
        label_tp = getattr(self, "label_tp_opacity", None)
        label_fp = getattr(self, "label_fp_opacity", None)
        label_fn = getattr(self, "label_fn_opacity", None)
        if label_tp is not None:
            label_tp.setText(f"{int(tp * 100)}%")
        if label_fp is not None:
            label_fp.setText(f"{int(fp * 100)}%")
        if label_fn is not None:
            label_fn.setText(f"{int(fn * 100)}%")

        if self.state.ground_truth is not None and self.state.gt_comparison_visible:
            self._render_current_prediction_view()

    def _render_current_prediction_view(self):
        if self.state.ground_truth is not None and self.state.gt_comparison_visible:
            tp_opacity, fp_opacity, fn_opacity = self._get_gt_overlay_opacities()
            self.vtk_renderer.show_gt_comparison(
                self.state.tumor_data,
                self.state.ground_truth,
                tp_opacity=tp_opacity,
                fp_opacity=fp_opacity,
                fn_opacity=fn_opacity,
            )
            self._set_comparison_legend_visible(True)
        elif self.state.tumor_data is not None:
            self.vtk_renderer.add_tumor_volume(self.state.tumor_data)
            self._set_comparison_legend_visible(False)
        else:
            self._set_comparison_legend_visible(False)

        self.vtk_widget.render()

    def _apply_active_tumor_mask(
        self,
        tumor_data: np.ndarray,
        *,
        inference_time_s: Optional[float] = None,
        status_message: Optional[str] = None,
    ):
        self.state.tumor_data = tumor_data.astype(np.uint8, copy=False)
        self.state.tumor_nii = Nifti1Image(self.state.tumor_data, affine=np.eye(4))
        self.state.tumor_voxel_count = int(self.state.tumor_data.sum())
        self.state.confidence = self.state.tumor_voxel_count / max(1, self.state.tumor_data.size)

        if inference_time_s is not None:
            self.state.metrics["Inference Time (s)"] = float(inference_time_s)
            self.label_inference_time.setText(f"Inference Time: {inference_time_s:.3f} s")

        self._render_current_prediction_view()
        self.generate_diagnosis()

        if self.state.ground_truth is not None:
            self._start_metrics_worker()
        elif status_message is None:
            status_message = (
                f"Tumor detected: {self.state.tumor_voxel_count} voxels | Load GT to compute metrics"
            )

        if status_message is not None:
            self._set_status(status_message)

    def on_load_image(self):
        """Load full BraTS case folder containing all modalities."""
        try:
            case_dir = QFileDialog.getExistingDirectory(
                self,
                "Select BraTS Case Folder",
                str(DEFAULT_LOAD_IMAGE_DIR if DEFAULT_LOAD_IMAGE_DIR.exists() else BRATS_TRAINING_DIR),
                QFileDialog.Option.ShowDirsOnly,
            )
            if not case_dir:
                return

            self._set_comparison_legend_visible(False)
            self.state.gt_comparison_visible = False
            # Delete cached predictions and metrics from previous case (privacy)
            old_case_id = self.state.case_id
            self._delete_prediction_cache(old_case_id)
            self._delete_metrics_cache(old_case_id)


            # Load visualization modalities with model-aligned orientation.
            image_4ch, case_id = ImageIOManager.load_case_modalities_aligned(case_dir)
            self.state.case_id = case_id
            self.state.case_dir = case_dir
            self.state.voxel_volume_mm3 = ImageIOManager.get_case_voxel_volume_mm3(case_dir)

            # Extract first modality for visualization (T1)
            brain_data = image_4ch[0]  # T1
            self.state.brain_data = brain_data
            self.state.brain_nii = Nifti1Image(brain_data, affine=np.eye(4))
            self.state.raw_tumor_data = None
            self.state.postprocessed_tumor_data = None
            self.state.tumor_data = None
            self.state.tumor_nii = None
            self.state.ground_truth = None
            self.state.tumor_voxel_count = 0
            self.state.confidence = 0.0
            self.state.postprocessing_enabled = False
            self.btn_hole_fill.setEnabled(False)
            self.btn_hole_fill.setText("Apply post-processing")
            self.btn_gt.setText("Show Ground Truth")
            self._hide_inference_overlay()

            # Force-clear any previously rendered volumes before displaying new case.
            self.vtk_renderer.clear_scene()

            # Clear metrics display for new case
            self.state.metrics = {}
            for label in [self.label_dice, self.label_iou, self.label_precision, self.label_recall, self.label_hd95]:
                label.setText(f"{label.text().split(':')[0]}: —")

            # Load brain to VTK renderer
            self.vtk_renderer.load_brain_volume(brain_data)
            self.vtk_widget.render()

            # Enable inference and GT buttons
            self.btn_infer.setEnabled(True)
            self.btn_gt.setEnabled(True)

            # Auto-find ground truth
            if case_id is not None:
                gt = ImageIOManager.find_ground_truth(case_id)
                if gt is not None:
                    self.state.ground_truth = gt
                    logger.info(f"Auto-loaded GT for {case_id}")

            self._set_status(f"Loaded case folder: {case_id}")
            self._set_diagnosis_text_color(DIAGNOSIS_COLOR_NEUTRAL)
            self.text_diagnosis.setPlainText(
                "Case loaded from folder.\nClick 'Find Tumors' to run detection."
            )

        except Exception as e:
            logger.exception(f"Load image error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def on_find_tumors(self):
        """Run inference to detect tumors."""
        try:
            if self.state.brain_data is None:
                QMessageBox.warning(self, "Error", "Load an image first")
                return

            self.btn_infer.setEnabled(False)
            self.btn_hole_fill.setEnabled(False)
            self.state.is_inferring = True

            if self.state.case_id is None:
                QMessageBox.warning(self, "Error", "Case ID is missing; reload image")
                self.btn_infer.setEnabled(True)
                self.state.is_inferring = False
                return

            if self.state.case_dir is None:
                QMessageBox.warning(self, "Error", "Case folder is missing; reload image")
                self.btn_infer.setEnabled(True)
                self.state.is_inferring = False
                return

            self._set_diagnosis_text_color(DIAGNOSIS_COLOR_NEUTRAL)
            self.text_diagnosis.setPlainText("Running inference...")
            self._show_inference_overlay()

            worker = InferenceWorker(
                self.model, self.device, self.threshold, self.state.case_dir, self.state, self.signals
            )
            self.thread_pool.start(worker)

        except Exception as e:
            logger.exception(f"Inference error: {e}")
            QMessageBox.critical(self, "Error", f"Inference failed:\n{e}")
            self.btn_infer.setEnabled(True)
            self.state.is_inferring = False
            self._hide_inference_overlay()

    def on_inference_done(self, tumor_data: np.ndarray, confidence: float, voxel_count: int, inference_time_s: float):
        """Handle inference completion."""
        self._hide_inference_overlay()
        self.state.is_inferring = False
        logger.info(
            f"Inference completed: {voxel_count} tumor voxels, "
            f"confidence: {confidence:.4f}, time: {inference_time_s:.3f}s"
        )

        self.state.raw_tumor_data = tumor_data.astype(np.uint8, copy=True)
        self.state.postprocessed_tumor_data = None
        self.state.is_postprocessing = True
        self.state.postprocessing_enabled = False
        self.state.gt_comparison_visible = False
        self.btn_hole_fill.setEnabled(False)
        self.btn_hole_fill.setText("Preparing post-processing...")
        self.btn_gt.setText("Show Ground Truth")

        self._start_postprocess_worker()

        self._apply_active_tumor_mask(
            self.state.raw_tumor_data,
            inference_time_s=inference_time_s,
        )

        self.btn_infer.setEnabled(True)

    def on_postprocess_done(self, postprocessed_mask: np.ndarray):
        self.state.postprocessed_tumor_data = postprocessed_mask.astype(np.uint8, copy=True)
        self.state.is_postprocessing = False
        self._save_prediction_cache()
        self.btn_hole_fill.setEnabled(True)
        self.btn_hole_fill.setText("Apply post-processing")
        logger.info(
            "Post-processing ready: raw=%d, post=%d voxels",
            int(self.state.raw_tumor_data.sum()) if self.state.raw_tumor_data is not None else -1,
            int(self.state.postprocessed_tumor_data.sum()),
        )
        if not self.state.postprocessing_enabled:
            self._set_status("Inference complete | Post-processing ready")

    def on_postprocess_error(self, error_msg: str):
        self.state.is_postprocessing = False
        self.state.postprocessed_tumor_data = None
        self.btn_hole_fill.setEnabled(True)
        self.btn_hole_fill.setText("Apply post-processing")
        logger.warning("Post-processing failed: %s", error_msg)
        self._set_status("Inference complete | Post-processing unavailable")

    def on_inference_error(self, error_msg: str):
        """Handle inference error."""
        self._hide_inference_overlay()
        logger.error(f"Inference error: {error_msg}")
        QMessageBox.critical(self, "Inference Error", error_msg)
        self.btn_infer.setEnabled(True)
        self.state.is_inferring = False

    def on_metric_ready(self, metric_name: str, value: float):
        """Update metric label."""
        name = metric_name
        formatted = f"{name}: {value:.4f}"
        label_map = {
            "Dice": self.label_dice,
            "IoU": self.label_iou,
            "Precision": self.label_precision,
            "Recall": self.label_recall,
            "Hausdorff95": self.label_hd95,
        }
        if name in label_map:
            label_map[name].setText(formatted)
            self.state.metrics[name] = value
            logger.info(f"Metric {name}: {value:.4f}")

    def on_metrics_done(self, is_postprocessed: bool):
        """Handle metrics completion. Cache metrics and start next computation phase."""
        logger.info(f"Metrics completed (postprocessed={is_postprocessed})")
        
        # Cache the metrics that were just computed
        self._save_metrics_to_cache(self.state.metrics, is_postprocessed=is_postprocessed)
        
        if not is_postprocessed:
            # Normal metrics just completed: display them and start postprocessed metrics
            self._set_status("Normal prediction metrics done. Computing postprocessed metrics...")
            self._start_metrics_worker(is_postprocessed=True)
        else:
            # Postprocessed metrics just completed: display only if postprocessing is enabled
            if self.state.postprocessing_enabled:
                self._set_status("Inference & metrics complete")
            else:
                self._set_status("Postprocessed metrics cached (will show when enabled)")

    def on_status_update(self, message: str):
        """Update status bar."""
        self._set_status(message)

    def on_rotate_left(self):
        self.vtk_renderer.rotate_horizontal(8.0)
        self.vtk_widget.render()
        self._set_status("Rotated left")

    def on_rotate_right(self):
        self.vtk_renderer.rotate_horizontal(-8.0)
        self.vtk_widget.render()
        self._set_status("Rotated right")

    def on_rotate_up(self):
        self.vtk_renderer.rotate_vertical(-8.0)
        self.vtk_widget.render()
        self._set_status("Rotated up")

    def on_rotate_down(self):
        self.vtk_renderer.rotate_vertical(8.0)
        self.vtk_widget.render()
        self._set_status("Rotated down")

    def on_zoom_in_button(self):
        self.vtk_renderer.zoom(1.05)
        self.vtk_widget.render()
        self._set_status("Zoomed in")

    def on_zoom_out_button(self):
        self.vtk_renderer.zoom(0.95)
        self.vtk_widget.render()
        self._set_status("Zoomed out")

    def on_reset_view(self):
        """Reset camera to fit entire volume."""
        try:
            self.vtk_renderer.reset_camera()
            self.vtk_widget.render()
            self._set_status("Camera reset")
        except Exception as e:
            logger.exception(f"Reset error: {e}")

    def on_load_ground_truth(self):
        """Toggle GT comparison overlay; when hidden, show current inference mask."""
        try:
            # Toggle OFF: return to inference-only view using current post-processing state.
            if self.state.gt_comparison_visible:
                self.state.gt_comparison_visible = False
                self.btn_gt.setText("Show Ground Truth")
                self._render_current_prediction_view()
                self._set_status("Ground truth hidden | Showing inference prediction")
                return

            if self.state.case_dir is None and self.state.case_id is None:
                QMessageBox.warning(self, "Error", "Load an image case folder first")
                return

            gt_mask = self.state.ground_truth

            # If GT not cached, load it from selected case folder or training lookup.
            if gt_mask is None and self.state.case_dir is not None:
                case_dir = Path(self.state.case_dir)
                seg_candidates = sorted(case_dir.glob("*seg.nii.gz"))
                if not seg_candidates:
                    seg_candidates = sorted(case_dir.glob("*seg.nii"))

                if seg_candidates:
                    seg_data, _ = ImageIOManager.load_nifti(str(seg_candidates[0]))
                    gt_mask = np.transpose((seg_data > 0).astype(np.uint8), (2, 1, 0))
                    logger.info(f"GT loaded from selected case folder: {seg_candidates[0]}")

            # 2) Fallback to training-data lookup by case ID.
            if gt_mask is None and self.state.case_id is not None:
                gt_mask = ImageIOManager.find_ground_truth(self.state.case_id)

            if gt_mask is None:
                QMessageBox.warning(
                    self,
                    "Ground Truth Not Found",
                    "No segmentation ground truth was found for the selected case.",
                )
                self._set_status("Ground truth not found for selected case")
                return

            self.state.ground_truth = gt_mask
            self.state.gt_comparison_visible = True
            self.btn_gt.setText("Hide Ground Truth")
            logger.info(f"GT loaded: {self.state.ground_truth.shape}")
            self._set_status("Ground truth loaded from selected case")

            # Visualize TP/FP/FN comparison overlays in 3D.
            self._render_current_prediction_view()

            # Recompute metrics if tumor already detected.
            if self.state.tumor_data is not None:
                self._start_metrics_worker()
            else:
                self._set_status(
                    "GT shown (FN in red). Run inference to also show TP (green) and FP (yellow)."
                )

        except Exception as e:
            logger.exception(f"Load GT error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load GT:\n{e}")

    def on_apply_hole_filling(self):
        """Toggle post-processing for prediction mask and refresh all outputs."""
        try:
            if self.state.raw_tumor_data is None:
                QMessageBox.warning(self, "Error", "Run inference first")
                return

            if self.state.postprocessed_tumor_data is None:
                if self.state.is_postprocessing:
                    QMessageBox.information(self, "Post-processing", "Post-processing is still running. Please wait a moment.")
                else:
                    QMessageBox.warning(self, "Post-processing", "Post-processed mask is unavailable for this case.")
                return

            self.state.postprocessing_enabled = not self.state.postprocessing_enabled

            if self.state.postprocessing_enabled:
                active_mask = self.state.postprocessed_tumor_data
                voxel_delta = int(active_mask.sum()) - int(self.state.raw_tumor_data.sum())
                self.btn_hole_fill.setText("Remove post processing")
                status = f"Post-processing enabled | Added {voxel_delta} voxels"
                
                # Try to display cached postprocessed metrics if available
                cached_post_metrics = self._load_metrics_from_cache(is_postprocessed=True)
                if cached_post_metrics:
                    self.state.metrics = cached_post_metrics
                    label_map = {
                        "Dice": self.label_dice,
                        "IoU": self.label_iou,
                        "Precision": self.label_precision,
                        "Recall": self.label_recall,
                        "Hausdorff95": self.label_hd95,
                    }
                    for name, value in cached_post_metrics.items():
                        if name in label_map:
                            label_map[name].setText(f"{name}: {value:.4f}")
                    logger.info("Displayed cached postprocessed metrics")
            else:
                active_mask = self.state.raw_tumor_data
                self.btn_hole_fill.setText("Apply post-processing")
                status = "Post-processing disabled | Raw prediction restored"
                
                # Restore normal prediction metrics if available
                cached_normal_metrics = self._load_metrics_from_cache(is_postprocessed=False)
                if cached_normal_metrics:
                    self.state.metrics = cached_normal_metrics
                    label_map = {
                        "Dice": self.label_dice,
                        "IoU": self.label_iou,
                        "Precision": self.label_precision,
                        "Recall": self.label_recall,
                        "Hausdorff95": self.label_hd95,
                    }
                    for name, value in cached_normal_metrics.items():
                        if name in label_map:
                            label_map[name].setText(f"{name}: {value:.4f}")
                    logger.info("Displayed cached normal metrics")

            self._apply_active_tumor_mask(active_mask, status_message=status)
        except Exception as e:
            logger.exception(f"Hole filling error: {e}")
            QMessageBox.critical(self, "Error", f"Hole filling failed:\n{e}")

    def generate_diagnosis(self):
        """Generate clinical diagnosis based on inference results."""
        if self.state.brain_data is None:
            self._set_diagnosis_text_color(DIAGNOSIS_COLOR_NEUTRAL)
            self.text_diagnosis.setPlainText("No brain volume loaded.")
            return

        brain_voxels = max(1, self.state.brain_data.size)
        voxel_volume_mm3 = self.state.voxel_volume_mm3 if self.state.voxel_volume_mm3 is not None else 1.0
        tumor_volume_mm3 = float(self.state.tumor_voxel_count) * float(voxel_volume_mm3)
        has_any_tumor = self.state.tumor_voxel_count > 0

        if has_any_tumor:

            diagnosis = (
                "NOSA has found a tumor\n\n"
                "RECOMMENDATION: Please visit a doctor for further evaluation.\n"
                f"Tumor Volume: {tumor_volume_mm3:.1f} mm^3\n"
            )
            self._set_diagnosis_text_color(DIAGNOSIS_COLOR_POSITIVE)
        else:
            diagnosis = (
                "NOSA has not found a tumor\n\n"
                "The model did not detect any significant tumor regions.\n\n"
                f"Suspected Tumor Volume: {tumor_volume_mm3:.1f} mm^3"
            )
            self._set_diagnosis_text_color(DIAGNOSIS_COLOR_NEGATIVE)

        self.text_diagnosis.setPlainText(diagnosis)


# Entry Point


def main():
    """Application entry point."""
    if os.name == "nt":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
        except Exception as e:
            logger.warning("Could not set Windows AppUserModelID: %s", e)

    app = QApplication(sys.argv)
    app.setApplicationName("NOSA v3.3 GUI")
    app.setApplicationVersion("3.3.0")
    if DEFAULT_APP_ICON.exists():
        app.setWindowIcon(QIcon(str(DEFAULT_APP_ICON)))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
