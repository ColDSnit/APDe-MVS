'''
Purpose:
CLI entry script and importable SAMRunner class for running SAM segmentation with a
unified backend adapter so APDe-MVS can use SAM1 or SAM3 through one stable interface.

Modes:
  Per-image mode (used by external callers):
    python run_SAM.py --image_path <img> --output_mask_path <mask>
                      --sam_backend sam3 --sam3_checkpoint <path>
  Batch scene mode (used by run.py and phase_4.py):
    python run_SAM.py --work_dir <apde_data_dir> --scans <scene_name>
                      --sam_backend sam3 --sam3_checkpoint <path>

SAMRunner (importable class for use in run.py):
    from tools.run_SAM import SAMRunner
    SAMRunner(data_dir, [scan], sam_backend="sam3", sam3_checkpoint="").run()

Status:
In Development

Future:
- Add optional JSON report generation (timings, scores, checkpoint hashes).
- Promote this script into recova pipeline orchestration.
'''

# Standard library imports for CLI handling, file IO, and runtime metadata.
import argparse  # Used to parse deterministic command-line arguments.
import json  # Used for optional prompt metadata input.
import os  # Used for path checks and directory creation.
import subprocess  # Used to query git commit metadata.
import sys  # Used for explicit process exit codes.
import time  # Used for lightweight timing instrumentation.
from pathlib import Path  # Used for scene directory traversal.
from typing import List, Optional, Tuple  # Used for explicit type hints.

# Third-party imports for image handling and numeric operations.
import cv2  # Used for image loading/saving.
import numpy as np  # Used for prompt arrays and mask operations.

# Ensure the tools/ directory is on sys.path so that sam_adapter can be
# imported both when run_SAM.py is executed directly (python tools/run_SAM.py)
# and when it is imported as tools.run_SAM from run.py (where only the APDe-MVS
# root is on sys.path, not tools/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local adapter imports (single source of SAM backend logic).
from sam_adapter import (  # noqa: E402
    SamBackend,
    SamRuntimeConfig,
    build_predictor,
    detect_sam3_available,
    predict_mask,
)


_BATCH_IMAGE_EXTS: frozenset = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})


def _write_bin_mat(path: Path, mat: np.ndarray) -> None:
    """Write a single-channel uint8 mask in APD.exe's binary .bin format.

    Format: version(int32=1) rows(int32) cols(int32) type(int32=0 for CV_8UC1)
    followed by raw row-major pixel bytes.
    """
    if mat.dtype != np.uint8:                                       # APD expects uint8 masks.
        mat = mat.astype(np.uint8)                                  # Coerce if needed.
    if len(mat.shape) == 3:                                         # Reduce to single channel.
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)                 # Convert if 3-channel.
    rows, cols = mat.shape                                          # Unpack dimensions.
    with open(str(path), "wb") as f:                               # Open for binary write.
        f.write(int(1).to_bytes(4, byteorder="little"))            # version = 1.
        f.write(int(rows).to_bytes(4, byteorder="little"))         # row count.
        f.write(int(cols).to_bytes(4, byteorder="little"))         # column count.
        f.write(int(0).to_bytes(4, byteorder="little"))            # type = 0 (CV_8UC1).
        f.write(mat.tobytes())                                      # Raw pixel data.


PROMPT_MODES = ("auto", "centre_box")  # auto: box + point from the object's silhouette; centre_box: legacy prompt.


def object_prompt(image_rgb: np.ndarray, pad_fraction: float = 0.03):
    """Box around, and a point inside, the largest bright object (Otsu silhouette); None when it looks unreliable.

    The legacy prompt (a box over the central 80% of the frame) makes SAM pick the dark background ring when the
    object fills most of the image. Measured on the mouse1 heart (16 views, IoU with the silhouette): centre box 0.56
    mean (0.04-0.07 on the back view), this box + point 0.91.
    """
    grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, sil = cv2.threshold(cv2.GaussianBlur(grey, (0, 0), 2), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(sil)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # Keep the biggest blob only.
    blob = (labels == largest).astype(np.uint8)
    ih, iw = grey.shape
    area_fraction = float(blob.sum()) / float(ih * iw)
    if not 0.05 <= area_fraction <= 0.95:  # Nearly empty or nearly full frame: thresholding failed.
        return None
    x, y, w, h = cv2.boundingRect(blob)
    pad = pad_fraction * max(ih, iw)
    box = np.array([max(x - pad, 0), max(y - pad, 0), min(x + w + pad, iw - 1), min(y + h + pad, ih - 1)], np.float32)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)  # Innermost pixel = safest positive point.
    py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return box, np.array([[px, py]], np.float32), np.array([1])


class SAMRunner:
    """
    Batch SAM runner: processes all images in an APDe-MVS scene directory.

    Prefers SAM3 when sam_backend="sam3" (default). If sam3 is not importable,
    falls back to SAM1 and prints a terminal notification. The predictor is built
    once and reused across all images in the batch.

    Each image is segmented using a center bounding box prompt (inner 80% of the
    frame). The resulting binary mask is saved to <scan_dir>/sa_masks/<stem>.png.

    Args:
        data_dir: Root data directory (contains scene sub-folders).
        scans: List of scene sub-folder names to process.
        max_size: Maximum image dimension before SAM inference; 0 = no resize.
        sam_backend: "sam3" (preferred) or "sam1".
        sam3_checkpoint: Path to sam3.pt. Empty string = auto-download from HF.
        sam_checkpoint: Path to SAM1 .pth checkpoint (required for SAM1 fallback).
        sam_model_type: SAM1 model type key ("vit_h", "vit_l", "vit_b"). SAM1 only.
        device: Inference device string (e.g. "cuda:0" or "cpu").
    """

    def __init__(
        self,
        data_dir: str,
        scans: List[str],
        max_size: int = 2560,
        sam_backend: str = "sam3",
        sam3_checkpoint: str = "",
        sam_checkpoint: str = "",
        sam_model_type: str = "vit_h",
        device: str = "cuda:0",
        prompt_mode: str = "auto",
    ) -> None:
        self.data_dir = data_dir
        self.scans = scans
        self.max_size = max_size
        self.sam_backend = sam_backend
        self.sam3_checkpoint = sam3_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.device = device
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}")
        self.prompt_mode = prompt_mode

    def _resolve_backend(self) -> Tuple[str, str, Optional[str]]:
        """Return (resolved_backend, checkpoint_path, model_type_or_none).

        Falls back to SAM1 (with terminal notification) when SAM3 is requested
        but the sam3 package is not importable.
        """
        if self.sam_backend == "sam3":
            if detect_sam3_available():
                return "sam3", self.sam3_checkpoint, None
            print(
                "[WARNING] SAM3 package not available; falling back to SAM1. "
                "Install sam3 in this environment or set sam_backend='sam1' to suppress this message.",
                flush=True,
            )
            if not self.sam_checkpoint:
                print(
                    "[WARNING] SAM1 fallback requested but sam_checkpoint is not set. "
                    "SAM mask generation will be skipped.",
                    flush=True,
                )
                return "skip", "", None
            return "sam1", self.sam_checkpoint, self.sam_model_type
        return "sam1", self.sam_checkpoint, self.sam_model_type

    def _maybe_resize(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.max_size <= 0:
            return image_rgb
        h, w = image_rgb.shape[:2]
        if max(h, w) <= self.max_size:
            return image_rgb
        scale = self.max_size / max(h, w)
        return cv2.resize(image_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def run(self) -> None:
        """Process all scans; generates sa_masks/ for each."""
        backend, checkpoint, model_type = self._resolve_backend()
        if backend == "skip":
            print("[WARNING] SAMRunner: skipping all scans — no usable backend.", flush=True)
            return

        runtime_cfg = SamRuntimeConfig(
            backend=SamBackend(backend),
            checkpoint_path=checkpoint,
            config_path=None,
            device=self.device,
            model_type=model_type,
        )
        print(f"[INFO] SAMRunner: building predictor (backend={backend}) ...", flush=True)
        predictor_wrapper = build_predictor(runtime_cfg)
        print("[INFO] SAMRunner: predictor ready.", flush=True)

        for scan in self.scans:
            scan_dir = Path(self.data_dir) / scan
            images_dir = scan_dir / "images"
            masks_dir = scan_dir / "sa_masks"
            if not images_dir.is_dir():
                print(f"[WARNING] SAMRunner: images dir not found, skipping scan: {images_dir}", flush=True)
                continue
            masks_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(
                f for f in images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in _BATCH_IMAGE_EXTS
            )
            print(f"[INFO] SAMRunner: scan '{scan}' — {len(image_files)} image(s)", flush=True)

            for img_path in image_files:
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    print(f"[WARNING] SAMRunner: cannot read {img_path}, skipping.", flush=True)
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = self._maybe_resize(image_rgb)

                # Prompt: box + point from the object's silhouette (auto), else the legacy centre box.
                ih, iw = image_rgb.shape[:2]
                prompt = object_prompt(image_rgb) if self.prompt_mode == "auto" else None
                if prompt is not None:
                    box, point_coords, point_labels = prompt
                    print(f"  prompt: object box {box.round(1).tolist()} + point {point_coords[0].round(1).tolist()}", flush=True)
                else:
                    mx, my = iw * 0.1, ih * 0.1
                    box = np.array([mx, my, iw - mx, ih - my], dtype=np.float32)
                    point_coords, point_labels = None, None
                    reason = "legacy mode" if self.prompt_mode == "centre_box" else "no reliable silhouette"
                    print(f"  prompt: centre box (inner 80%) [{reason}]", flush=True)

                result = predict_mask(predictor_wrapper, image_rgb, point_coords=point_coords,
                                      point_labels=point_labels, box_xyxy=box)
                mask_path = masks_dir / (img_path.stem + ".png")
                cv2.imwrite(str(mask_path), result.mask_u8)
                # Also save .bin format for APD.exe (expects version=1 header + raw CV_8UC1 data).
                _write_bin_mat(masks_dir / (img_path.stem + ".bin"), result.mask_u8)
                print(f"  mask saved: {mask_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    """
    Defines and parses CLI arguments supporting two modes:

    Per-image mode: --image_path + --output_mask_path  (for direct single-image calls)
    Batch mode:     --work_dir  + --scans               (for phase_4.py / run.py pipeline)

    Exactly one mode must be active. In batch mode, SAM3 is preferred and falls back
    to SAM1 with a terminal notification if SAM3 is not importable.
    """
    parser = argparse.ArgumentParser(
        description="Run SAM segmentation via adapter backend for APDe-MVS-compatible mask output."
    )

    # ── Backend ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--sam_backend",
        type=str,
        default="sam3",
        choices=[SamBackend.SAM1.value, SamBackend.SAM3.value],
        help="Preferred segmentation backend (default: sam3). Falls back to SAM1 in batch mode if sam3 unavailable.",
    )

    # ── Checkpoint paths ─────────────────────────────────────────────────────
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default="",
        help="Path to SAM3 checkpoint (.pt). Leave blank for auto-download from HuggingFace.",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="",
        help="Path to SAM1 checkpoint (.pth). Required only for SAM1 backend or SAM3→SAM1 fallback.",
    )
    parser.add_argument(
        "--sam_config",
        type=str,
        default=None,
        help="Optional SAM config file. Not needed for SAM3 (BPE vocab is bundled).",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        help="SAM1 model type key (vit_h / vit_l / vit_b). Used only for SAM1 backend or fallback.",
    )

    # ── Device ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device string (e.g. cuda:0 or cpu).",
    )

    # ── Per-image mode ───────────────────────────────────────────────────────
    parser.add_argument("--image_path", type=str, default=None, help="[Per-image] Path to input RGB image.")
    parser.add_argument("--output_mask_path", type=str, default=None, help="[Per-image] Path to output mask PNG.")
    parser.add_argument("--point_xy", type=str, default=None, help='[Per-image] Single point prompt "x,y".')
    parser.add_argument("--point_label", type=int, default=1, choices=[0, 1], help="[Per-image] Point label (1=fg).")
    parser.add_argument("--box_xyxy", type=str, default=None, help='[Per-image] Box prompt "x1,y1,x2,y2".')
    parser.add_argument("--prompt_mode", type=str, default="auto", choices=PROMPT_MODES,
                        help="[Batch] auto: box + point from the object's silhouette (fallback: centre box); "
                             "centre_box: legacy inner-80%% box.")
    parser.add_argument("--output_meta_json", type=str, default=None, help="[Per-image] Optional metadata JSON output.")

    # ── Batch scene mode ─────────────────────────────────────────────────────
    parser.add_argument("--work_dir", type=str, default=None, help="[Batch] APDe data root containing scene folders.")
    parser.add_argument("--scans", type=str, nargs="+", default=None, help="[Batch] Scene sub-folder name(s).")
    parser.add_argument("--max_size", type=int, default=2560, help="[Batch] Max image dimension before inference (0=none).")

    return parser.parse_args()


def _get_git_commit_or_unknown() -> str:
    """
    Returns current git commit hash if available; otherwise returns 'unknown'.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def _parse_point_xy(point_xy: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parses optional single-point prompt string into SAM-compatible arrays.
    """
    if point_xy is None:
        return None, None

    parts = point_xy.split(",")
    if len(parts) != 2:
        raise ValueError(f"--point_xy must be formatted as x,y. Got: {point_xy}")

    x_val = float(parts[0].strip())
    y_val = float(parts[1].strip())

    point_coords = np.array([[x_val, y_val]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    return point_coords, point_labels


def _parse_box_xyxy(box_xyxy: Optional[str]) -> Optional[np.ndarray]:
    """
    Parses optional box string into SAM-compatible [x1,y1,x2,y2] array.
    """
    if box_xyxy is None:
        return None

    parts = box_xyxy.split(",")
    if len(parts) != 4:
        raise ValueError(f"--box_xyxy must be formatted as x1,y1,x2,y2. Got: {box_xyxy}")

    x1, y1, x2, y2 = [float(p.strip()) for p in parts]
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _is_per_image_mode(args: argparse.Namespace) -> bool:
    return bool(args.image_path or args.output_mask_path)


def _validate_paths(args: argparse.Namespace) -> None:
    """Validate per-image mode args before model initialisation."""
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Input image does not exist: {args.image_path}")

    # For SAM3 with auto-download, sam3_checkpoint may be empty — that's valid.
    # sam_checkpoint is only required for SAM1 backend.
    if args.sam_backend == SamBackend.SAM1.value:
        if not args.sam_checkpoint:
            raise ValueError("--sam_checkpoint is required when --sam_backend sam1")
        if not os.path.isfile(args.sam_checkpoint):
            raise FileNotFoundError(f"SAM1 checkpoint does not exist: {args.sam_checkpoint}")
    elif args.sam_backend == SamBackend.SAM3.value:
        if args.sam3_checkpoint and not os.path.isfile(args.sam3_checkpoint):
            raise FileNotFoundError(f"SAM3 checkpoint does not exist: {args.sam3_checkpoint}")

    if args.sam_config is not None and not os.path.isfile(args.sam_config):
        raise FileNotFoundError(f"SAM config does not exist: {args.sam_config}")

    if args.sam_backend == SamBackend.SAM1.value and not args.sam_model_type:
        raise ValueError("--sam_model_type is required when --sam_backend sam1")

    output_dir = os.path.dirname(args.output_mask_path) or "."
    os.makedirs(output_dir, exist_ok=True)


def _load_image_rgb(image_path: str) -> np.ndarray:
    """
    Loads input image from disk and converts BGR->RGB for SAM inference.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def _save_mask_png(mask_u8: np.ndarray, output_path: str) -> None:
    """
    Writes uint8 binary mask to PNG and validates write success.
    """
    ok = cv2.imwrite(output_path, mask_u8)
    if not ok:
        raise RuntimeError(f"Failed to write output mask: {output_path}")


def main() -> int:
    """
    CLI entry point: dispatches to per-image or batch scene mode based on args.

    Per-image mode  — requires --image_path + --output_mask_path.
    Batch mode      — requires --work_dir  + --scans.
    """
    args = _parse_args()

    # ── Batch scene mode ──────────────────────────────────────────────────────
    if args.work_dir is not None or args.scans is not None:
        if not args.work_dir or not args.scans:
            print("[ERROR] Batch mode requires both --work_dir and --scans.", file=sys.stderr)
            return 1
        try:
            runner = SAMRunner(
                data_dir=args.work_dir,
                scans=args.scans,
                max_size=args.max_size,
                sam_backend=args.sam_backend,
                sam3_checkpoint=args.sam3_checkpoint or "",
                sam_checkpoint=args.sam_checkpoint or "",
                sam_model_type=args.sam_model_type or "vit_h",
                device=args.device,
                prompt_mode=args.prompt_mode,
            )
            runner.run()
            return 0
        except Exception as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    # ── Per-image mode ────────────────────────────────────────────────────────
    if not args.image_path or not args.output_mask_path:
        print(
            "[ERROR] Specify either (--image_path + --output_mask_path) for per-image mode "
            "or (--work_dir + --scans) for batch mode.",
            file=sys.stderr,
        )
        return 1

    try:
        _validate_paths(args)

        # For SAM3: use sam3_checkpoint (may be empty for auto-download).
        # For SAM1: use sam_checkpoint.
        checkpoint = args.sam3_checkpoint if args.sam_backend == SamBackend.SAM3.value else args.sam_checkpoint

        runtime_cfg = SamRuntimeConfig(
            backend=SamBackend(args.sam_backend),
            checkpoint_path=checkpoint,
            config_path=args.sam_config,
            device=args.device,
            model_type=args.sam_model_type,
        )

        recova_commit = _get_git_commit_or_unknown()

        # Deterministic startup metadata block for logs.
        print("=== SAM Runtime Metadata ===")
        print(f"backend: {runtime_cfg.backend.value}")
        print(f"checkpoint: {runtime_cfg.checkpoint_path}")
        print(f"config: {runtime_cfg.config_path}")
        print(f"device: {runtime_cfg.device}")
        print(f"model_type: {runtime_cfg.model_type}")
        print(f"recova_commit: {recova_commit}")
        print("============================")

        image_rgb = _load_image_rgb(args.image_path)

        point_coords, point_labels_default = _parse_point_xy(args.point_xy)
        if point_coords is not None:
            point_labels = np.array([args.point_label], dtype=np.int32)
        else:
            point_labels = point_labels_default

        box_xyxy = _parse_box_xyxy(args.box_xyxy)

        t0 = time.time()
        predictor_wrapper = build_predictor(runtime_cfg)
        t1 = time.time()

        result = predict_mask(
            predictor_wrapper=predictor_wrapper,
            image_rgb=image_rgb,
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=box_xyxy,
        )
        t2 = time.time()

        _save_mask_png(result.mask_u8, args.output_mask_path)
        t3 = time.time()

        print(f"predictor_init_seconds: {t1 - t0:.4f}")
        print(f"inference_seconds: {t2 - t1:.4f}")
        print(f"save_seconds: {t3 - t2:.4f}")
        print(f"output_mask_path: {args.output_mask_path}")
        print(f"mask_shape: {result.mask_u8.shape}")
        print(f"mask_dtype: {result.mask_u8.dtype}")
        print(f"mask_unique_values: {np.unique(result.mask_u8).tolist()}")
        print(f"score: {result.score}")

        if args.output_meta_json:
            meta_dir = os.path.dirname(args.output_meta_json) or "."
            os.makedirs(meta_dir, exist_ok=True)

            payload = {
                "backend": runtime_cfg.backend.value,
                "checkpoint": runtime_cfg.checkpoint_path,
                "config": runtime_cfg.config_path,
                "device": runtime_cfg.device,
                "model_type": runtime_cfg.model_type,
                "recova_commit": recova_commit,
                "predictor_init_seconds": t1 - t0,
                "inference_seconds": t2 - t1,
                "save_seconds": t3 - t2,
                "output_mask_path": args.output_mask_path,
                "mask_shape": list(result.mask_u8.shape),
                "mask_dtype": str(result.mask_u8.dtype),
                "mask_unique_values": np.unique(result.mask_u8).tolist(),
                "score": result.score,
            }

            with open(args.output_meta_json, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)

            print(f"output_meta_json: {args.output_meta_json}")

        return 0

    except Exception as exc:
        # Deterministic single-line error output for easier log parsing.
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())