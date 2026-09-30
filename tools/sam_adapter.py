'''
Purpose:
Provide a unified adapter interface for SAM backends (SAM1 and SAM3) so APDe-MVS tooling
can call one consistent API regardless of upstream package differences.

Status:
In Development

Future:
- Add text/concept prompting (SAM3's open-vocabulary feature) as an alternative to
  the current SAM2-style point/box interactive path.
- Add batched prompts and richer metadata reporting.
'''

import os  # File existence validation.
from dataclasses import dataclass  # Runtime config/value objects.
from enum import Enum  # Backend enum.
from typing import Any, Dict, Optional, Tuple  # Type hints for readable contracts.

import numpy as np  # Mask normalisation.


def detect_sam3_available() -> bool:
    """Return True if the sam3 package can be imported; False otherwise."""
    try:
        import sam3  # noqa: F401
        return True
    except Exception:
        return False


class SamBackend(str, Enum):
    """Supported backends."""
    SAM1 = "sam1"
    SAM3 = "sam3"


@dataclass
class SamRuntimeConfig:
    """Validated runtime settings."""
    backend: SamBackend
    checkpoint_path: str
    config_path: Optional[str] = None
    device: str = "cuda:0"
    model_type: Optional[str] = None  # Required for SAM1.


@dataclass
class SamPredictResult:
    """Unified prediction output."""
    mask_u8: np.ndarray
    score: Optional[float] = None
    raw: Optional[Any] = None


def _validate_file(path_value: Optional[str], label: str) -> None:
    """Fail fast for missing files."""
    if path_value is None:
        raise ValueError(f"{label} is required.")
    if not os.path.isfile(path_value):
        raise FileNotFoundError(f"{label} does not exist: {path_value}")


def _normalise_mask(mask_any: Any, expected_hw: Tuple[int, int]) -> np.ndarray:
    """Force mask to APDe-MVS contract: HxW, uint8, {0,255}."""
    if mask_any is None:
        raise ValueError("Mask is None.")

    mask_np = np.asarray(mask_any)

    if mask_np.ndim == 3:
        # If multi-mask output is returned, take first for deterministic behaviour.
        mask_np = mask_np[0]

    if mask_np.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask_np.shape}")

    if mask_np.shape != expected_hw:
        raise ValueError(f"Mask shape mismatch. Expected {expected_hw}, got {mask_np.shape}")

    mask_u8 = (mask_np.astype(bool).astype(np.uint8) * 255)
    return mask_u8


class _Sam1Wrapper:
    """Legacy SAM1 wrapper."""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def predict(
        self,
        image_rgb: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box_xyxy: Optional[np.ndarray] = None,
    ) -> SamPredictResult:
        self.predictor.set_image(image_rgb)
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            multimask_output=False,
        )
        chosen_mask = masks[0] if np.asarray(masks).ndim == 3 else masks
        chosen_score = float(scores[0]) if scores is not None and len(scores) > 0 else None
        mask_u8 = _normalise_mask(chosen_mask, image_rgb.shape[:2])
        return SamPredictResult(mask_u8=mask_u8, score=chosen_score, raw={"scores": scores, "logits": logits})


def _normalise_device_for_sam3(device: str) -> str:
    """
    Map a torch-style device string to the form build_sam3_image_model expects.

    build_sam3_image_model only special-cases the literal string "cuda" (it calls
    model.cuda() when device == "cuda"); any indexed form like "cuda:0" would fall
    through and leave the model on CPU. Collapse all CUDA variants to "cuda".
    """
    if device.startswith("cuda"):
        return "cuda"
    return device


class _Sam3Wrapper:
    """
    SAM3 wrapper around the official image inference path:
    build_sam3_image_model(enable_inst_interactivity=True) wrapped by
    sam3.model.sam3_image_processor.Sam3Processor.

    Per the upstream sam3_for_sam1_task example, the flow is:
      inference_state = processor.set_image(pil_image)
      masks, scores, logits = model.predict_inst(
          inference_state, point_coords=, point_labels=, box=, multimask_output=)

    set_image must receive a PIL image: its numpy/tensor branch reads
    image.shape[-2:] (CHW assumption), which would transpose an HWC array.
    """

    def __init__(self, model: Any, processor: Any, builder_info: Dict[str, Any]) -> None:
        self.model = model
        self.processor = processor
        self.builder_info = builder_info

    def predict(
        self,
        image_rgb: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box_xyxy: Optional[np.ndarray] = None,
    ) -> SamPredictResult:
        from PIL import Image  # Local import; Pillow is a SAM3 runtime dependency.

        # Convert HWC RGB uint8 numpy to PIL so the processor reads dims correctly.
        pil_image = Image.fromarray(image_rgb)
        inference_state = self.processor.set_image(pil_image)

        masks, iou_scores, _low_res_logits = self.model.predict_inst(
            inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            multimask_output=False,
        )

        # masks is (num_masks, H, W); num_masks == 1 for multimask_output=False.
        masks_np = np.asarray(masks)
        chosen_mask = masks_np[0] if masks_np.ndim == 3 else masks_np

        chosen_score = None
        try:
            scores_np = np.asarray(iou_scores)
            if scores_np.size > 0:
                chosen_score = float(scores_np.reshape(-1)[0])
        except Exception:
            chosen_score = None

        mask_u8 = _normalise_mask(chosen_mask, image_rgb.shape[:2])

        return SamPredictResult(
            mask_u8=mask_u8,
            score=chosen_score,
            raw={"builder_info": self.builder_info, "iou_scores": iou_scores},
        )


def _build_sam3_image_predictor(
    runtime_cfg: SamRuntimeConfig,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Build the SAM3 image model and its processor.

    If checkpoint_path is non-empty, load from that local file (load_from_HF=False).
    If checkpoint_path is empty or None, auto-download from HuggingFace (load_from_HF=True).
    The BPE tokenizer vocab is bundled inside the sam3 package — no external config needed.
    """
    checkpoint_path = runtime_cfg.checkpoint_path or ""
    load_from_hf = not checkpoint_path
    if checkpoint_path:
        _validate_file(checkpoint_path, "sam3_checkpoint")

    from sam3 import build_sam3_image_model  # type: ignore
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

    device = _normalise_device_for_sam3(runtime_cfg.device)

    if load_from_hf:
        print("[INFO] SAM3 checkpoint not specified; auto-downloading from HuggingFace (facebook/sam3).", flush=True)

    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path or None,
        device=device,
        eval_mode=True,
        load_from_HF=load_from_hf,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )

    if getattr(model, "inst_interactive_predictor", None) is None:
        raise RuntimeError(
            "SAM3 image model was built without an interactive predictor. "
            "Expected model.inst_interactive_predictor to be populated when "
            "enable_inst_interactivity=True."
        )

    processor = Sam3Processor(model, device=device)

    builder_info: Dict[str, Any] = {
        "checkpoint": runtime_cfg.checkpoint_path,
        "device": device,
        "strategy": "sam3.build_sam3_image_model(enable_inst_interactivity=True)"
        " + Sam3Processor -> model.predict_inst",
    }

    return model, processor, builder_info


def build_predictor(runtime_cfg: SamRuntimeConfig) -> Any:
    """Build backend predictor wrapper."""
    # For SAM1, checkpoint is always required and must exist.
    # For SAM3, an empty checkpoint triggers auto-download — skip file validation.
    if runtime_cfg.backend == SamBackend.SAM1:
        _validate_file(runtime_cfg.checkpoint_path, "checkpoint_path")

    if runtime_cfg.backend == SamBackend.SAM1:
        if not runtime_cfg.model_type:
            raise ValueError("model_type is required for SAM1 backend.")
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        sam_model = sam_model_registry[runtime_cfg.model_type](checkpoint=runtime_cfg.checkpoint_path)
        sam_model.to(device=runtime_cfg.device)
        return _Sam1Wrapper(SamPredictor(sam_model))

    if runtime_cfg.backend == SamBackend.SAM3:
        model, processor, builder_info = _build_sam3_image_predictor(runtime_cfg)
        return _Sam3Wrapper(model, processor, builder_info)

    raise ValueError(f"Unsupported backend: {runtime_cfg.backend}")


def predict_mask(
    predictor_wrapper: Any,
    image_rgb: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    box_xyxy: Optional[np.ndarray] = None,
) -> SamPredictResult:
    """Backend-agnostic predict entry."""
    if image_rgb is None:
        raise ValueError("image_rgb is required.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got {image_rgb.shape}")
    return predictor_wrapper.predict(
        image_rgb=image_rgb,
        point_coords=point_coords,
        point_labels=point_labels,
        box_xyxy=box_xyxy,
    )