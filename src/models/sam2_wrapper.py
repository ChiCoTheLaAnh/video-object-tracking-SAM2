from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_context(device: str):
    try:
        import torch
    except ImportError:
        return nullcontext()
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _load_image_predictor(config: Dict[str, Any]):
    model_cfg_value = str(config["model_cfg"])
    model_cfg_path = Path(model_cfg_value)
    checkpoint_path = Path(config["checkpoint_path"])
    if model_cfg_path.exists():
        model_cfg = str(model_cfg_path)
    else:
        # SAM2 uses Hydra config names such as `configs/sam2.1/sam2.1_hiera_s.yaml`.
        model_cfg = model_cfg_value
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint_path}. Run `bash setup_colab.sh --with-models` first."
        )

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError("SAM2 is not installed. Run `bash setup_colab.sh --with-models` in Colab first.") from exc

    device = _resolve_device(config.get("device", "auto"))
    sam_model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    return SAM2ImagePredictor(sam_model), device


def _load_video_predictor(config: Dict[str, Any]):
    model_cfg_value = str(config["model_cfg"])
    model_cfg_path = Path(model_cfg_value)
    checkpoint_path = Path(config["checkpoint_path"])
    if model_cfg_path.exists():
        model_cfg = str(model_cfg_path)
    else:
        model_cfg = model_cfg_value
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint_path}. Run `bash setup_colab.sh --with-models` first."
        )

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError("SAM2 video predictor is unavailable. Confirm the official repo install succeeded.") from exc

    device = _resolve_device(config.get("device", "auto"))
    predictor = build_sam2_video_predictor(
        model_cfg,
        str(checkpoint_path),
        device=device,
        apply_postprocessing=bool(config.get("apply_postprocessing", True)),
        vos_optimized=bool(config.get("vos_optimized", False)),
    )
    return predictor, device


def predict_image_masks(image_rgb: np.ndarray, boxes_xyxy: Sequence[Sequence[float]], config: Dict[str, Any]) -> List[np.ndarray]:
    predictor, device = _load_image_predictor(config)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for SAM2 inference.") from exc

    masks: List[np.ndarray] = []
    with torch.inference_mode():
        with _torch_context(device):
            predictor.set_image(image_rgb)
            for box in boxes_xyxy:
                mask_batch, _, _ = predictor.predict(box=np.asarray(box)[None, :], multimask_output=False)
                mask = _to_numpy(mask_batch).squeeze()
                masks.append(mask > float(config.get("mask_threshold", 0.0)))
    return masks


def _save_video_frames_for_sam2(frames: Iterable[np.ndarray], output_dir: Path) -> Path:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_rgb in enumerate(frames):
        frame_path = output_dir / f"{index:05d}.jpg"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return output_dir


def _fallback_frame_by_frame(frames: Sequence[np.ndarray], boxes_xyxy: Sequence[Sequence[float]], config: Dict[str, Any]) -> List[np.ndarray]:
    masks_per_frame: List[np.ndarray] = []
    for frame_rgb in frames:
        frame_masks = predict_image_masks(frame_rgb, boxes_xyxy, config)
        combined = np.any(np.stack(frame_masks, axis=0), axis=0) if frame_masks else np.zeros(frame_rgb.shape[:2], dtype=bool)
        masks_per_frame.append(combined)
    return masks_per_frame


def propagate_video_masks(
    frames: Sequence[np.ndarray],
    init_boxes_xyxy: Sequence[Sequence[float]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not frames:
        return {"masks": [], "mode": "empty"}

    predictor = None
    device = _resolve_device(config.get("device", "auto"))
    if not config.get("propagation", {}).get("enabled", True):
        masks = _fallback_frame_by_frame(frames, init_boxes_xyxy, config)
        return {"masks": masks, "mode": "frame_by_frame"}

    try:
        predictor, device = _load_video_predictor(config)
        import torch

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            frame_dir = _save_video_frames_for_sam2(frames, Path(temp_dir) / "frames")
            with torch.inference_mode():
                with _torch_context(device):
                    state = predictor.init_state(video_path=str(frame_dir))
                    object_id = int(config.get("propagation", {}).get("object_id", 1))
                    start_frame = int(config.get("propagation", {}).get("start_frame", 0))
                    seed_box = np.asarray(init_boxes_xyxy[0], dtype=np.float32)
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=start_frame,
                        obj_id=object_id,
                        box=seed_box,
                    )

                    masks_by_frame = {}
                    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
                        object_ids_list = _to_numpy(object_ids).tolist() if hasattr(object_ids, "__iter__") else [int(object_ids)]
                        if object_id not in object_ids_list:
                            continue
                        matched_index = object_ids_list.index(object_id)
                        frame_mask = _to_numpy(mask_logits[matched_index]).squeeze() > float(config.get("mask_threshold", 0.0))
                        masks_by_frame[int(frame_idx)] = frame_mask

            ordered_masks = [masks_by_frame.get(index, np.zeros(frames[0].shape[:2], dtype=bool)) for index in range(len(frames))]
            return {"masks": ordered_masks, "mode": "sam2_video_predictor"}
    except Exception as exc:
        if not config.get("fallback", {}).get("allow_frame_by_frame", True):
            raise RuntimeError("SAM2 video propagation failed and frame-by-frame fallback is disabled.") from exc
        masks = _fallback_frame_by_frame(frames, init_boxes_xyxy, config)
        return {"masks": masks, "mode": "frame_by_frame_fallback", "fallback_reason": str(exc)}
