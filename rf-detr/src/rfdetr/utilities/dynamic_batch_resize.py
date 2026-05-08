"""Dynamic batch-level resize augmentation helpers."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F

from rfdetr.utilities.tensors import NestedTensor

DYNAMIC_BATCH_RESIZE_KEY = "DynamicBatchResize"


def extract_dynamic_batch_resize_config(aug_config: Any) -> Optional[dict[str, Any]]:
    """Return the DynamicBatchResize config embedded in an aug_config.

    The public augmentation surface stays ``aug_config``-based, but this transform
    is intentionally applied after collation so every image in a batch receives
    the same sampled spatial size.
    """
    if aug_config is None:
        return None

    raw_config: Any = None
    if isinstance(aug_config, Mapping):
        raw_config = aug_config.get(DYNAMIC_BATCH_RESIZE_KEY)
    elif isinstance(aug_config, Sequence) and not isinstance(aug_config, (str, bytes)):
        for entry in aug_config:
            if isinstance(entry, Mapping) and DYNAMIC_BATCH_RESIZE_KEY in entry:
                raw_config = entry[DYNAMIC_BATCH_RESIZE_KEY]
                break

    if raw_config is None:
        return None
    if raw_config is True:
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise TypeError(
            f"{DYNAMIC_BATCH_RESIZE_KEY} parameters must be a mapping, got "
            f"{type(raw_config).__name__}"
        )

    config = dict(raw_config)
    if not bool(config.get("enabled", True)):
        return None
    if float(config.get("p", 1.0)) <= 0.0:
        return None
    return config


def candidate_dynamic_batch_sizes(config: Mapping[str, Any], divisor: int = 32) -> list[int]:
    """Build valid square resize candidates from a DynamicBatchResize config."""
    divisor = max(1, int(config.get("divisor", divisor)))
    if "sizes" in config:
        raw_sizes = config["sizes"]
        if isinstance(raw_sizes, (str, bytes)) or not isinstance(raw_sizes, Sequence):
            raise TypeError("DynamicBatchResize.sizes must be a sequence of integers")
        sizes = [int(size) for size in raw_sizes]
    else:
        min_size = int(config.get("min_size", config.get("min", 320)))
        max_size = int(config.get("max_size", config.get("max", 800)))
        step = max(1, int(config.get("step", divisor)))
        start = int(math.ceil(min_size / divisor) * divisor)
        stop = int(math.floor(max_size / divisor) * divisor)
        sizes = list(range(start, stop + 1, step))

    sizes = sorted({size for size in sizes if size > 0 and size % divisor == 0})
    if not sizes:
        raise ValueError(
            f"{DYNAMIC_BATCH_RESIZE_KEY} has no valid sizes. Check min_size, "
            f"max_size, step/sizes, and divisor={divisor}."
        )
    return sizes


def choose_dynamic_batch_size(
    config: Mapping[str, Any],
    *,
    divisor: int = 32,
    rng: Optional[random.Random] = None,
) -> Optional[int]:
    """Sample a square batch size, or ``None`` when probability skips resize."""
    rng = rng or random
    p = float(config.get("p", 1.0))
    if p < 1.0 and rng.random() >= p:
        return None
    return rng.choice(candidate_dynamic_batch_sizes(config, divisor=divisor))


def resize_tensor_batch(images: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    """Resize a 4D image batch or 5D video batch to a common spatial size."""
    if isinstance(size, int):
        size_hw = (size, size)
    else:
        size_hw = (int(size[0]), int(size[1]))

    if images.ndim == 4:
        return F.interpolate(images, size=size_hw, mode="bilinear", align_corners=False)
    if images.ndim == 5:
        batch, frames, channels, _height, _width = images.shape
        resized = F.interpolate(
            images.reshape(batch * frames, channels, _height, _width),
            size=size_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch, frames, channels, *size_hw)
    raise ValueError(f"Expected a 4D or 5D tensor batch, got shape {tuple(images.shape)}")


def update_target_spatial_size(targets: Sequence[dict[str, Any]], size: int | tuple[int, int]) -> None:
    """Update train-time target size fields and masks after batch resizing."""
    if isinstance(size, int):
        height, width = size, size
    else:
        height, width = int(size[0]), int(size[1])

    for target in targets:
        if "size" in target:
            target["size"] = torch.as_tensor([height, width], dtype=torch.int64, device=target["size"].device)
        if "masks" not in target:
            continue
        masks = target["masks"]
        if not torch.is_tensor(masks) or masks.ndim != 3:
            continue
        if masks.shape[-2:] == (height, width):
            continue
        if masks.shape[0] == 0:
            target["masks"] = masks.new_zeros((0, height, width))
            continue
        resized = F.interpolate(masks[:, None].float(), size=(height, width), mode="nearest")[:, 0]
        target["masks"] = resized.to(dtype=masks.dtype)


def resize_nested_tensor_batch(
    samples: NestedTensor,
    targets: Sequence[dict[str, Any]],
    size: int | tuple[int, int],
) -> None:
    """Resize an RF-DETR NestedTensor batch in-place."""
    samples.tensors = resize_tensor_batch(samples.tensors, size)
    if samples.mask is not None:
        if isinstance(size, int):
            size_hw = (size, size)
        else:
            size_hw = (int(size[0]), int(size[1]))
        samples.mask = F.interpolate(
            samples.mask[:, None].float(),
            size=size_hw,
            mode="nearest",
        )[:, 0].bool()
    update_target_spatial_size(targets, size)