import random

import torch

from rfdetr.utilities.dynamic_batch_resize import (
    DYNAMIC_BATCH_RESIZE_KEY,
    candidate_dynamic_batch_sizes,
    choose_dynamic_batch_size,
    extract_dynamic_batch_resize_config,
    resize_nested_tensor_batch,
    resize_tensor_batch,
)
from rfdetr.utilities.tensors import NestedTensor


def test_extract_dynamic_batch_resize_from_dict_aug_config() -> None:
    config = extract_dynamic_batch_resize_config(
        {DYNAMIC_BATCH_RESIZE_KEY: {"min_size": 320, "max_size": 800}}
    )
    assert config == {"min_size": 320, "max_size": 800}


def test_extract_dynamic_batch_resize_from_list_aug_config() -> None:
    config = extract_dynamic_batch_resize_config(
        [
            {"HorizontalFlip": {"p": 0.5}},
            {DYNAMIC_BATCH_RESIZE_KEY: {"enabled": True, "p": 1.0}},
        ]
    )
    assert config == {"enabled": True, "p": 1.0}


def test_candidate_sizes_default_range_is_320_to_800_by_32() -> None:
    sizes = candidate_dynamic_batch_sizes({"min_size": 320, "max_size": 800}, divisor=32)
    assert sizes[0] == 320
    assert sizes[-1] == 800
    assert all(size % 32 == 0 for size in sizes)


def test_choose_dynamic_batch_size_respects_probability_skip() -> None:
    size = choose_dynamic_batch_size({"p": 0.0}, rng=random.Random(0))
    assert size is None


def test_resize_nested_tensor_batch_updates_tensors_mask_size_and_masks() -> None:
    samples = NestedTensor(
        torch.randn(2, 3, 512, 512),
        torch.zeros(2, 512, 512, dtype=torch.bool),
    )
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.long),
            "size": torch.tensor([512, 512], dtype=torch.int64),
            "masks": torch.ones(1, 512, 512, dtype=torch.bool),
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
            "size": torch.tensor([512, 512], dtype=torch.int64),
            "masks": torch.zeros(0, 512, 512, dtype=torch.bool),
        },
    ]

    resize_nested_tensor_batch(samples, targets, 320)

    assert samples.tensors.shape == (2, 3, 320, 320)
    assert samples.mask.shape == (2, 320, 320)
    assert targets[0]["size"].tolist() == [320, 320]
    assert targets[0]["masks"].shape == (1, 320, 320)
    assert targets[1]["masks"].shape == (0, 320, 320)


def test_resize_tensor_batch_supports_video_batches() -> None:
    images = torch.randn(2, 5, 3, 512, 512)
    resized = resize_tensor_batch(images, 384)
    assert resized.shape == (2, 5, 3, 384, 384)