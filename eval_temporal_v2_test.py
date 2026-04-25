"""Evaluate Temporal RF-DETR (temporal_v2) on dataset2_split test set."""

import json
from pathlib import Path

import torch

from rfdetr_temporal.config import Config
from rfdetr_temporal.dataset import get_dataloader
from rfdetr_temporal.model import TemporalRFDETR, _build_criterion
from rfdetr_temporal.evaluate import evaluate


def main():
    # Load config from the run
    run_dir = Path("rfdetr_temporal/runs/temporal_v2")
    with open(run_dir / "config.json") as f:
        cfg_dict = json.load(f)

    cfg = Config(**{
        k: v for k, v in cfg_dict.items()
        if k in Config.__dataclass_fields__
    })
    # Override data_root to Path
    cfg.data_root = Path(cfg_dict["data_root"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = TemporalRFDETR(cfg).to(device)

    # Load best checkpoint
    ckpt_path = run_dir / "best.pth"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Build criterion and postprocess
    criterion, postprocess = _build_criterion(cfg)
    criterion = criterion.to(device)

    # Build test dataloader
    test_loader = get_dataloader("test", cfg, shuffle=False)

    # Run evaluation
    metrics = evaluate(model, test_loader, criterion, postprocess, cfg, device)

    # Print results
    print("\n" + "=" * 50)
    print("Temporal RF-DETR v2 — Test Results (dataset2_split)")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s}  {v:.5f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
