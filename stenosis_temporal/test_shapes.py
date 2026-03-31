"""Quick smoke test: verify tensor shapes through the full pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from stenosis_temporal.config import Config
from stenosis_temporal.model.detector import StenosisTemporalDetector


def main():
    cfg = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = StenosisTemporalDetector(cfg).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.1f}M")

    B, T = 2, cfg.T
    images = torch.randn(B, T, 1, cfg.img_h, cfg.img_w, device=device)

    # Create dummy targets
    targets = []
    for b in range(B):
        frame_targets = []
        for t in range(T):
            frame_targets.append({
                "boxes": torch.tensor([[100.0, 100.0, 130.0, 130.0]], device=device),
                "labels": torch.tensor([0], dtype=torch.int64, device=device),
            })
        targets.append(frame_targets)

    # ── Training forward ──
    print("\n--- Training mode ---")
    model.train()
    losses = model(images, targets)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # ── Inference forward ──
    print("\n--- Inference mode ---")
    model.eval()
    with torch.no_grad():
        results = model(images, None)
    print(f"  Got {len(results)} frame results")
    for r in results[:3]:
        print(f"    batch={r['batch_idx']} frame={r['frame_idx']} "
              f"dets={r['boxes'].shape[0]} "
              f"score_range=[{r['scores'].min().item() if len(r['scores']) else 0:.3f}, "
              f"{r['scores'].max().item() if len(r['scores']) else 0:.3f}]")

    print("\nShape test PASSED.")


if __name__ == "__main__":
    main()
