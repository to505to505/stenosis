"""Quick diagnostic: run ~400 training steps to check for NaN loss.

Usage:
    python -m stqd_det.diagnose_nan
"""
import sys
import torch
from torch.amp import GradScaler, autocast

from stqd_det.config import Config
from stqd_det.model.detector import STQDDet


def main():
    cfg = Config()
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)

    model = STQDDet(cfg).to(device)
    model.init_weights()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler("cuda", enabled=cfg.amp)

    B, T = 2, cfg.T
    nan_count = 0
    total_steps = 500

    for step in range(total_steps):
        images = torch.randn(B, T, 1, cfg.img_h, cfg.img_w, device=device)
        targets = []
        for b in range(B):
            ft = []
            for t in range(T):
                n_gt = torch.randint(0, 5, (1,)).item()
                if n_gt > 0:
                    boxes = torch.rand(n_gt, 4, device=device) * 400 + 50
                    boxes[:, 2] = boxes[:, 0] + (boxes[:, 2].abs() % 100 + 20)
                    boxes[:, 3] = boxes[:, 1] + (boxes[:, 3].abs() % 100 + 20)
                    boxes.clamp_(0, 512)
                    labels = torch.randint(0, cfg.num_classes, (n_gt,), device=device)
                else:
                    boxes = torch.zeros(0, 4, device=device)
                    labels = torch.zeros(0, dtype=torch.long, device=device)
                ft.append({"boxes": boxes, "labels": labels})
            targets.append(ft)

        optimizer.zero_grad()
        with autocast("cuda", enabled=cfg.amp):
            losses = model(images, targets)

        total_loss = losses["total_loss"]

        if not torch.isfinite(total_loss):
            nan_count += 1
            new_scale = max(scaler.get_scale() * scaler.get_backoff_factor(), 1.0)
            scaler.update(new_scale)
            if step % 10 == 0 or nan_count <= 5:
                print(f"  NaN at step {step}, scale→{scaler.get_scale():.0f}")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(
                f"  step {step}/{total_steps}  "
                f"loss={total_loss.item():.4f}  "
                f"scale={scaler.get_scale():.0f}  "
                f"nan_so_far={nan_count}"
            )

    print(f"\n{'='*50}")
    print(f"Done: {total_steps} steps, {nan_count} NaN ({nan_count/total_steps*100:.1f}%)")
    if nan_count == 0:
        print("SUCCESS: No NaN detected!")
    elif nan_count < 10:
        print("OK: Occasional NaN but recoverable")
    else:
        print("PROBLEM: Too many NaN — needs further investigation")
    print(f"Final scale: {scaler.get_scale():.0f}")
    print(f"GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
