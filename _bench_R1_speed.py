"""Benchmark inference latency / FPS for the video_overfit_R1 model
on the local GPU. One forward pass = 1 centre-frame prediction over a
T=5 window of 512x512 frames.

Reports:
  - Per-frame latency (mean ± std, ms)
  - FPS = 1000 / mean_latency_ms (centre frames per second)
  - GPU name
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from rfdetr_video.config import Config
from rfdetr_video.model import VideoRFDETR

ROOT = Path("/home/dsa/stenosis")
RUN_DIR = ROOT / "rfdetr_video" / "runs" / "video_overfit_R1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cfg(run_dir: Path) -> Config:
    with open(run_dir / "config.json") as f:
        raw = json.load(f)
    cfg = Config()
    for k, v in raw.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def _load_model(run_dir: Path, cfg: Config) -> VideoRFDETR:
    model = VideoRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def benchmark(model, cfg, *, warmup=15, iters=80, amp=True):
    T = cfg.T
    size = cfg.img_size
    x = torch.randn(1, T, 3, size, size, device=DEVICE)

    # Warm up
    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", enabled=amp):
                _ = model(x, query_mode="student")
    torch.cuda.synchronize()

    # Time
    lat_ms = np.empty(iters)
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=amp):
                _ = model(x, query_mode="student")
        torch.cuda.synchronize()
        lat_ms[i] = (time.perf_counter() - t0) * 1000.0

    return lat_ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    cfg = _load_cfg(RUN_DIR)
    print(f"Config: T={cfg.T}, img_size={cfg.img_size}, AMP=on")
    model = _load_model(RUN_DIR, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params (total): {n_params/1e6:.1f} M")

    print("\n── AMP fp16 (training-style) ──")
    lat = benchmark(model, cfg, amp=True)
    print(f"  Per centre-frame latency: {lat.mean():.2f} ms  "
          f"(std {lat.std():.2f}, min {lat.min():.2f}, p95 {np.percentile(lat,95):.2f})")
    print(f"  FPS (centre frames / sec) : {1000.0 / lat.mean():.1f}")

    print("\n── FP32 (no AMP) ──")
    lat32 = benchmark(model, cfg, amp=False)
    print(f"  Per centre-frame latency: {lat32.mean():.2f} ms  "
          f"(std {lat32.std():.2f}, min {lat32.min():.2f}, p95 {np.percentile(lat32,95):.2f})")
    print(f"  FPS (centre frames / sec) : {1000.0 / lat32.mean():.1f}")

    # Save
    out = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "T": cfg.T, "img_size": cfg.img_size, "params_M": n_params/1e6,
        "amp": {
            "latency_ms_mean": float(lat.mean()),
            "latency_ms_std":  float(lat.std()),
            "latency_ms_min":  float(lat.min()),
            "latency_ms_p95":  float(np.percentile(lat, 95)),
            "fps":             float(1000.0 / lat.mean()),
        },
        "fp32": {
            "latency_ms_mean": float(lat32.mean()),
            "latency_ms_std":  float(lat32.std()),
            "latency_ms_min":  float(lat32.min()),
            "latency_ms_p95":  float(np.percentile(lat32, 95)),
            "fps":             float(1000.0 / lat32.mean()),
        },
    }
    (RUN_DIR / "speed_benchmark.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote → {RUN_DIR / 'speed_benchmark.json'}")


if __name__ == "__main__":
    main()
