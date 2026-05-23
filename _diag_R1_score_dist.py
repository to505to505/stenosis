"""Quick diagnostic: dump score distribution from R1 model on dataset2_split_test."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

from rfdetr_video.config import Config
from rfdetr_video.model import VideoRFDETR, build_criterion
from rfdetr_temporal.dataset import build_sequence_index

ROOT = Path("/home/dsa/stenosis")
RUN_DIR = ROOT / "rfdetr_video" / "runs" / "video_overfit_R1"
DEVICE = torch.device("cuda")

with open(RUN_DIR / "config.json") as f:
    raw = json.load(f)
cfg = Config()
for k, v in raw.items():
    if hasattr(cfg, k):
        try:
            setattr(cfg, k, v)
        except Exception:
            pass

model = VideoRFDETR(cfg).to(DEVICE)
ckpt = torch.load(RUN_DIR / "best.pth", map_location="cpu", weights_only=False)
sd = ckpt.get("model", ckpt)
model.load_state_dict(sd, strict=False)
model.eval()
_criterion, postprocess = build_criterion(cfg)

mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
std_t = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)
centre = cfg.T // 2

img_dir = ROOT / "data" / "dataset2_split" / "test" / "images"
sequences = build_sequence_index(img_dir)

def _to_tensor(img, size):
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    arr = np.stack([img, img, img], axis=0)
    t = torch.from_numpy(arr)
    return (t - mean_t) / std_t

all_scores = []
n_windows = 0
with torch.no_grad():
    for pid, sid, paths in sequences[:5]:   # first 5 sequences
        n = len(paths)
        if n < cfg.T:
            windows = [list(paths) + [paths[-1]] * (cfg.T - n)]
        else:
            windows = [paths[s:s + cfg.T] for s in range(n - cfg.T + 1)]
        for win in windows:
            frames = []
            orig_h = orig_w = None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_tensor(img, cfg.img_size))
            frames_t = torch.stack(frames, dim=0).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda", enabled=True):
                out = model(frames_t, query_mode="student")
            centre_out = {
                "pred_logits": out["pred_logits"][:, centre],
                "pred_boxes": out["pred_boxes"][:, centre],
            }
            orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
            res = postprocess(centre_out, orig_size)[0]
            all_scores.append(res["scores"].cpu().numpy())
            n_windows += 1

all_scores = np.concatenate(all_scores)
print(f"n_windows={n_windows}, total detections out of postprocess = {len(all_scores)}")
print(f"  per-window mean = {len(all_scores)/n_windows:.1f}")
print(f"  min={all_scores.min():.4f}, max={all_scores.max():.4f}")
print()
print(f"Score distribution (buckets):")
edges = [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.01]
for lo, hi in zip(edges[:-1], edges[1:]):
    n = int(((all_scores >= lo) & (all_scores < hi)).sum())
    bar = "#" * min(n // 50, 60)
    print(f"  [{lo:.2f}, {hi:.2f})  n={n:6d}  {bar}")
print()
print(f"Cumulative >= thr:")
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]:
    n = int((all_scores >= thr).sum())
    print(f"  score >= {thr:.2f} : {n:6d}")
