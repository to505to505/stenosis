"""Count-stability diagnostic for Video RF-DETR on cadica_50plus_new.

This is an evaluation-only check for the trained L_num behavior. It does not
disable the loss, because that would require retraining; instead it measures the
raw per-frame counts implied by the model's (B,T,Q,K) logits.

Usage:
    PYTHONPATH=rf-detr/src PYTHONUNBUFFERED=1 \
      /home/dsa/miniconda3/envs/stenosis/bin/python -u _eval_cadica_video_count_variance.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from _eval_cadica_video_ablate import IMG_DIR, ROOT, _load_cfg_video, _to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_DIR = ROOT / "rfdetr_video" / "runs" / "stfs_nodistill_v6_etf"


class CountStats:
    def __init__(self) -> None:
        self.windows = 0
        self.frames = 0
        self.soft_abs_dev_sum = 0.0
        self.soft_var_sum = 0.0
        self.hard_var_sum = 0.0
        self.hard_range_sum = 0.0
        self.exact_stable = 0
        self.any_hard_flicker = 0
        self.four_of_five_gap = 0
        self.examples: list[dict] = []

    def record(self, key: str, frame_names: list[str], logits: torch.Tensor, threshold: float, soft_temp: float) -> None:
        probs = logits.sigmoid().amax(dim=-1)          # (T, Q)
        soft = torch.sigmoid((probs - threshold) / soft_temp).sum(dim=-1)
        hard = (probs >= threshold).sum(dim=-1).float()
        max_prob = probs.max(dim=-1).values
        soft_ref = soft.median()

        soft_np = soft.detach().cpu().numpy().astype(float)
        hard_np = hard.detach().cpu().numpy().astype(float)
        max_prob_np = max_prob.detach().cpu().numpy().astype(float)

        self.windows += 1
        self.frames += int(logits.shape[0])
        self.soft_abs_dev_sum += float(np.abs(soft_np - float(soft_ref)).mean())
        self.soft_var_sum += float(np.var(soft_np))
        self.hard_var_sum += float(np.var(hard_np))
        hard_range = float(hard_np.max() - hard_np.min())
        self.hard_range_sum += hard_range
        if hard_range == 0.0:
            self.exact_stable += 1
        if hard_np.max() > 0 and hard_np.min() == 0:
            self.any_hard_flicker += 1
        if int((hard_np > 0).sum()) == len(hard_np) - 1 and int((hard_np == 0).sum()) == 1:
            self.four_of_five_gap += 1
            if len(self.examples) < 8:
                self.examples.append({
                    "window": key,
                    "frames": frame_names,
                    "hard_counts": [int(x) for x in hard_np.tolist()],
                    "soft_counts": [round(float(x), 3) for x in soft_np.tolist()],
                    "max_probs": [round(float(x), 4) for x in max_prob_np.tolist()],
                    "soft_median_ref": round(float(soft_ref), 3),
                })

    def summary(self) -> dict:
        windows = max(self.windows, 1)
        return {
            "windows": self.windows,
            "frames": self.frames,
            "mean_soft_abs_dev_from_median": self.soft_abs_dev_sum / windows,
            "mean_soft_count_variance": self.soft_var_sum / windows,
            "mean_hard_count_variance": self.hard_var_sum / windows,
            "mean_hard_count_range": self.hard_range_sum / windows,
            "exact_hard_count_stability_rate": self.exact_stable / windows,
            "hard_flicker_window_rate": self.any_hard_flicker / windows,
            "four_of_five_gap_rate": self.four_of_five_gap / windows,
            "four_of_five_gap_examples": self.examples,
        }


def _windows(paths: list[Path], T: int) -> list[list[Path]]:
    n = len(paths)
    if n == 0:
        return []
    if n < T:
        return [list(paths) + [paths[-1]] * (T - n)]
    return [paths[start:start + T] for start in range(n - T + 1)]


def main() -> None:
    from rfdetr_temporal.dataset import build_sequence_index
    from rfdetr_video.model import VideoRFDETR

    cfg = _load_cfg_video(RUN_DIR)
    cfg.rfdetr_checkpoint = str(RUN_DIR / "best.pth")
    threshold = float(cfg.consistency_threshold)
    soft_temp = max(float(cfg.consistency_soft_temp), 1e-6)
    print(f"run={RUN_DIR}")
    print(f"T={cfg.T} img_size={cfg.img_size} threshold={threshold} soft_temp={soft_temp}")

    model = VideoRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(RUN_DIR / "best.pth", map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    print(f"loaded missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()

    mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)
    sequences = build_sequence_index(IMG_DIR)
    stats = {
        "final": CountStats(),
        "first_pass": CountStats(),
    }
    t0 = time.time()

    for seq_idx, (pid, sid, paths) in enumerate(sequences):
        for start_idx, window in enumerate(_windows(paths, cfg.T)):
            frames = []
            for path in window:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                frames.append(_to_tensor(img, cfg.img_size, mean, std))
            ft = torch.stack(frames, 0).unsqueeze(0).to(DEVICE)
            key = f"{pid}_v{sid}:{start_idx}"
            frame_names = [path.name for path in window]

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = model(ft, query_mode="student")
            stats["final"].record(key, frame_names, out["pred_logits"][0], threshold, soft_temp)
            stats["first_pass"].record(key, frame_names, out["first_pass"]["pred_logits"][0], threshold, soft_temp)

        if (seq_idx + 1) % 40 == 0 or (seq_idx + 1) == len(sequences):
            print(f"  {seq_idx + 1}/{len(sequences)}  {time.time() - t0:.0f}s")

    result = {
        "run": "video_nodistill_v6_etf",
        "run_dir": str(RUN_DIR),
        "threshold": threshold,
        "soft_temp": soft_temp,
        "variants": {name: stat.summary() for name, stat in stats.items()},
    }
    out_path = ROOT / "_eval_cadica_video_count_variance_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()