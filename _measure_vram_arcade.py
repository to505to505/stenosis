"""One-shot VRAM/RAM measurement for the arcade_dataset2_trainval RF-DETR checkpoint."""

import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image

from rfdetr import RFDETRSmall

CKPT = Path("/home/dsa/stenosis/rfdetr_runs/arcade_dataset2_trainval/checkpoint_best_total.pth")
IMAGE = Path("/home/dsa/stenosis/data/stenosis_arcade/train/images/1.png")
SHAPE = (512, 512)


def mb(value: int) -> float:
    return round(value / 1024**2, 2)


def current_rss_bytes() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def main() -> None:
    result = {
        "checkpoint": str(CKPT),
        "image": str(IMAGE),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "inference_shape": list(SHAPE),
    }
    if not torch.cuda.is_available():
        result["error"] = "CUDA is not available"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    device = torch.device("cuda:0")
    result["device"] = str(device)
    result["device_name"] = torch.cuda.get_device_name(device)
    result["rss_before_mb"] = mb(current_rss_bytes())

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    model = RFDETRSmall(pretrain_weights=str(CKPT), device="cuda:0")
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    result["load_time_s"] = round(t1 - t0, 3)
    result["rss_after_load_mb"] = mb(current_rss_bytes())
    result["gpu_after_load"] = {
        "allocated_mb": mb(torch.cuda.memory_allocated(device)),
        "reserved_mb": mb(torch.cuda.memory_reserved(device)),
        "peak_allocated_mb": mb(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mb": mb(torch.cuda.max_memory_reserved(device)),
    }

    image = Image.open(IMAGE).convert("RGB")
    result["source_image_size"] = list(image.size)

    static_alloc = torch.cuda.memory_allocated(device)
    static_reserved = torch.cuda.memory_reserved(device)

    def run_pass() -> dict:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        rss_before = current_rss_bytes()
        s0 = time.perf_counter()
        with torch.inference_mode():
            detections = model.predict(image, threshold=0.15, shape=SHAPE)
        torch.cuda.synchronize(device)
        s1 = time.perf_counter()
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        return {
            "latency_s": round(s1 - s0, 4),
            "detections": int(len(detections)),
            "rss_before_mb": mb(rss_before),
            "rss_after_mb": mb(current_rss_bytes()),
            "gpu_current_allocated_mb": mb(torch.cuda.memory_allocated(device)),
            "gpu_current_reserved_mb": mb(torch.cuda.memory_reserved(device)),
            "gpu_peak_allocated_mb": mb(peak_alloc),
            "gpu_peak_reserved_mb": mb(peak_reserved),
            "gpu_static_allocated_before_pass_mb": mb(static_alloc),
            "gpu_static_reserved_before_pass_mb": mb(static_reserved),
            "gpu_extra_peak_vs_static_allocated_mb": mb(max(0, peak_alloc - static_alloc)),
            "gpu_extra_peak_vs_static_reserved_mb": mb(max(0, peak_reserved - static_reserved)),
        }

    result["first_pass"] = run_pass()
    result["second_pass"] = run_pass()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
