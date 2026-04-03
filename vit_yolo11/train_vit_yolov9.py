"""Train ViT-Small + YOLOv9 (GELAN neck) hybrid detector for stenosis detection.

Logs all losses (box, cls, dfl) and validation metrics (mAP50, mAP50-95,
precision, recall) to Weights & Biases.

All training hyperparameters (augmentations, optimizer, LR schedule, etc.)
are configured in train_cfg_v9.yaml. CLI args override the config file.

Usage:
    python vit_yolo11/train_vit_yolov9.py                              # use train_cfg_v9.yaml defaults
    python vit_yolo11/train_vit_yolov9.py --cfg vit_yolo11/train_cfg_v9.yaml
    python vit_yolo11/train_vit_yolov9.py --epochs 50 --batch 8        # override specific params
    python vit_yolo11/train_vit_yolov9.py --no-pretrained               # train from scratch
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ultralytics"))

# Prevent wandb from wrapping stdout/stderr and printing status messages
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

import torch
import wandb
import yaml
from ultralytics import YOLO, settings

# Ensure W&B integration is enabled
settings.update(wandb=True)

WEIGHTS_PATH = ROOT.parent / "vasomim" / "weights" / "vit_small_encoder_512.pth"
YAML_PATH = ROOT / "vit-yolov9.yaml"
YAML_PATH_MS = ROOT / "vit-yolov9-ms.yaml"
DEFAULT_CFG = ROOT / "train_cfg_v9.yaml"


def load_pretrained_vit(model: YOLO, weights_path: Path) -> None:
    """Load VasoMIM pretrained encoder weights into the ViTEncoder layer."""
    vit_encoder = model.model.model[0]  # layer 0 = ViTEncoder
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    msg = vit_encoder.vit.load_state_dict(state, strict=False)
    print(f"[ViTEncoder] Loaded pretrained weights from {weights_path}")
    print(f"  missing : {msg.missing_keys}")
    print(f"  unexpected: {msg.unexpected_keys}")


def main():
    parser = argparse.ArgumentParser(description="Train ViT-YOLOv9 stenosis detector")
    parser.add_argument("--cfg", type=str, default=str(DEFAULT_CFG),
                        help="Path to training config YAML (augmentations, optimizer, etc.)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip loading VasoMIM pretrained weights")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained ViT weights (default: vasomim/weights/vit_small_encoder_512.pth)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Use multi-scale ViT (features from blocks 3,7,11)")
    parser.add_argument("--wandb-project", type=str, default="vit-yolov9-stenosis",
                        help="W&B project name")
    # Quick overrides (take precedence over config file)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    # Load training config from YAML
    with open(args.cfg) as f:
        train_cfg = yaml.safe_load(f)

    # CLI overrides take precedence
    for key in ("data", "epochs", "batch", "imgsz", "device", "lr0", "freeze", "name"):
        val = getattr(args, key)
        if val is not None:
            train_cfg[key] = val

    # Build model from YAML
    yaml_path = YAML_PATH_MS if args.multi_scale else YAML_PATH
    model = YOLO(str(yaml_path))

    # Load pretrained ViT encoder weights
    weights_path = Path(args.weights) if args.weights else WEIGHTS_PATH
    if not args.no_pretrained and weights_path.exists():
        load_pretrained_vit(model, weights_path)
    elif not args.no_pretrained:
        print(f"[WARNING] Pretrained weights not found at {weights_path}, training from scratch")

    # Remove keys that are not model.train() arguments
    train_cfg.pop("task", None)
    train_cfg.pop("model", None)

    # Make project path absolute so Ultralytics doesn't prepend RUNS_DIR/detect/
    if "project" in train_cfg and not Path(train_cfg["project"]).is_absolute():
        train_cfg["project"] = str((ROOT.parent / train_cfg["project"]).resolve())

    # Freeze ViTEncoder for first N epochs, then unfreeze
    UNFREEZE_EPOCH = 10  # unfreeze ViT backbone after this epoch

    def unfreeze_vit_callback(trainer):
        """Unfreeze ViTEncoder after UNFREEZE_EPOCH epochs."""
        if trainer.epoch == UNFREEZE_EPOCH:
            vit_encoder = trainer.model.model[0]  # layer 0 = ViTEncoder
            # Unfreeze all ViT parameters
            for param in vit_encoder.parameters():
                param.requires_grad = True
            # Lower LR for backbone params (already in optimizer, just frozen)
            vit_param_ids = {id(p) for p in vit_encoder.parameters()}
            for pg in trainer.optimizer.param_groups:
                if any(id(p) in vit_param_ids for p in pg["params"]):
                    pg["lr"] = trainer.args.lr0 * 0.1
                    pg["initial_lr"] = trainer.args.lr0 * 0.1
            n_unfrozen = sum(p.numel() for p in vit_encoder.parameters())
            print(f"\n[INFO] Epoch {UNFREEZE_EPOCH}: Unfroze ViTEncoder ({n_unfrozen/1e6:.1f}M params, lr={trainer.args.lr0 * 0.1})")

    # If using pretrained weights, freeze ViT for first N epochs then unfreeze;
    # without pretrained weights, train everything from the start.
    if not args.no_pretrained:
        model.add_callback("on_train_epoch_start", unfreeze_vit_callback)
        train_cfg.setdefault("freeze", 1)
    else:
        train_cfg["freeze"] = 0
        print("[INFO] No pretrained weights → ViT backbone NOT frozen (freeze=0)")

    # Set wandb project name (Ultralytics callback will call wandb.init with dir=save_dir)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Train — all params come from train_cfg_v9.yaml + CLI overrides
    try:
        results = model.train(**train_cfg)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")

    # ── Post-training: save best.txt, train.csv, verify checkpoints ──
    if not hasattr(model, "trainer") or model.trainer is None:
        return
    save_dir = Path(model.trainer.save_dir)

    # Log extra config to the active wandb run
    if wandb.run:
        wandb.config.update({
            "architecture": "ViT-Small + YOLOv9 (GELAN neck)",
            "backbone": "VasoMIM ViT-Small (384d, 12 blocks)",
            "neck": "GELAN (RepNCSPELAN4 + ADown + SPPELAN)",
            "pretrained": not args.no_pretrained,
        }, allow_val_change=True)
        wandb.finish()

    # 1. Copy results.csv → train.csv (full epoch-by-epoch history)
    results_csv = save_dir / "results.csv"
    train_csv = save_dir / "train.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, train_csv)
        print(f"[INFO] Training history saved to {train_csv}")

    # 2. Write best.txt with best metrics
    best_txt = save_dir / "best.txt"
    metrics = model.trainer.metrics or {}
    best_fitness = model.trainer.best_fitness
    best_epoch = getattr(model.trainer.stopper, "best_epoch", "N/A")
    with open(best_txt, "w") as f:
        f.write("ViT-YOLOv9 Best Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best fitness:  {best_fitness}\n")
        if best_epoch != "N/A":
            f.write(f"Best epoch:    {best_epoch}\n")
        f.write("\n--- Metrics ---\n")
        for k, v in sorted(metrics.items()):
            f.write(f"{k:30s}  {v}\n")
        f.write("\n--- Config ---\n")
        for k, v in sorted(train_cfg.items()):
            f.write(f"{k:30s}  {v}\n")
    print(f"[INFO] Best metrics saved to {best_txt}")

    # 3. List saved checkpoints
    wdir = save_dir / "weights"
    if wdir.exists():
        ckpts = sorted(wdir.glob("*.pt"))
        print(f"[INFO] Checkpoints ({len(ckpts)}): {[c.name for c in ckpts]}")

    print(f"[INFO] All results saved to {save_dir}")


if __name__ == "__main__":
    main()
