"""Train RF-DETR on stenosis detection dataset.

Logs all metrics (mAP50, mAP50-95, F1, precision, recall, per-class AP,
losses) to Weights & Biases. Saves best.txt with best metrics and config.

RF-DETR uses PyTorch Lightning under the hood with DINOv2 backbone.

Usage:
    python train_rfdetr.py                                # defaults (RFDETRSmall, 100 epochs)
    python train_rfdetr.py --model-size medium             # use RFDETRMedium (576×576)
    python train_rfdetr.py --epochs 50 --batch-size 8      # override params
    python train_rfdetr.py --wandb-project my_project      # custom W&B project
    python train_rfdetr.py --no-wandb                      # disable W&B
    python train_rfdetr.py --run-test                       # evaluate on test set after training
    python train_rfdetr.py --pretrained-backbone vasomim/weights/vit_small_encoder_512.pth
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

# Suppress wandb stdout/stderr
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

ROOT = Path(__file__).resolve().parent

# RF-DETR model variants
MODEL_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# Default dataset path
DEFAULT_DATASET = str(ROOT / "data" / "stenosis_arcade")
DEFAULT_OUTPUT = str(ROOT / "rfdetr_runs")


def get_model_class(size: str):
    """Import and return the RF-DETR variant class."""
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    return classes[size]


def ensure_valid_symlink(dataset_dir: str) -> None:
    """RF-DETR expects 'valid/' but our dataset uses 'val/'. Create symlink."""
    ds = Path(dataset_dir)
    valid_dir = ds / "valid"
    val_dir = ds / "val"
    if val_dir.exists() and not valid_dir.exists():
        valid_dir.symlink_to(val_dir.resolve())
        print(f"[INFO] Created symlink: {valid_dir} -> {val_dir}")


def convert_timm_to_rfdetr(timm_state: dict) -> dict:
    """Convert timm ViT state dict to RF-DETR's DINOv2 backbone format.

    VasoMIM uses timm-style keys (blocks.0.attn.qkv.weight),
    RF-DETR uses transformers-style keys (encoder.encoder.layer.0.attention...).
    Also splits fused QKV into separate Q, K, V matrices.
    """
    rfdetr_state = {}

    for key, value in timm_state.items():
        # ── Embeddings ──
        if key == "cls_token":
            rfdetr_state["embeddings.cls_token"] = value
        elif key == "pos_embed":
            rfdetr_state["embeddings.position_embeddings"] = value
        elif key == "patch_embed.proj.weight":
            rfdetr_state["embeddings.patch_embeddings.projection.weight"] = value
        elif key == "patch_embed.proj.bias":
            rfdetr_state["embeddings.patch_embeddings.projection.bias"] = value

        # ── Transformer blocks ──
        elif key.startswith("blocks."):
            parts = key.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])
            prefix = f"encoder.layer.{layer_idx}"

            if rest == "norm1.weight":
                rfdetr_state[f"{prefix}.norm1.weight"] = value
            elif rest == "norm1.bias":
                rfdetr_state[f"{prefix}.norm1.bias"] = value
            elif rest == "norm2.weight":
                rfdetr_state[f"{prefix}.norm2.weight"] = value
            elif rest == "norm2.bias":
                rfdetr_state[f"{prefix}.norm2.bias"] = value

            # Split fused QKV → separate Q, K, V
            elif rest == "attn.qkv.weight":
                q, k, v = value.chunk(3, dim=0)
                rfdetr_state[f"{prefix}.attention.attention.query.weight"] = q
                rfdetr_state[f"{prefix}.attention.attention.key.weight"] = k
                rfdetr_state[f"{prefix}.attention.attention.value.weight"] = v
            elif rest == "attn.qkv.bias":
                q, k, v = value.chunk(3, dim=0)
                rfdetr_state[f"{prefix}.attention.attention.query.bias"] = q
                rfdetr_state[f"{prefix}.attention.attention.key.bias"] = k
                rfdetr_state[f"{prefix}.attention.attention.value.bias"] = v

            # Attention output projection
            elif rest == "attn.proj.weight":
                rfdetr_state[f"{prefix}.attention.output.dense.weight"] = value
            elif rest == "attn.proj.bias":
                rfdetr_state[f"{prefix}.attention.output.dense.bias"] = value

            # MLP (same key names)
            elif rest == "mlp.fc1.weight":
                rfdetr_state[f"{prefix}.mlp.fc1.weight"] = value
            elif rest == "mlp.fc1.bias":
                rfdetr_state[f"{prefix}.mlp.fc1.bias"] = value
            elif rest == "mlp.fc2.weight":
                rfdetr_state[f"{prefix}.mlp.fc2.weight"] = value
            elif rest == "mlp.fc2.bias":
                rfdetr_state[f"{prefix}.mlp.fc2.bias"] = value

        # ── Final layer norm ──
        elif key == "norm.weight":
            rfdetr_state["layernorm.weight"] = value
        elif key == "norm.bias":
            rfdetr_state["layernorm.bias"] = value

    return rfdetr_state


def load_vasomim_backbone(model, weights_path: str) -> None:
    """Load VasoMIM pretrained ViT weights into RF-DETR's DINOv2 backbone.

    Converts timm ViT-Small keys to RF-DETR's transformers-style format,
    splitting fused QKV into separate Q/K/V matrices.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

    timm_state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    rfdetr_state = convert_timm_to_rfdetr(timm_state)

    # backbone[0] is DinoV2 wrapper, .encoder is also DinoV2,
    # .encoder.encoder is the actual WindowedDinov2WithRegistersBackbone
    backbone_encoder = model.model.model.backbone[0].encoder.encoder
    msg = backbone_encoder.load_state_dict(rfdetr_state, strict=False)

    loaded = len(rfdetr_state) - len(msg.unexpected_keys)
    print(f"[VasoMIM] Loaded {loaded}/{len(rfdetr_state)} keys into RF-DETR backbone")
    if msg.missing_keys:
        # Expected: layer_scale, mask_token, register_tokens
        non_trivial = [k for k in msg.missing_keys
                       if "layer_scale" not in k and "mask_token" not in k and "register" not in k]
        if non_trivial:
            print(f"[VasoMIM] Missing (non-trivial): {non_trivial}")
        else:
            print(f"[VasoMIM] Missing (expected — layer_scale/mask_token): {len(msg.missing_keys)} keys")
    if msg.unexpected_keys:
        print(f"[VasoMIM] Unexpected: {msg.unexpected_keys}")


def write_best_txt(output_dir: str, train_config: dict, model_size: str) -> None:
    """Parse training logs and write best.txt with best metrics + config."""
    out = Path(output_dir)

    # 1. Try to get best epoch from checkpoint
    best_epoch = "N/A"
    best_ckpt = out / "checkpoint_best_total.pth"
    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
        # PTL stores epoch in loops.fit_loop.epoch_progress
        try:
            epoch_progress = ckpt["loops"]["fit_loop"]["epoch_progress"]
            best_epoch = epoch_progress["total"]["completed"] - 1  # 0-indexed
        except (KeyError, TypeError):
            best_epoch = ckpt.get("epoch", "N/A")

    # 2. Parse metrics from CSV log
    # RF-DETR saves metrics.csv at the output_dir root
    metrics_csv = out / "metrics.csv"
    if not metrics_csv.exists():
        # Fallback: check lightning_logs path
        metrics_csv = out / "lightning_logs" / "version_0" / "metrics.csv"
    best_metrics = {}
    best_map = 0.0

    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)

        # Find best epoch by val/mAP_50_95
        if "val/mAP_50_95" in df.columns:
            val_df = df.dropna(subset=["val/mAP_50_95"])
            if len(val_df) > 0:
                best_idx = val_df["val/mAP_50_95"].idxmax()
                best_row = val_df.loc[best_idx]
                best_map = best_row["val/mAP_50_95"]

                # Collect all val/ and ema/ metrics from best row
                for col in sorted(df.columns):
                    if col.startswith(("val/", "train/")):
                        v = best_row.get(col)
                        if pd.notna(v):
                            best_metrics[col] = v

        # Also get EMA mAP if available
        if "val/ema_mAP_50_95" in df.columns:
            ema_df = df.dropna(subset=["val/ema_mAP_50_95"])
            if len(ema_df) > 0:
                best_ema_idx = ema_df["val/ema_mAP_50_95"].idxmax()
                best_ema_row = ema_df.loc[best_ema_idx]
                for col in sorted(df.columns):
                    if col.startswith("val/ema_"):
                        v = best_ema_row.get(col)
                        if pd.notna(v):
                            best_metrics[col] = v

    # 3. Write best.txt
    best_txt = out / "best.txt"
    with open(best_txt, "w") as f:
        f.write(f"RF-DETR ({MODEL_VARIANTS[model_size]}) Best Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best mAP50-95: {best_map:.5f}\n")
        f.write(f"Best epoch:    {best_epoch}\n")

        f.write("\n--- Metrics ---\n")
        for k, v in sorted(best_metrics.items()):
            f.write(f"{k:35s}  {v}\n")

        f.write("\n--- Config ---\n")
        for k, v in sorted(train_config.items()):
            f.write(f"{k:35s}  {v}\n")

    print(f"[INFO] Best metrics saved to {best_txt}")


def copy_train_csv(output_dir: str) -> None:
    """Copy metrics.csv → train.csv for consistency."""
    out = Path(output_dir)
    # RF-DETR saves metrics.csv at the output_dir root
    metrics_csv = out / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = out / "lightning_logs" / "version_0" / "metrics.csv"
    train_csv = out / "train.csv"
    if metrics_csv.exists():
        shutil.copy2(metrics_csv, train_csv)
        print(f"[INFO] Training history saved to {train_csv}")


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR stenosis detector")

    # Model
    parser.add_argument("--model-size", type=str, default="small",
                        choices=list(MODEL_VARIANTS.keys()),
                        help="RF-DETR variant: nano (384), small (512), medium (576), large (704)")
    parser.add_argument("--pretrained-backbone", type=str, default=None,
                        help="Path to VasoMIM ViT-Small weights (.pth) to replace DINOv2 backbone")

    # Data
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET,
                        help="Path to YOLO-format dataset directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints/logs (default: rfdetr_runs/<name>)")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (default: auto-generated from model size + timestamp)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder learning rate")
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4, help="Backbone encoder learning rate")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["step", "cosine"])
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs)")

    # Evaluation
    parser.add_argument("--run-test", action="store_true", default=True,
                        help="Run evaluation on test set after training")
    parser.add_argument("--no-run-test", dest="run_test", action="store_false")
    parser.add_argument("--checkpoint-interval", type=int, default=10)

    # Logging
    parser.add_argument("--wandb-project", type=str, default="rfdetr-stenosis")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # Hardware
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device index or 'cpu'")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── Run name & output directory ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"rfdetr_{args.model_size}_{timestamp}"
    output_dir = args.output_dir or str(ROOT / "rfdetr_runs" / run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Ensure valid/ symlink exists ──
    ensure_valid_symlink(args.dataset_dir)

    # ── Build model ──
    ModelClass = get_model_class(args.model_size)
    model = ModelClass()
    print(f"[INFO] Model: {MODEL_VARIANTS[args.model_size]}")

    # ── Load custom backbone weights ──
    if args.pretrained_backbone:
        load_vasomim_backbone(model, args.pretrained_backbone)

    # ── Grayscale-optimized augmentations ──
    # Stenosis angiograms are grayscale — disable color augmentations,
    # keep geometric + brightness augmentations
    aug_config = {
        # Geometric
        "HorizontalFlip": {"p": 0.5},
        "VerticalFlip": {"p": 0.5},
        "Rotate": {"limit": 20, "p": 0.3},
        "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.05, 0.05), "p": 0.3},
        "Perspective": {"scale": (0.02, 0.05), "p": 0.15},
        # Pixel-level (grayscale-safe)
        "RandomBrightnessContrast": {"brightness_limit": 0.3, "contrast_limit": 0.3, "p": 0.5},
        "CLAHE": {"clip_limit": 4.0, "p": 0.3},
        "RandomGamma": {"gamma_limit": (80, 120), "p": 0.3},
        "GaussianBlur": {"blur_limit": 3, "p": 0.2},
        "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.2},
        "Sharpen": {"alpha": (0.2, 0.5), "lightness": (0.5, 1.0), "p": 0.2},
    }

    # ── Assemble training config ──
    train_config = {
        "dataset_dir": args.dataset_dir,
        "output_dir": output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "lr_scheduler": args.lr_scheduler,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "use_ema": True,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.patience,
        "run_test": args.run_test,
        "checkpoint_interval": args.checkpoint_interval,
        "log_per_class_metrics": True,
        "class_names": ["stenosis_0", "stenosis_1"],
        "wandb": not args.no_wandb,
        "project": args.wandb_project,
        "run": run_name,
        "tensorboard": True,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "aug_config": aug_config,
        "progress_bar": "rich",
    }

    # Print config summary
    print(f"\n{'='*50}")
    print(f"RF-DETR Training Config")
    print(f"{'='*50}")
    for k, v in sorted(train_config.items()):
        print(f"  {k:30s}  {v}")
    print(f"{'='*50}\n")

    # ── Train ──
    try:
        model.train(**train_config)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    finally:
        # ── Post-training: save best.txt, train.csv ──

        # Create a flat config dict for best.txt
        flat_config = {k: str(v) for k, v in train_config.items()}
        flat_config["model_size"] = args.model_size
        flat_config["model_variant"] = MODEL_VARIANTS[args.model_size]
        flat_config["pretrained_backbone"] = str(args.pretrained_backbone or "DINOv2 (default)")

        write_best_txt(output_dir, flat_config, args.model_size)
        copy_train_csv(output_dir)

        # List saved checkpoints
        ckpts = sorted(Path(output_dir).glob("*.pth"))
        if ckpts:
            print(f"[INFO] Checkpoints ({len(ckpts)}): {[c.name for c in ckpts]}")

        print(f"[INFO] All results saved to {output_dir}")


if __name__ == "__main__":
    main()
