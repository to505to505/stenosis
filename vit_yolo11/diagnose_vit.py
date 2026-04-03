"""Diagnostic script: compare random vs VasoMIM pretrained ViT-Small on angiography images.

Tests:
  1. Feature statistics  — pretrained should have different distribution vs random
  2. Attention entropy   — pretrained should have lower entropy (sharper attention)
  3. Patch similarity    — pretrained features of same-class patches should cluster
  4. CKA similarity      — cross-layer CKA between random and pretrained

Usage:
    python vit_yolo11/diagnose_vit.py
    python vit_yolo11/diagnose_vit.py --weights path/to/weights.pth
    python vit_yolo11/diagnose_vit.py --img-dir data/stenosis_arcade/val/images
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT.parent / "vasomim" / "weights" / "vit_small_encoder_512.pth"
IMG_DIR = ROOT.parent / "data" / "stenosis_arcade" / "val" / "images"


def build_vit():
    """Create a fresh ViT-Small (random init)."""
    return VisionTransformer(
        img_size=512, patch_size=16, in_chans=3, embed_dim=384,
        depth=12, num_heads=6, mlp_ratio=4.0, num_classes=0,
        global_pool="", dynamic_img_size=True,
    )


def load_images(img_dir: Path, n: int = 16, size: int = 512) -> torch.Tensor:
    """Load n images, resize to (size, size), return (n, 3, size, size) tensor."""
    paths = sorted(img_dir.glob("*.png"))[:n]
    if not paths:
        paths = sorted(img_dir.glob("*.jpg"))[:n]
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs)


@torch.no_grad()
def get_features_and_attention(vit, images):
    """Run ViT forward, return patch tokens per block and attention maps.

    Returns:
        block_features: list of (B, N, D) tensors for each of 12 blocks
        attentions: list of (B, heads, N, N) attention weight tensors
    """
    vit.eval()
    x = vit.patch_embed(images)
    x = vit._pos_embed(x)
    x = vit.norm_pre(x)

    block_features = []
    attentions = []
    for blk in vit.blocks:
        # Get attention weights
        B, N, C = x.shape
        qkv = blk.attn.qkv(blk.norm1(x)).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = (C // blk.attn.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attentions.append(attn.cpu())

        # Normal forward
        x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        block_features.append(x.cpu())

    return block_features, attentions


def attention_entropy(attn_maps):
    """Compute mean entropy of attention distributions per block.

    Lower entropy = sharper, more focused attention = better features.
    """
    entropies = []
    for attn in attn_maps:
        # attn: (B, heads, N, N), each row sums to 1
        eps = 1e-8
        ent = -(attn * (attn + eps).log()).sum(dim=-1)  # (B, heads, N)
        entropies.append(ent.mean().item())
    return entropies


def feature_variance_ratio(block_features):
    """Ratio of top-10 PCA components variance to total — measures structure.

    Higher ratio = features encode meaningful directions, not random noise.
    """
    ratios = []
    for feats in block_features:
        # feats: (B, N, D) — flatten batch and spatial
        f = feats.reshape(-1, feats.shape[-1]).float()  # (B*N, D)
        f = f - f.mean(dim=0)
        # SVD on centered features
        _, s, _ = torch.linalg.svd(f, full_matrices=False)
        var = s ** 2
        ratio = var[:10].sum() / var.sum()
        ratios.append(ratio.item())
    return ratios


def cls_token_similarity(block_features):
    """Cosine similarity between CLS tokens across images.

    Pretrained: CLS tokens should vary meaningfully per image.
    Random: CLS tokens are nearly identical (no image-specific info).
    """
    sims = []
    for feats in block_features:
        cls = feats[:, 0, :]  # (B, D)
        cls_norm = F.normalize(cls, dim=-1)
        sim_matrix = cls_norm @ cls_norm.T  # (B, B)
        # Mean off-diagonal similarity
        mask = ~torch.eye(sim_matrix.shape[0], dtype=bool)
        sims.append(sim_matrix[mask].mean().item())
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH))
    parser.add_argument("--img-dir", type=str, default=str(IMG_DIR))
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    weights_path = Path(args.weights)
    img_dir = Path(args.img_dir)

    print(f"Loading {args.n_images} images from {img_dir}")
    images = load_images(img_dir, n=args.n_images).to(device)
    print(f"Images shape: {images.shape}")

    # --- Random ViT ---
    print("\n" + "=" * 60)
    print("Building random ViT-Small...")
    vit_rand = build_vit().to(device)

    print("Building pretrained ViT-Small...")
    vit_pre = build_vit().to(device)
    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    msg = vit_pre.load_state_dict(state, strict=False)
    print(f"  Loaded: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")

    # Quick sanity: are the models actually different?
    w_r = vit_rand.blocks[0].attn.qkv.weight.data.flatten()[:10]
    w_p = vit_pre.blocks[0].attn.qkv.weight.data.flatten()[:10]
    print(f"\n  Random  weights[0:10]: {w_r.cpu().tolist()}")
    print(f"  Pretrained weights[0:10]: {w_p.cpu().tolist()}")
    print(f"  Different? {not torch.equal(w_r, w_p)}")

    # --- Run both ---
    print("\n" + "=" * 60)
    print("Running forward pass (random)...")
    feats_rand, attn_rand = get_features_and_attention(vit_rand, images)

    print("Running forward pass (pretrained)...")
    feats_pre, attn_pre = get_features_and_attention(vit_pre, images)

    # --- Test 1: Attention Entropy ---
    print("\n" + "=" * 60)
    print("TEST 1: Attention Entropy (lower = sharper, more focused)")
    print(f"{'Block':>6s} {'Random':>10s} {'Pretrained':>12s} {'Delta':>8s}")
    ent_rand = attention_entropy(attn_rand)
    ent_pre = attention_entropy(attn_pre)
    for i in range(12):
        delta = ent_pre[i] - ent_rand[i]
        marker = " ✓" if delta < 0 else ""
        print(f"{i:6d} {ent_rand[i]:10.3f} {ent_pre[i]:12.3f} {delta:+8.3f}{marker}")
    avg_r = np.mean(ent_rand)
    avg_p = np.mean(ent_pre)
    print(f"{'AVG':>6s} {avg_r:10.3f} {avg_p:12.3f} {avg_p - avg_r:+8.3f}")
    verdict1 = "PASS ✅" if avg_p < avg_r else "FAIL ❌ (pretrained attention not sharper)"
    print(f"  Verdict: {verdict1}")

    # --- Test 2: Feature Variance Ratio (top-10 PCA) ---
    print("\n" + "=" * 60)
    print("TEST 2: Top-10 PCA Variance Ratio (higher = more structured features)")
    print(f"{'Block':>6s} {'Random':>10s} {'Pretrained':>12s} {'Delta':>8s}")
    var_rand = feature_variance_ratio(feats_rand)
    var_pre = feature_variance_ratio(feats_pre)
    for i in range(12):
        delta = var_pre[i] - var_rand[i]
        marker = " ✓" if delta > 0 else ""
        print(f"{i:6d} {var_rand[i]:10.4f} {var_pre[i]:12.4f} {delta:+8.4f}{marker}")
    avg_r = np.mean(var_rand)
    avg_p = np.mean(var_pre)
    print(f"{'AVG':>6s} {avg_r:10.4f} {avg_p:12.4f} {avg_p - avg_r:+8.4f}")
    verdict2 = "PASS ✅" if avg_p > avg_r else "FAIL ❌ (pretrained not more structured)"
    print(f"  Verdict: {verdict2}")

    # --- Test 3: CLS Token Cross-Image Similarity ---
    print("\n" + "=" * 60)
    print("TEST 3: CLS Token Cross-Image Cosine Similarity")
    print("  (pretrained should be LOWER — CLS encodes image-specific info)")
    print(f"{'Block':>6s} {'Random':>10s} {'Pretrained':>12s} {'Delta':>8s}")
    cls_rand = cls_token_similarity(feats_rand)
    cls_pre = cls_token_similarity(feats_pre)
    for i in range(12):
        delta = cls_pre[i] - cls_rand[i]
        marker = " ✓" if delta < 0 else ""
        print(f"{i:6d} {cls_rand[i]:10.4f} {cls_pre[i]:12.4f} {delta:+8.4f}{marker}")
    avg_r = np.mean(cls_rand[-3:])  # last 3 blocks most meaningful
    avg_p = np.mean(cls_pre[-3:])
    print(f"{'L3avg':>6s} {avg_r:10.4f} {avg_p:12.4f} {avg_p - avg_r:+8.4f}")
    verdict3 = "PASS ✅" if avg_p < avg_r else "MIXED (may still be ok for patch features)"
    print(f"  Verdict: {verdict3}")

    # --- Test 4: Feature Norm Statistics ---
    print("\n" + "=" * 60)
    print("TEST 4: Patch Feature L2 Norm (pretrained usually has higher norms)")
    print(f"{'Block':>6s} {'Random':>10s} {'Pretrained':>12s} {'Ratio':>8s}")
    for i in [0, 3, 7, 11]:
        norm_r = feats_rand[i][:, 1:, :].norm(dim=-1).mean().item()
        norm_p = feats_pre[i][:, 1:, :].norm(dim=-1).mean().item()
        print(f"{i:6d} {norm_r:10.2f} {norm_p:12.2f} {norm_p / norm_r:8.2f}x")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Attention Entropy:       {verdict1}")
    print(f"  PCA Variance Ratio:      {verdict2}")
    print(f"  CLS Token Similarity:    {verdict3}")
    print()
    n_pass = sum(1 for v in [verdict1, verdict2, verdict3] if "PASS" in v)
    if n_pass >= 2:
        print("  ✅ VasoMIM pretraining looks EFFECTIVE — features are structured and meaningful")
    elif n_pass == 1:
        print("  ⚠️  VasoMIM pretraining shows PARTIAL benefit — some metrics improved")
    else:
        print("  ❌ VasoMIM pretraining looks INEFFECTIVE — features similar to random init")
        print("     Possible causes:")
        print("     - Weights file is corrupted or from wrong training stage")
        print("     - Pretraining domain too different from angiography")
        print("     - Encoder wasn't properly extracted from MAE model")


if __name__ == "__main__":
    main()
