"""
Extract encoder-only weights from a VasoMIM pretrain checkpoint
and interpolate pos_embed for a target image size.

Usage:
    python extract_encoder.py \
        --checkpoint weights/checkpoint-300.pth \
        --output weights/vit_small_encoder.pth \
        --target_img_size 512 \
        --patch_size 16
"""
import argparse
import torch
import torch.nn.functional as F


ENCODER_PREFIXES = (
    "cls_token",
    "pos_embed",
    "patch_embed.",
    "blocks.",
    "norm.",
)


def interpolate_pos_embed(pos_embed, target_img_size, patch_size):
    """
    pos_embed: [1, 1 + old_num_patches, embed_dim]  (1 for cls token)
    Returns:   [1, 1 + new_num_patches, embed_dim]
    """
    cls_token = pos_embed[:, :1, :]       # [1, 1, D]
    patch_pos = pos_embed[:, 1:, :]       # [1, N_old, D]

    old_size = int(patch_pos.shape[1] ** 0.5)
    new_size = target_img_size // patch_size

    if old_size == new_size:
        print(f"pos_embed: {old_size}x{old_size} -> {new_size}x{new_size} (no change)")
        return pos_embed

    D = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, old_size, old_size, D).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(), size=(new_size, new_size),
        mode="bicubic", align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, D)

    print(f"pos_embed: {old_size}x{old_size} -> {new_size}x{new_size}")
    return torch.cat([cls_token, patch_pos], dim=1)


def extract_encoder(checkpoint_path, output_path, target_img_size, patch_size):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    full_state = ckpt["model"]

    encoder_state = {
        k: v for k, v in full_state.items()
        if any(k.startswith(p) for p in ENCODER_PREFIXES)
    }

    print(f"Full checkpoint keys: {len(full_state)}")
    print(f"Encoder-only keys:    {len(encoder_state)}")
    print(f"Dropped keys:         {len(full_state) - len(encoder_state)}")

    encoder_state["pos_embed"] = interpolate_pos_embed(
        encoder_state["pos_embed"], target_img_size, patch_size,
    )

    torch.save(encoder_state, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/checkpoint-300.pth")
    parser.add_argument("--output", default="weights/vit_small_encoder_512.pth")
    parser.add_argument("--target_img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    args = parser.parse_args()
    extract_encoder(args.checkpoint, args.output, args.target_img_size, args.patch_size)
