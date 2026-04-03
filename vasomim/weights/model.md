from timm.models.vision_transformer import VisionTransformer

model = VisionTransformer(
    img_size=512, patch_size=16, embed_dim=384, depth=12, num_heads=6,
    mlp_ratio=4., num_classes=YOUR_NUM_CLASSES,
)
state = torch.load("weights/vit_small_encoder_512.pth", map_location="cpu")
model.load_state_dict(state, strict=False)
# missing: head.weight, head.bias — ожидаемо