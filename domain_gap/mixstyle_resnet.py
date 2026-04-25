"""ResNet-50 with MixStyle injected after ``layer1`` and ``layer2``.

MixStyle (Zhou et al., ICLR 2021) is a domain-generalisation regulariser
that mixes per-channel style statistics (spatial mean / std) between
samples of a mini-batch during training only. The semantic content of
each sample is preserved (it is normalised, then re-stylised), while the
"style" (low-level appearance, contrast, intensity distribution) becomes
a random interpolation between two real domains.

Placement follows the original paper: applied to *low/mid* level features
(after ``layer1`` and ``layer2``), not after ``layer3``/``layer4`` where
features are too semantic to mix safely.

Critical rule: MixStyle is a no-op when ``self.training`` is False, so
the eval-mode forward pass is bit-identical to a vanilla ResNet-50.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class MixStyle(nn.Module):
    """MixStyle module.

    Args:
        p:     probability of activating MixStyle on a given forward
               (independent per call). Default 0.5.
        alpha: shape parameter of the Beta distribution used to sample
               the mixing weight lambda ~ Beta(alpha, alpha). Default 0.1
               (matches the original paper).
        eps:   numerical stability constant for std.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)

    def extra_repr(self) -> str:
        return f'p={self.p}, alpha={self.alpha}, eps={self.eps}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity at eval time and randomly skipped during training.
        if not self.training:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        if x.size(0) < 2:
            return x

        b = x.size(0)
        # Per-sample, per-channel spatial statistics (gamma = std, beta = mean).
        mu = x.mean(dim=[2, 3], keepdim=True)                      # (B, C, 1, 1)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        # Sample a *per-sample* mixing weight (broadcastable). The paper
        # uses one lam per sample which works better than a scalar.
        lam = self._beta.sample((b, 1, 1, 1)).to(x.device, x.dtype)

        # Shuffle mini-batch -> get the "other" style for each sample.
        perm = torch.randperm(b, device=x.device)
        mu2 = mu[perm]
        sig2 = sig[perm]

        mu_mix = lam * mu + (1.0 - lam) * mu2
        sig_mix = lam * sig + (1.0 - lam) * sig2
        return x_normed * sig_mix + mu_mix


class MixStyleResNet50(nn.Module):
    """torchvision ResNet-50 with MixStyle hooks after layer1 and layer2.

    Penultimate (2048-D) features are exposed via ``forward_features``;
    the classification head is a single ``nn.Linear(2048, num_classes)``.
    Loading ImageNet weights is delegated to ``resnet50(weights=...)``.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Stem.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        # Stages.
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # Head.
        self.avgpool = backbone.avgpool
        in_feat = backbone.fc.in_features
        self.fc = nn.Linear(in_feat, num_classes)

        # Two MixStyle hooks at low/mid levels only.
        self.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.mixstyle1(x)      # inject after layer1
        x = self.layer2(x)
        x = self.mixstyle2(x)      # inject after layer2
        x = self.layer3(x)         # keep semantic features intact
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.fc(feats)


def build_mixstyle_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    mixstyle_p: float = 0.5,
    mixstyle_alpha: float = 0.1,
) -> MixStyleResNet50:
    return MixStyleResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
    )
