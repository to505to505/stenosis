"""Domain-Adversarial Neural Network (DANN) variant of ResNet-50.

Implements the classic Ganin & Lempitsky (ICML 2015) recipe:

    - Feature extractor: ImageNet-pretrained torchvision ResNet-50,
      penultimate ``avgpool`` -> 2048-D bottleneck (exposed via
      ``forward_features`` to mirror the MixStyle pattern in
      ``mixstyle_resnet.py`` so feature-extraction code is a drop-in).
    - Label predictor: ``nn.Linear(2048, num_classes)`` (binary
      stenosis-presence head, matching every other domain_gap baseline).
    - Domain classifier: small MLP
      ``Linear(2048, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 1)``,
      attached to the bottleneck via a Gradient Reversal Layer (GRL).

The GRL is a plain ``torch.autograd.Function`` whose forward is identity
and whose backward multiplies the upstream gradient by ``-alpha``.  This
makes the backbone *minimise* label loss while *maximising* domain-
classifier loss, pushing the bottleneck toward domain-invariant
representations without touching the feature extractor's forward path.

The domain head and GRL are inert at eval time: ``forward_features`` is
all that's needed at extract / inference time, exactly like the other
domain_gap scripts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class GradientReversalFn(torch.autograd.Function):
    """Identity in the forward pass; negates and scales gradients in
    backward.

    Args:
        x:     input feature tensor.
        alpha: scalar (or 0-D tensor) controlling the adversarial
               strength.  Backward returns ``-alpha * grad_output``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Functional wrapper around :class:`GradientReversalFn`."""
    return GradientReversalFn.apply(x, alpha)


class DANNResNet50(nn.Module):
    """ResNet-50 backbone + binary label head + adversarial domain head.

    The label head and the domain head share the 2048-D bottleneck.  The
    domain head is fed via :func:`grad_reverse` so the backbone learns
    representations that *fool* the domain classifier.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        domain_hidden: int = 1024,
        domain_dropout: float = 0.5,
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
        # Pool.
        self.avgpool = backbone.avgpool

        in_feat = backbone.fc.in_features  # 2048 for resnet50
        self.feat_dim = in_feat

        # Label predictor (binary stenosis-presence head, matches the
        # other domain_gap baselines).
        self.fc = nn.Linear(in_feat, num_classes)

        # Domain classifier.  Two-hidden-layer MLP, BCEWithLogits.
        self.domain_head = nn.Sequential(
            nn.Linear(in_feat, domain_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden, domain_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 2048)

    def classify(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)

    def discriminate(self, feats: torch.Tensor, alpha: float) -> torch.Tensor:
        feats_rev = grad_reverse(feats, alpha)
        return self.domain_head(feats_rev).squeeze(-1)  # (B,)

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Convenience forward.

        - ``alpha is None`` (eval / extract):  returns ``cls_logits``.
        - ``alpha is not None`` (training):    returns
          ``(cls_logits, domain_logits)`` where ``domain_logits`` is
          computed through the GRL with the supplied ``alpha``.
        """
        feats = self.forward_features(x)
        cls_logits = self.classify(feats)
        if alpha is None:
            return cls_logits
        dom_logits = self.discriminate(feats, alpha)
        return cls_logits, dom_logits


def build_dann_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    domain_hidden: int = 1024,
    domain_dropout: float = 0.5,
) -> DANNResNet50:
    return DANNResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        domain_hidden=domain_hidden,
        domain_dropout=domain_dropout,
    )
