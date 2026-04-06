"""Sequential Quantum Noise Box (SQNB) Generator.

Replaces standard Gaussian diffusion noise with Quantum (Poisson) noise
for bounding box proposals.

Key differences from DiffusionDet:
  - Forward process uses Poisson distribution P(λ) instead of Gaussian N(0,σ²).
  - λ_t = B_{t-1} + γ_t * f_t(B_t) where γ_t follows a noise schedule.
  - Sequence perturbation: random quantum noise only for frame n=0;
    frames n=1..N-1 perturb from frame 0's distribution for consistency.

Training: noise boxes around ground truth.
Inference: noise boxes around image center.
"""

import math

import torch
import torch.nn as nn

from ..config import Config


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for diffusion noise scaling (from DiffusionDet).

    Returns α_bar values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class SQNBGenerator(nn.Module):
    """Sequential Quantum Noise Box generator using Poisson noise.

    Training:
      - Takes GT boxes, adds Poisson noise scaled by the diffusion schedule.
      - Pads with random noise boxes to reach num_proposals.

    Inference:
      - Generates proposals centered at image center with Poisson noise.

    Sequence perturbation:
      - Frame 0 gets fresh Poisson noise.
      - Frames 1..N-1 use frame 0's noise + small perturbation.

    Args:
        cfg: Config with num_proposals, diffusion_steps, snr_scale, img_h, img_w.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_proposals = cfg.num_proposals
        self.diffusion_steps = cfg.diffusion_steps
        self.snr_scale = cfg.snr_scale
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w

        # Pre-compute noise schedule: α_bar from cosine schedule
        alphas_cumprod = cosine_beta_schedule(cfg.diffusion_steps)
        # clip to avoid numerical issues
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-5, max=1.0 - 1e-5)

        # Register buffers (moved to device with model)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    @torch.no_grad()
    def forward_diffusion(
        self,
        gt_boxes: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate noisy proposals around GT boxes for training.

        Args:
            gt_boxes: List of (M_n, 4) GT boxes per frame in xyxy format,
                      concatenated as (sum(M_n), 4). Accompanied by frame indices.
            num_frames: N, number of frames.

        Returns:
            noisy_boxes: (N, num_proposals, 4) in xyxy format, normalized to [0,1].
            timesteps: (N,) sampled diffusion timesteps per frame.
            noise: (N, num_proposals, 4) the noise that was added (for loss).
        """
        device = gt_boxes[0].device if len(gt_boxes) > 0 and len(gt_boxes[0]) > 0 else self.alphas_cumprod.device
        N = num_frames
        P = self.num_proposals
        img_scale = torch.tensor(
            [self.img_w, self.img_h, self.img_w, self.img_h],
            dtype=torch.float32, device=device,
        )

        all_noisy = []
        all_timesteps = []
        all_noise = []
        base_noise = None  # frame 0's noise for sequence perturbation

        for n in range(N):
            # Sample random diffusion timestep
            t = torch.randint(0, self.diffusion_steps, (1,), device=device).item()
            all_timesteps.append(t)

            sqrt_alpha = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

            # Get GT boxes for this frame (xyxy, absolute pixels)
            frame_gt = gt_boxes[n]  # (M, 4) or (0, 4)

            if len(frame_gt) > 0:
                # Normalize GT to [0, 1]
                gt_norm = frame_gt / img_scale.unsqueeze(0)
                # Convert to cxcywh
                gt_cxcywh = box_xyxy_to_cxcywh(gt_norm)

                # Repeat GT to fill proposals (with random selection)
                M = gt_cxcywh.shape[0]
                num_gt_proposals = min(P, M * 4)  # up to 4x GT count
                indices = torch.randint(0, M, (num_gt_proposals,), device=device)
                selected_gt = gt_cxcywh[indices]  # (num_gt_proposals, 4)

                # Generate Poisson noise for GT proposals
                # λ = |signal| * snr_scale (ensure positive for Poisson parameter)
                lam_gt = (selected_gt.abs() * self.snr_scale).clamp(min=0.1)

                if n == 0 or base_noise is None:
                    # Frame 0: fresh Poisson noise
                    poisson_noise_gt = torch.poisson(lam_gt) - lam_gt  # zero-centered
                    # Store base noise pattern for sequence perturbation
                    base_noise = poisson_noise_gt[:min(P, len(poisson_noise_gt))]
                else:
                    # Frames 1..N-1: perturb from frame 0's noise
                    base_subset = base_noise[:num_gt_proposals]
                    if base_subset.shape[0] < num_gt_proposals:
                        # Pad by repeating
                        repeats = (num_gt_proposals // base_subset.shape[0]) + 1
                        base_subset = base_subset.repeat(repeats, 1)[:num_gt_proposals]
                    # Small perturbation using Poisson with small λ
                    perturbation = torch.poisson(torch.ones_like(base_subset) * 0.5) - 0.5
                    poisson_noise_gt = base_subset + perturbation * 0.1

                # Apply noise: x_t = sqrt(α_bar) * x_0 + sqrt(1-α_bar) * noise
                noisy_gt = sqrt_alpha * selected_gt + sqrt_one_minus_alpha * poisson_noise_gt

                # Fill remaining proposals with random noise boxes
                num_random = P - num_gt_proposals
                if num_random > 0:
                    random_boxes = self._random_poisson_boxes(num_random, device)
                    noisy = torch.cat([noisy_gt, random_boxes], dim=0)
                    noise_full = torch.cat([
                        poisson_noise_gt,
                        random_boxes,  # noise for random boxes is the boxes themselves
                    ], dim=0)
                else:
                    noisy = noisy_gt[:P]
                    noise_full = poisson_noise_gt[:P]
            else:
                # No GT: all random proposals
                noisy = self._random_poisson_boxes(P, device)
                noise_full = noisy.clone()

            # Clamp to valid range [0, 1] in cxcywh format
            noisy = noisy.clamp(min=0.0, max=1.0)

            all_noisy.append(noisy)
            all_noise.append(noise_full[:P])

        noisy_boxes = torch.stack(all_noisy, dim=0)  # (N, P, 4) cxcywh normalized
        timesteps = torch.tensor(all_timesteps, device=device, dtype=torch.long)
        noise = torch.stack(all_noise, dim=0)

        # Convert back to xyxy and denormalize
        noisy_xyxy = box_cxcywh_to_xyxy(noisy_boxes)
        noisy_xyxy = noisy_xyxy * img_scale.reshape(1, 1, 4)
        # Clamp to image bounds
        noisy_xyxy[..., 0::2].clamp_(min=0, max=self.img_w)
        noisy_xyxy[..., 1::2].clamp_(min=0, max=self.img_h)

        return noisy_xyxy, timesteps, noise

    @torch.no_grad()
    def generate_proposals(
        self, num_frames: int, device: torch.device
    ) -> torch.Tensor:
        """Generate noise proposals for inference (centered at image center).

        Args:
            num_frames: N, number of frames.
            device: Target device.

        Returns:
            proposals: (N, num_proposals, 4) in xyxy format, absolute pixels.
        """
        P = self.num_proposals
        img_scale = torch.tensor(
            [self.img_w, self.img_h, self.img_w, self.img_h],
            dtype=torch.float32, device=device,
        )

        # Center box in normalized cxcywh: (0.5, 0.5, 1.0, 1.0)
        center = torch.tensor([0.5, 0.5, 1.0, 1.0], device=device)
        center = center.unsqueeze(0).expand(P, -1)  # (P, 4)

        # Poisson noise centered at moderate λ
        lam = center.abs() * self.snr_scale
        lam = lam.clamp(min=0.1)

        all_proposals = []
        base_noise = None

        for n in range(num_frames):
            if n == 0:
                noise = torch.poisson(lam) - lam  # zero-centered
                base_noise = noise
            else:
                perturbation = torch.poisson(torch.ones_like(base_noise) * 0.5) - 0.5
                noise = base_noise + perturbation * 0.1

            proposals_cxcywh = center + noise * 0.5  # scale noise down
            proposals_cxcywh = proposals_cxcywh.clamp(min=0.0, max=1.0)

            proposals_xyxy = box_cxcywh_to_xyxy(proposals_cxcywh)
            proposals_xyxy = proposals_xyxy * img_scale.unsqueeze(0)
            proposals_xyxy[..., 0::2].clamp_(min=0, max=self.img_w)
            proposals_xyxy[..., 1::2].clamp_(min=0, max=self.img_h)

            all_proposals.append(proposals_xyxy)

        return torch.stack(all_proposals, dim=0)  # (N, P, 4)

    def _random_poisson_boxes(
        self, count: int, device: torch.device
    ) -> torch.Tensor:
        """Generate random Poisson-distributed boxes in cxcywh [0,1] space.

        These serve as additional noise proposals to fill up to num_proposals.
        """
        # Use moderate λ for center and size independently
        lam_center = torch.full((count, 2), 5.0, device=device)
        lam_size = torch.full((count, 2), 3.0, device=device)

        # Sample and normalize to [0, 1]
        centers = torch.poisson(lam_center) / (2 * lam_center)  # roughly [0, ~2]
        centers = centers.clamp(0.0, 1.0)

        sizes = torch.poisson(lam_size) / (2 * lam_size)
        sizes = sizes.clamp(0.05, 1.0)  # minimum 5% of image

        return torch.cat([centers, sizes], dim=-1)  # (count, 4)
