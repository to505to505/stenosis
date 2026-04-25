# Domain Gap Analysis — ResNet-50 Baseline (4 datasets)

## Setup
- **Backbone:** ResNet-50 (ImageNet `IMAGENET1K_V2` init), `fc → Linear(2)`.
- **Training data:** Dataset A only (`data/stenosis_arcade`, 1000 train / 200 val) — binary label (has stenosis box vs. not).
- **Augmentations:** Resize 512×512, H-flip, color jitter, small affine, ImageNet normalization.
- **Schedule:** AdamW lr=1e-4, cosine, 12 epochs, AMP.
- **Best val accuracy:** **0.980**. Saved to `domain_gap/checkpoints/resnet50_arcade_best.pth`.

## Feature extraction
- Loaded best weights, replaced `fc` with `nn.Identity()` → 2048-D embedding per image.
- Eval-only preprocessing: Resize 512×512 + ImageNet normalization, `requires_grad=False`.
- **Per-sequence decimation** (keep every N-th frame) before random sampling of **1000 frames per dataset**:

| Dataset | Source | Format | Stride | Pool | Sampled |
|---|---|---|---|---|---|
| A | `data/stenosis_arcade` (train/val/test) | static PNG | — | 1500 | 1000 |
| B | `data/cadica_50plus_new` (sequences `pXX_vYY_*`) | PNG video frames | 2 | 1148 | 1000 |
| C | `data/dataset2_split` (grouped by `<study>_<video>_<clip>`) | JPG video frames | 5 | 1795 | 1000 |
| D | `/home/dsa/CD584/**/*.dcm` (72 multi-frame DICOM series) | DICOM (MONOCHROME2, JPEG-Lossless) | 5 | 1057 | 1000 |

Embeddings saved to `domain_gap/features/embeddings.npz` — **4000 × 2048** features + dataset labels + paths/frame-ids.

### Notes for CD584 (Dataset D)
- DICOMs are multi-frame JPEG-Lossless (Process 14). Required `pylibjpeg` + `pylibjpeg-libjpeg` (explicit `import libjpeg` to register the decoder with pydicom in this env).
- Several Rotterdamfastextent files lacked tag (0028,0004) `PhotometricInterpretation` — filled with `MONOCHROME2` fallback before decoding. All 72 DICOM series loaded successfully after the fix.
- Each DICOM is an 8-bit angiographic cine of shape (F, 512, 512). Per-frame min-max normalized to `uint8`, decimated stride=5, then 1000 random frames sampled across series.

## Dimensionality reduction
- PCA 2048 → 50, then t-SNE (perplexity 30, 1500 iter, `random_state=42`).
- Secondary UMAP (n_neighbors=30, min_dist=0.1) for sanity check.
- Plots: `domain_gap/figures/tsne.{png,pdf}`, `domain_gap/figures/tsne_umap.{png,pdf}`. 2D coords cached in `domain_gap/figures/embeddings_2d.npz`.

## Quantitative domain-gap diagnostics
L2-normalized 2048-D features.

**5-fold kNN (k=5) domain-identification accuracy: 0.993** (chance = 0.25)

Centroid cosine distances:

| Pair | cos-dist |
|---|---|
| A ↔ B | 0.185 |
| A ↔ C | 0.080 |
| A ↔ D | 0.159 |
| B ↔ C | 0.169 |
| B ↔ D | **0.240** |
| C ↔ D | 0.088 |

## Brief analysis
The t-SNE plot shows **four largely disjoint clusters**: A (blue, stenosis_arcade static 2D) in the lower half, B (orange, cadica) on the upper left, C (green, dataset2) in the center, and D (red, CD584 DICOM cines) forms many small tight sub-clusters on the right — one per DICOM series, consistent with per-acquisition appearance cues (beam geometry, contrast phase, patient anatomy). A 5-fold kNN classifier recovers the dataset-of-origin label with **99.3% accuracy** versus a 25% chance baseline, confirming a severe domain shift.

- **B ↔ D is the largest gap (cos-dist 0.240)** — cadica (arcade-registered preprocessed PNG frames) looks very different from raw CD584 DICOM cines.
- **A ↔ C (0.080)** and **C ↔ D (0.088)** are the smallest gaps — dataset2 sits between the static 2D arcade domain and the raw clinical DICOM domain.
- D forms **many small sub-clusters** rather than one blob: each DICOM series behaves like its own mini-domain, which must be accounted for in any domain-adaptation / test-time adaptation scheme (e.g. per-sequence batch-norm statistics, prompt-based DA).

This reinforces the earlier conclusion: "video" is **not** a single target domain. Any successful adaptation (UDA, self-training, style transfer, TTA) should be measured by (1) the kNN domain-id accuracy falling toward 0.25, and (2) overlap between all four clusters in t-SNE/UMAP.

## Files
- `domain_gap/train.py` — ResNet-50 training on Dataset A only.
- `domain_gap/extract_features.py` — A/B/C (PNG/JPG) sampling + embedding.
- `domain_gap/extract_features_cd584.py` — CD584 DICOM decoding + decimation + embedding, appended to `embeddings.npz`.
- `domain_gap/visualize_tsne.py` — PCA → t-SNE (+UMAP) plot (4-class aware).
- `domain_gap/checkpoints/resnet50_arcade_best.pth`
- `domain_gap/features/embeddings.npz` — 4000 × 2048 features + labels + ids.
- `domain_gap/figures/tsne.{png,pdf}`, `tsne_umap.{png,pdf}`, `embeddings_2d.npz`
- Logs: `domain_gap/train.log`, `extract.log`, `extract_d.log`, `viz.log`.

## Reproduce
```bash
python domain_gap/train.py --epochs 12 --batch-size 16 --num-workers 6
python domain_gap/extract_features.py --stride-b 2 --stride-c 5 --num-workers 6
python domain_gap/extract_features_cd584.py --stride 5 --num-workers 6
python domain_gap/visualize_tsne.py
```

---

# FDA Re-evaluation — A → B alignment, 4-cluster t-SNE

Goal: apply Fourier Domain Adaptation (Yang & Soatto, CVPR 2020) to push
the static 2D source (`stenosis_arcade`, **A**) toward the internal video
domain (`cadica_50plus_new`, **B**), then check whether the alignment
also helps zero-shot generalisation to the **unseen** external video
domain (`dataset2_split`, **C**). Dataset C is **never** read during
adaptation or training — only at the embedding/eval step.

## Pipeline

1. **`fda_transform.py`** — FFT-based low-frequency amplitude swap.
   For every source image `x_s` we sample a random target `x_t` from
   Dataset B, FFT both, replace the centred low-frequency window of
   `|F(x_s)|` (half-width `b = floor(min(H,W) * beta)`, default
   `beta=0.01`) with `|F(x_t)|`, keep the source phase, inverse-FFT.
   Output: `data/stenosis_arcade_fda_B/{train,val,test}/{images,labels}`
   (1500 PNGs at 512×512, labels copied verbatim).
2. **`train_fda.py`** — retrain ResNet-50 from ImageNet on the FDA-aligned
   training split (binary stenosis-presence). 12 epochs, AdamW 1e-4,
   AMP, cosine LR. Best `val_acc = 0.980` at epoch 1, checkpoint
   `checkpoints/resnet50_arcade_fda_best.pth`.
3. **`extract_features_fda.py`** — freeze the FDA-trained backbone,
   replace `fc → Identity`, embed 1000 frames from each of A, A*, B, C
   (same stride/sampling rules as the baseline). Output:
   `features/embeddings_fda.npz` — **4000 × 2048**.
4. **`visualize_tsne_fda.py`** — PCA(50) → t-SNE (perplexity 30,
   1500 iter) and UMAP. Centroid cosine distance matrix and 5-fold
   kNN(k=5) domain-id accuracy reported in
   `figures/cluster_distances_fda.txt`.
5. **`compute_fid.py`** — canonical FID with Inception-V3 pool3 on the
   exact same 4×1000 image populations. Saved to `figures/fid_fda.txt`.

## Results

### Centroid cosine distance (lower = closer)

|        |    A   |   A\*  |    B   |    C   |
|:------:|:------:|:------:|:------:|:------:|
| **A**  | 0.0000 | 0.0089 | 0.2295 | 0.0770 |
| **A\***| 0.0089 | 0.0000 | **0.2105** | 0.0796 |
| **B**  | 0.2295 | 0.2105 | 0.0000 | 0.1424 |
| **C**  | 0.0770 | 0.0796 | 0.1424 | 0.0000 |

- `d(A, B) = 0.2295  →  d(A*, B) = 0.2105` (Δ = **−0.019**) — FDA mildly
  pulls the source toward the internal-video target on the
  domain-classifier feature space.
- `d(A, C) = 0.0770  →  d(A*, C) = 0.0796` (Δ = +0.003) — essentially
  unchanged; the alignment does **not** generalise to the external
  domain on this metric (expected, since C was never used).
- 5-fold kNN domain-ID accuracy drops from 0.993 (4-dataset baseline) to
  **0.899** with A* added — A and A* are nearly indistinguishable to
  the kNN, which inflates confusion within the source pair.

### FID (Inception-V3 pool3, lower = closer)

| pair | FID |
|---|---|
| FID(A,  B) | 47.10 |
| FID(A\*, B) | 52.15 |
| FID(A,  C) | 40.54 |
| FID(A\*, C) | 44.57 |
| FID(B,  C) | 37.79 |
| FID(A,  A\*) | 17.30 |

ΔFID(·, B) = **+5.06**, ΔFID(·, C) = **+4.03**. On *raw* Inception
features the small low-frequency swap (β = 0.01) introduces ringing /
halo texture that ImageNet-trained Inception reads as a new style
component, so FID slightly *increases*. The two metrics tell
complementary stories: FDA shifts the **task-relevant** representation
(domain-classifier features) toward B, but does not produce
photometrically B-like images at the pixel-statistics level.

### Visual summary

`figures/tsne_fda.png` shows A (blue) and A\* (purple) overlapping
almost entirely — confirming the visual evidence for the small cosine
shift — while B (orange) and C (green) remain in their own clusters.
A second sub-blob of A/A\* / B mixes at the bottom-right, suggesting a
sub-population of source frames that *does* project into the B manifold
after FDA.

### Files added

- `fda_transform.py`, `train_fda.py`, `extract_features_fda.py`
- `visualize_tsne_fda.py`, `compute_fid.py`
- `data/stenosis_arcade_fda_B/` — 1500 FDA-adapted PNGs + copied labels
- `checkpoints/resnet50_arcade_fda_best.pth`
- `features/embeddings_fda.npz` — 4000 × 2048
- `figures/tsne_fda.{png,pdf}`, `figures/tsne_umap_fda.{png,pdf}`
- `figures/embeddings_2d_fda.npz`
- `figures/cluster_distances_fda.txt`, `figures/fid_fda.txt`

### Reproduce

```bash
python -m domain_gap.fda_transform --beta 0.01
python -m domain_gap.train_fda --epochs 12 --batch-size 16 --num-workers 6
python -m domain_gap.extract_features_fda --stride-b 2 --stride-c 5 --num-workers 6
python -m domain_gap.visualize_tsne_fda
python -m domain_gap.compute_fid
```
