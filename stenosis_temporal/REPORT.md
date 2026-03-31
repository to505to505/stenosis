# Spatio-Temporal Stenosis Detector — Implementation Report

## 1. Overview

This project implements a **spatio-temporal coronary artery stenosis detector** inspired by the paper's three-stage architecture: **FPE → PSTFA → MTO**. The model takes a sliding window of T=5 consecutive X-ray angiography frames, extracts per-frame features and proposals, aggregates temporal context via a Transformer, and produces per-frame stenosis detections.

The implementation is fully end-to-end trainable and evaluates using COCO-style mAP@0.5 and mAP@0.5:0.95 metrics.

**Total parameters: ~53.7M**

---

## 2. Architecture

### 2.1 Feature and Proposal Extraction (FPE)

**Files:** `model/fpe.py`

The FPE module processes each frame independently and produces multi-scale features + region proposals.

| Component | Details |
|---|---|
| **Channel Adapter** | `Conv2d(1→3, kernel_size=1)` — converts grayscale input to 3-channel for the pretrained backbone. Weights initialized to `1/3` so the initial output approximates the grayscale value across all 3 channels. |
| **Backbone** | ResNet-50 pretrained on ImageNet (via `torchvision.models.resnet50`). Outputs feature maps at strides 4, 8, 16, 32. |
| **FPN** | Feature Pyramid Network with `C=256` output channels on all levels. Includes `LastLevelMaxPool` producing a 5th "pool" level. |
| **RPN** | Standard Region Proposal Network from torchvision. Generates `S=400` proposals per frame after NMS. Uses 5 anchor sizes `(16, 32, 64, 128, 256)` with 3 aspect ratios `(0.5, 1.0, 2.0)` per FPN level. |
| **RoI Align** | `MultiScaleRoIAlign` on FPN levels 0–3, output size `7×7`, sampling ratio 2. |

**Input:** `(B×T, 1, 512, 512)` flattened grayscale frames  
**Output:** FPN feature maps (OrderedDict), list of S=400 proposals per frame, RPN losses (training)

**Proposal padding/truncation:** RPN may return variable numbers of proposals. We pad (by repeating the last proposal) or truncate to ensure exactly S=400 proposals per frame for consistent tensor shapes downstream.

### 2.2 Proposal-aware Spatio-Temporal Feature Aggregation (PSTFA)

**Files:** `model/pstfa.py`

The PSTFA module is the core temporal reasoning component. It consists of two sub-modules: **PSSTT** (tokenization) and **TFA** (Transformer aggregation).

#### 2.2.1 PSSTT — Proposal-Shifted Spatio-Temporal Tokenization

For each proposal in the reference frame, PSSTT generates `T×(K+1) = 5×5 = 25` RoI tokens:

1. **Spatial shifts (K=4):** Each proposal box is shifted in 4 directions (up, down, left, right) by `shift_fraction=0.5` of its own width/height. Together with the original box, this gives `K+1=5` spatially variant boxes.

2. **Temporal replication:** These 5 shifted boxes are applied via RoI Align to **all T=5 frames** of the FPN level-0 feature map (stride 4, spatial scale 1/4).

3. **Token order:** The reference frame is processed first, followed by support frames in temporal order.

4. **Projection:** Each RoI feature `(C×7×7 = 12,544-dim)` is linearly projected to `D=512` dimensions.

**Output:** `(S, 25, 512)` token sequence per batch element.

#### 2.2.2 TFA — Transformer-based Feature Aggregation

| Component | Details |
|---|---|
| **Positional Embedding** | Learned `(1, 25, 512)` parameter, initialized with truncated normal (σ=0.02) |
| **Transformer Encoder** | 4 layers, `d_model=512`, `nhead=8`, `dim_feedforward=2048`, GELU activation, **pre-norm** (`norm_first=True`), dropout=0.1 |
| **Feature Fusion** | 1×1 Conv1d over token dimension → global average pooling → LayerNorm |
| **Output Projection** | Linear `512 → 12,544` → reshape to `(S, 256, 7, 7)` |

**Input:** `(S, 25, 512)` tokens  
**Output:** `(S, 256, 7, 7)` aggregated RoI features

### 2.3 Multi-Task Outputs (MTO)

**Files:** `model/mto.py`

Two parallel heads operating on the aggregated RoI features:

| Head | Architecture | Output |
|---|---|---|
| **Classification** | AdaptiveAvgPool2d(1) → FC(256→1024) → ReLU → FC(1024→2) | `(S, 2)` logits (background + stenosis) |
| **Regression** | AdaptiveAvgPool2d(1) → FC(256→1024) → ReLU → FC(1024→4) | `(S, 4)` box deltas (dx, dy, dw, dh) |

Both heads share the same pooling operation but have independent weights.

### 2.4 End-to-End Detector

**Files:** `model/detector.py`

The `StenosisTemporalDetector` combines all modules and handles:

**Training forward pass:**
1. Reshape `(B, T, 1, H, W)` → `(B×T, 1, H, W)`, run FPE
2. For each batch element, for each of the T=5 frames as reference: run PSTFA → MTO
3. Assign GT to proposals via IoU matching (`fg_iou_thresh=0.5`)
4. Compute losses: RPN objectness (CE) + RPN box (SL1) + detection classification (CE) + detection regression (Smooth L1, foreground only)
5. Average detection losses over `B×T` reference frames

**Inference forward pass:**
1. Same FPE + PSTFA + MTO pipeline
2. Decode box deltas to absolute coordinates using Faster R-CNN encoding
3. Score filtering (`score_thresh=0.05`) → NMS (`nms_thresh=0.5`) → top-100 per frame

**Box encoding:** Standard Faster R-CNN delta encoding:
- `dx = (gx - px) / pw`, `dy = (gy - py) / ph`
- `dw = log(gw / pw)`, `dh = log(gh / ph)`
- Decoding clamps `dw, dh ≤ 4.0` to prevent exponential overflow

---

## 3. Dataset

**Files:** `dataset.py`

### 3.1 Data Source

Uses `dataset2_split` — a patient-level split of coronary angiography images:
- **64 patients**, split 44 train / 10 valid / 10 test
- **8,323 total images**, 512×512 grayscale
- **Single class:** Stenosis
- **YOLO format** labels (normalized center x, y, w, h)

### 3.2 Filename Convention

Pattern: `{patient_id}_{sequence_id}_{frame_number}_bmp_jpg.rf.{uuid}.jpg`

Example: `14_021_1_0046_bmp_jpg.rf.abc123.jpg`
- Patient: `14_021`
- Sequence: `1`
- Frame: `0046`

### 3.3 Sequence Construction

1. **Group** images by `(patient_id, sequence_id)` → 151 sequences
2. **Sort** frames within each sequence by frame number
3. **Sliding window** of `T=5` consecutive frames
   - Sequences shorter than T are padded by repeating the last frame
   - Produces ~5,213 training windows

### 3.4 Preprocessing

| Step | Details |
|---|---|
| Load | `cv2.IMREAD_GRAYSCALE` |
| Resize | From original (640×640 from Roboflow) to 512×512 if needed |
| Normalize | `(pixel - 103.53) / 57.12` |
| Format | `torch.float32`, shape `(T, 1, H, W)` |

Labels are converted from YOLO normalized format to absolute `x1y1x2y2` at load time.

---

## 4. Training

**Files:** `train.py`

### 4.1 Optimizer and Schedule

| Parameter | Value |
|---|---|
| Optimizer | SGD, momentum=0.9, weight_decay=1e-4 |
| Base LR | 0.02 |
| Warmup | Linear warmup over 500 iterations |
| LR Schedule | Multi-step decay at epochs 60 and 80, γ=0.1 |
| Epochs | 100 |
| Batch size | 2 |

The scheduler operates in **iteration space** (not epoch space), with milestones converted as `epoch × iters_per_epoch`.

### 4.2 Mixed Precision

- Uses `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`
- All forward passes run in FP16; losses and optimizer step in FP32

### 4.3 Gradient Accumulation

Configurable via `grad_accum_steps` (default 1 = disabled). When enabled:
- Loss is divided by accumulation steps
- `optimizer.step()` only every `accum` mini-batches
- Handles leftover batches at epoch end

### 4.4 Weight Initialization

- **Backbone (ResNet-50):** Pretrained ImageNet weights, untouched
- **All other layers:** Xavier uniform initialization
- **Biases:** Zeros
- **Positional embeddings:** Truncated normal (σ=0.02)

### 4.5 Checkpointing

| File | When saved | Contents |
|---|---|---|
| `runs/train/last.pt` | Every epoch | Full state (model, optimizer, scheduler, scaler, epoch, global_step, best_map) |
| `runs/train/best.pt` | When mAP@0.5:0.95 improves | Model weights, epoch, AP@0.5, AP@0.5:0.95 |

TensorBoard logs are written to `runs/train/tb_logs/`.

---

## 5. Evaluation

**Files:** `evaluate.py`

### 5.1 Metrics

- **mAP@0.5:** Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95:** Mean AP averaged over IoU thresholds [0.5, 0.55, ..., 0.95] — COCO-style primary metric

### 5.2 AP Computation

1. All detections across the validation set are collected with scores
2. Sorted by confidence descending
3. Greedy matching to ground truth boxes at each IoU threshold
4. Precision-recall curve computed, AP via all-point interpolation
5. Best model checkpoint is selected by mAP@0.5:0.95

---

## 6. Differences from the Original Paper

| Aspect | Original Paper | This Implementation | Rationale |
|---|---|---|---|
| **Input resolution** | Not specified | 512×512 grayscale | Matches dataset native resolution |
| **Classes** | Multi-class stenosis grading | Single binary class (stenosis vs background) | Dataset has only one annotation class |
| **Dataset** | CADICA or similar multi-center | `dataset2_split` — single-center Russian angiography dataset | Available data |
| **Backbone initialization** | COCO-pretrained Faster R-CNN | ImageNet-pretrained ResNet-50 | COCO detection weights not readily available as standalone backbone; ImageNet provides equivalent feature quality |
| **Channel adapter** | Not needed (RGB input) | 1→3 Conv2d adapter for grayscale | Grayscale angiography images require channel conversion |
| **RPN implementation** | Custom | torchvision's `RegionProposalNetwork` | Proven, well-tested implementation with identical functionality |
| **FPN level for PSTFA** | Not specified | Level 0 only (stride 4, spatial_scale=1/4) | Highest resolution for small stenosis lesions |
| **TFA output fusion** | Described as "fusion" | 1×1 Conv1d → mean pool → LayerNorm → linear projection | Reasonable interpretation of the paper's fusion step |
| **Transformer variant** | Standard encoder | Pre-norm (`norm_first=True`) with GELU | Pre-norm is more stable for training |
| **Box encoding** | Not specified | Standard Faster R-CNN delta encoding with `dw/dh` clamping at 4.0 | Prevents numerical instability |
| **LR schedule** | Not specified | Linear warmup (500 iter) + multi-step decay at 60/80 epochs | Standard detection training schedule |
| **Data augmentation** | Not specified | None (only resize + normalize) | Could be added for improved generalization |

---

## 7. Engineering Decisions

### 7.1 Memory Optimization (Available but Currently Disabled)

The implementation includes several configurable memory optimizations for GPU-constrained environments:

| Feature | Config flag | Status | Effect |
|---|---|---|---|
| **Gradient checkpointing (backbone)** | `gradient_checkpointing` | `False` | Saves ~2-3 GB by recomputing ResNet layer activations during backward |
| **Gradient checkpointing (transformer)** | Same flag | `False` | Saves ~1-2 GB by recomputing Transformer layer activations |
| **Proposal chunking** | `proposal_chunk_size` | `400` (= S, no chunking) | Can be reduced to process proposals in smaller batches through PSTFA |
| **Gradient accumulation** | `grad_accum_steps` | `1` (disabled) | Allows effective larger batch size with less peak VRAM |
| **Early tensor deletion** | Always on | Active | `del flat_images, aggregated_roi` after use |

Current configuration is optimized for **speed** on a 12 GB GPU. For tighter VRAM budgets, enable gradient checkpointing and reduce chunk size.

### 7.2 Proposal Processing

- RPN proposals are padded/truncated to exactly S=400 per frame to maintain fixed tensor shapes
- Padding replicates the last proposal (has near-zero effect since duplicates get suppressed by NMS at inference)

### 7.3 Training Labels

- Single class dataset: all annotations are mapped to label `0` (stenosis) internally, with `num_classes=2` (background=0 + stenosis=1) for the classification head
- Proposal-to-GT assignment uses simple max-IoU matching with `fg_iou_thresh=0.5`
- Regression loss is computed only on foreground (IoU ≥ 0.5) proposals

---

## 8. Project Structure

```
stenosis_temporal/
├── __init__.py
├── config.py              # All hyperparameters (dataclass)
├── dataset.py             # Sequence builder, temporal dataset, collate
├── train.py               # Training loop with warmup, AMP, TensorBoard
├── evaluate.py            # mAP@0.5 and mAP@0.5:0.95 evaluation
├── model/
│   ├── __init__.py        # Exports StenosisTemporalDetector
│   ├── fpe.py             # ResNet50 + FPN + RPN + channel adapter
│   ├── pstfa.py           # PSSTT tokenization + TFA Transformer
│   ├── mto.py             # Classification + regression heads
│   └── detector.py        # End-to-end detector (FPE → PSTFA → MTO)
├── tests/
│   ├── __init__.py
│   └── test_all.py        # 50 comprehensive tests
└── runs/                  # Training outputs (checkpoints, TensorBoard)
```

---

## 9. Test Suite

**50 tests** covering all modules:

| Category | Count | What is tested |
|---|---|---|
| Config | 2 | Default values, `num_tokens` property |
| Dataset | 10 | Filename parsing, window building, label loading, normalization, collation |
| Box encoding | 2 | Encode/decode roundtrip, identity transform |
| FPE | 4 | Backbone output shapes, FPE shapes, eval mode, channel adapter |
| PSSTT | 5 | Shifted box shapes, original-first ordering, shift directions, clamping, output shapes |
| TFA | 2 | Output shapes, positional embedding shape |
| PSTFA | 2 | Output shapes, chunked processing equivalence |
| MTO | 2 | Output shapes, single-input handling |
| Detector | 7 | Training losses, loss summation, backward pass, empty GT, inference shapes, clamping, score filtering |
| Init/Assignment | 4 | Weight initialization, backbone preservation, IoU assignment, empty GT assignment |
| Integration | 1 | Forward pass on real dataset images |

---

## 10. Usage

### Training
```bash
cd stenosis_temporal
python train.py --device 0
```

### Resume from checkpoint
```bash
python train.py --device 0 --resume runs/train/last.pt
```

### Evaluation
```bash
python evaluate.py --checkpoint runs/train/best.pt --split test --device 0
```

### Running tests
```bash
python -m pytest tests/test_all.py -v
```
