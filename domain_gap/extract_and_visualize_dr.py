"""Extract 2048-D embeddings from a domain-randomization-trained
ResNet-50 backbone for A/B/C, plus a 3-cluster t-SNE/UMAP plot and
cosine-distance comparison vs. the baseline (and MixStyle if present).

Outputs:
    features/embeddings_dr.npz                (3000 x 2048)
    figures/tsne_dr.{png,pdf}, tsne_umap_dr.{png,pdf}
    figures/embeddings_2d_dr.npz
    figures/cluster_distances_dr.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from domain_gap.extract_features import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PathDataset,
    _collect_dataset_a,
    _collect_dataset_b,
    _collect_dataset_c,
    _sample,
)

ROOT = Path('/home/dsa/stenosis')
OUT_DIR = ROOT / 'domain_gap' / 'figures'
EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_dr.npz'
CKPT = ROOT / 'domain_gap' / 'checkpoints' / 'resnet50_arcade_dr_best.pth'
BASELINE_EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings.npz'
MIXSTYLE_EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_mixstyle.npz'

ORDER = ('A', 'B', 'C')
DATASET_META = {
    'A': ('Dataset A — stenosis_arcade (2D, source, clean)',   '#1f77b4'),
    'B': ('Dataset B — cadica_50plus_new (video, unseen)',     '#ff7f0e'),
    'C': ('Dataset C — dataset2_split (video, unseen ext.)',   '#2ca02c'),
}


def build_feature_extractor(device) -> nn.Module:
    model = resnet50(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, 2)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state['model'])
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


@torch.no_grad()
def embed(model, loader, device) -> np.ndarray:
    feats = []
    for x, _ in tqdm(loader, desc='embed'):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            f = model(x)
        feats.append(f.float().cpu().numpy())
    return (np.concatenate(feats, axis=0) if feats
            else np.zeros((0, 2048), dtype=np.float32))


def _scatter(ax, emb2: np.ndarray, labels: np.ndarray, title: str):
    for tag in ORDER:
        name, color = DATASET_META[tag]
        mask = labels == tag
        ax.scatter(
            emb2[mask, 0], emb2[mask, 1], s=8, alpha=0.55,
            c=color, label=f'{name}  (n={int(mask.sum())})',
            edgecolors='none',
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.35)


def _centroid_dist(features: np.ndarray, labels: np.ndarray,
                   tags=ORDER) -> np.ndarray:
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    centroids = np.stack([f[labels == t].mean(axis=0) for t in tags], axis=0)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    return cosine_distances(centroids)


def _knn_domain_id(features: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(f, labels):
        clf = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
        clf.fit(f[tr], labels[tr])
        accs.append((clf.predict(f[te]) == labels[te]).mean())
    return float(np.mean(accs))


def _report(features: np.ndarray, labels: np.ndarray) -> str:
    dist = _centroid_dist(features, labels)
    lines = ['== Domain-Randomization: centroid cosine distance ==', '']
    header = '       ' + ' '.join(f'{t:>8}' for t in ORDER)
    lines.append(header)
    for i, t in enumerate(ORDER):
        row = ' '.join(f'{dist[i, j]:8.4f}' for j in range(len(ORDER)))
        lines.append(f'{t:>6} {row}')
    d_ab = float(dist[0, 1]); d_ac = float(dist[0, 2]); d_bc = float(dist[1, 2])
    knn = _knn_domain_id(features, labels)
    lines.append('')
    lines.append('== Headline (Domain Randomization, source-only training) ==')
    lines.append(f'  d(A, B) = {d_ab:.4f}')
    lines.append(f'  d(A, C) = {d_ac:.4f}')
    lines.append(f'  d(B, C) = {d_bc:.4f}')
    lines.append(f'  5-fold kNN(k=5,cosine) domain-id acc: {knn:.4f} '
                 f'(chance=0.333)')

    def _comp(name, path):
        if not path.exists():
            return
        ref = np.load(path, allow_pickle=True)
        rf = ref['features'].astype(np.float32)
        rl = ref['labels']
        keep = np.isin(rl, list(ORDER))
        rf, rl = rf[keep], rl[keep]
        rdist = _centroid_dist(rf, rl)
        rab = float(rdist[0, 1]); rac = float(rdist[0, 2]); rbc = float(rdist[1, 2])
        rknn = _knn_domain_id(rf, rl)
        lines.append('')
        lines.append(f'== Comparison vs. {name} ==')
        lines.append(f'  d(A, B): {name}={rab:.4f}  DR={d_ab:.4f}   '
                     f'delta={d_ab - rab:+.4f}')
        lines.append(f'  d(A, C): {name}={rac:.4f}  DR={d_ac:.4f}   '
                     f'delta={d_ac - rac:+.4f}')
        lines.append(f'  d(B, C): {name}={rbc:.4f}  DR={d_bc:.4f}   '
                     f'delta={d_bc - rbc:+.4f}')
        lines.append(f'  kNN domain-id: {name}={rknn:.4f}  DR={knn:.4f}   '
                     f'delta={knn - rknn:+.4f}')

    _comp('baseline', BASELINE_EMB_PATH)
    _comp('mixstyle', MIXSTYLE_EMB_PATH)
    return '\n'.join(lines)


def _do_extract(args, device):
    pool_a = _collect_dataset_a()
    pool_b = _collect_dataset_b(args.stride_b)
    pool_c = _collect_dataset_c(args.stride_c)
    print(f'pool sizes: A={len(pool_a)} B={len(pool_b)} C={len(pool_c)}')

    rng_a = random.Random(args.seed)
    rng_b = random.Random(args.seed + 1)
    rng_c = random.Random(args.seed + 2)
    sample_a = _sample(pool_a, args.n, rng_a)
    sample_b = _sample(pool_b, args.n, rng_b)
    sample_c = _sample(pool_c, args.n, rng_c)
    print(f'sampled:    A={len(sample_a)} B={len(sample_b)} C={len(sample_c)}')

    # Clean (no augmentation) eval transform via torchvision (matches
    # extract_features.py — we want apples-to-apples with the baseline).
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    model = build_feature_extractor(device)

    feats_all, labels_all, paths_all = [], [], []
    for tag, sample in (('A', sample_a), ('B', sample_b), ('C', sample_c)):
        loader = DataLoader(
            PathDataset(sample, tf), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
        f = embed(model, loader, device)
        feats_all.append(f)
        labels_all.extend([tag] * len(sample))
        paths_all.extend(str(p) for p in sample)
        print(f'{tag}: features {f.shape}')

    features = np.concatenate(feats_all, axis=0)
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMB_PATH, features=features,
        labels=np.array(labels_all), paths=np.array(paths_all),
    )
    print(f'saved {EMB_PATH} -> features={features.shape}')
    return features, np.array(labels_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride-b', type=int, default=2)
    ap.add_argument('--stride-c', type=int, default=5)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pca-dim', type=int, default=50)
    ap.add_argument('--perplexity', type=float, default=30.0)
    ap.add_argument('--no-umap', action='store_true')
    ap.add_argument('--reuse', action='store_true',
                    help='Reuse existing embeddings_dr.npz instead of '
                         'recomputing features.')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.reuse and EMB_PATH.exists():
        data = np.load(EMB_PATH, allow_pickle=True)
        features = data['features'].astype(np.float32)
        labels = data['labels']
        print(f'reused {EMB_PATH} -> {features.shape}')
    else:
        features, labels = _do_extract(args, device)

    # Dim-reduce + plot.
    x = features
    if args.pca_dim and args.pca_dim < x.shape[1]:
        x = PCA(n_components=args.pca_dim,
                random_state=args.seed).fit_transform(x)
        print(f'PCA -> {x.shape}')

    print('running t-SNE...')
    tsne = TSNE(
        n_components=2, perplexity=args.perplexity, init='pca',
        learning_rate='auto', random_state=args.seed, max_iter=1500,
    )
    emb_tsne = tsne.fit_transform(x)

    emb_umap = None
    if not args.no_umap:
        try:
            import umap  # noqa: WPS433
            print('running UMAP...')
            reducer = umap.UMAP(n_components=2, random_state=args.seed,
                                n_neighbors=30, min_dist=0.1)
            emb_umap = reducer.fit_transform(x)
        except Exception as exc:  # pragma: no cover
            print(f'UMAP skipped: {exc}')

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    _scatter(ax, emb_tsne, labels,
             'Domain Randomization — Domain Gap (t-SNE, 3 clusters)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'tsne_dr.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'tsne_dr.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DIR / "tsne_dr.png"} / .pdf')

    if emb_umap is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=200)
        _scatter(axes[0], emb_tsne, labels, 't-SNE')
        _scatter(axes[1], emb_umap, labels, 'UMAP')
        fig2.suptitle('Domain Randomization — Domain Gap (3 clusters)',
                      fontsize=15)
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / 'tsne_umap_dr.png', dpi=300,
                     bbox_inches='tight')
        fig2.savefig(OUT_DIR / 'tsne_umap_dr.pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f'saved {OUT_DIR / "tsne_umap_dr.png"} / .pdf')

    np.savez_compressed(OUT_DIR / 'embeddings_2d_dr.npz',
                        tsne=emb_tsne,
                        umap=emb_umap if emb_umap is not None
                        else np.zeros((0, 2)),
                        labels=labels)

    report = _report(features, labels)
    print('\n' + report)
    (OUT_DIR / 'cluster_distances_dr.txt').write_text(report + '\n')


if __name__ == '__main__':
    main()
