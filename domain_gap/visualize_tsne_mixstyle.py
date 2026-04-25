"""3-cluster t-SNE / UMAP for the MixStyle re-evaluation:

    A - Source 2D       (blue)
    B - Internal Video  (orange)
    C - External Video  (green)

Loads MixStyle embeddings from ``features/embeddings_mixstyle.npz`` and,
if available, the baseline embeddings from ``features/embeddings.npz``
to print a side-by-side cosine-distance comparison.

Saves:
    figures/tsne_mixstyle.{png,pdf}
    figures/tsne_umap_mixstyle.{png,pdf}
    figures/embeddings_2d_mixstyle.npz
    figures/cluster_distances_mixstyle.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

ROOT = Path('/home/dsa/stenosis')
OUT_DIR = ROOT / 'domain_gap' / 'figures'
EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_mixstyle.npz'
BASELINE_EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings.npz'

ORDER = ('A', 'B', 'C')
DATASET_META = {
    'A': ('Dataset A — stenosis_arcade (2D, source)',          '#1f77b4'),
    'B': ('Dataset B — cadica_50plus_new (video, unseen)',     '#ff7f0e'),
    'C': ('Dataset C — dataset2_split (video, unseen ext.)',   '#2ca02c'),
}


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


def _quantitative_report(features: np.ndarray, labels: np.ndarray) -> str:
    dist = _centroid_dist(features, labels)
    lines = ['== MixStyle: centroid cosine distance (lower = closer) ==', '']
    header = '       ' + ' '.join(f'{t:>8}' for t in ORDER)
    lines.append(header)
    for i, t in enumerate(ORDER):
        row = ' '.join(f'{dist[i, j]:8.4f}' for j in range(len(ORDER)))
        lines.append(f'{t:>6} {row}')

    def _pair(a, b):
        return float(dist[ORDER.index(a), ORDER.index(b)])

    d_ab = _pair('A', 'B')
    d_ac = _pair('A', 'C')
    d_bc = _pair('B', 'C')
    lines.append('')
    lines.append('== Headline (MixStyle, source-only training) ==')
    lines.append(f'  d(A, B) = {d_ab:.4f}')
    lines.append(f'  d(A, C) = {d_ac:.4f}')
    lines.append(f'  d(B, C) = {d_bc:.4f}')

    # Try to load baseline (no MixStyle) and print the delta on the same
    # 3-class subset.
    if BASELINE_EMB_PATH.exists():
        base = np.load(BASELINE_EMB_PATH, allow_pickle=True)
        bf = base['features'].astype(np.float32)
        bl = base['labels']
        keep = np.isin(bl, list(ORDER))
        bf, bl = bf[keep], bl[keep]
        bdist = _centroid_dist(bf, bl)
        b_ab = float(bdist[ORDER.index('A'), ORDER.index('B')])
        b_ac = float(bdist[ORDER.index('A'), ORDER.index('C')])
        b_bc = float(bdist[ORDER.index('B'), ORDER.index('C')])
        b_knn = _knn_domain_id(bf, bl)
        lines.append('')
        lines.append('== Comparison vs. baseline (no MixStyle) ==')
        lines.append(f'  d(A, B):  baseline={b_ab:.4f}  '
                     f'mixstyle={d_ab:.4f}   '
                     f'delta={d_ab - b_ab:+.4f}')
        lines.append(f'  d(A, C):  baseline={b_ac:.4f}  '
                     f'mixstyle={d_ac:.4f}   '
                     f'delta={d_ac - b_ac:+.4f}')
        lines.append(f'  d(B, C):  baseline={b_bc:.4f}  '
                     f'mixstyle={d_bc:.4f}   '
                     f'delta={d_bc - b_bc:+.4f}')

    knn = _knn_domain_id(features, labels)
    lines.append('')
    lines.append(f'5-fold kNN(k=5,cosine) domain-id acc (MixStyle): '
                 f'{knn:.4f} (chance=0.333)')
    if BASELINE_EMB_PATH.exists():
        lines.append(f'5-fold kNN(k=5,cosine) domain-id acc (baseline): '
                     f'{b_knn:.4f}')
        lines.append(f'  delta = {knn - b_knn:+.4f} '
                     f'(lower = better domain mixing)')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pca-dim', type=int, default=50)
    ap.add_argument('--perplexity', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-umap', action='store_true')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(EMB_PATH, allow_pickle=True)
    features = data['features'].astype(np.float32)
    labels = data['labels']
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f'loaded {features.shape} label counts: {counts}')

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
             'MixStyle Re-evaluation — Domain Gap (t-SNE, 3 clusters)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'tsne_mixstyle.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'tsne_mixstyle.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DIR / "tsne_mixstyle.png"} / .pdf')

    if emb_umap is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=200)
        _scatter(axes[0], emb_tsne, labels, 't-SNE')
        _scatter(axes[1], emb_umap, labels, 'UMAP')
        fig2.suptitle('MixStyle Re-evaluation — Domain Gap (3 clusters)',
                      fontsize=15)
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / 'tsne_umap_mixstyle.png', dpi=300,
                     bbox_inches='tight')
        fig2.savefig(OUT_DIR / 'tsne_umap_mixstyle.pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f'saved {OUT_DIR / "tsne_umap_mixstyle.png"} / .pdf')

    np.savez_compressed(OUT_DIR / 'embeddings_2d_mixstyle.npz',
                        tsne=emb_tsne,
                        umap=emb_umap if emb_umap is not None
                        else np.zeros((0, 2)),
                        labels=labels)

    report = _quantitative_report(features, labels)
    print('\n' + report)
    (OUT_DIR / 'cluster_distances_mixstyle.txt').write_text(report + '\n')


if __name__ == '__main__':
    main()
