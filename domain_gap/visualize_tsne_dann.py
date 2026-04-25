"""3-cluster t-SNE / UMAP for the DANN re-evaluation:

    A - Source 2D       (blue)   -- stenosis_arcade
    B - Internal Video  (orange) -- cadica_50plus_new (DANN target)
    C - External Video  (green)  -- dataset2_split    (unseen)

Loads DANN embeddings from ``features/embeddings_dann.npz`` and, when
available, the baseline / MixStyle embeddings to print a side-by-side
cosine-distance + kNN-domain-id comparison.

Saves:
    figures/tsne_dann.{png,pdf}
    figures/tsne_umap_dann.{png,pdf}
    figures/embeddings_2d_dann.npz
    figures/cluster_distances_dann.txt
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
EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_dann.npz'
BASELINE_EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings.npz'
MIXSTYLE_EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_mixstyle.npz'

ORDER = ('A', 'B', 'C')
DATASET_META = {
    'A': ('Dataset A — stenosis_arcade (2D, source)',          '#1f77b4'),
    'B': ('Dataset B — cadica_50plus_new (video, DANN target)', '#ff7f0e'),
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


def _knn_pair(features: np.ndarray, labels: np.ndarray, a: str, b: str) -> float:
    """Binary kNN domain-id between two datasets (chance = 0.5)."""
    keep = np.isin(labels, [a, b])
    f = features[keep]
    y = labels[keep]
    return _knn_domain_id(f, y)


def _quantitative_report(features: np.ndarray, labels: np.ndarray) -> str:
    dist = _centroid_dist(features, labels)
    lines = ['== DANN: centroid cosine distance (lower = closer) ==', '']
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
    lines.append('== Headline (DANN, source A labeled + target B unlabeled) ==')
    lines.append(f'  d(A, B) = {d_ab:.4f}   (target pair -- should collapse)')
    lines.append(f'  d(A, C) = {d_ac:.4f}   (unseen external)')
    lines.append(f'  d(B, C) = {d_bc:.4f}')

    knn3 = _knn_domain_id(features, labels)
    knn_ab = _knn_pair(features, labels, 'A', 'B')
    knn_ac = _knn_pair(features, labels, 'A', 'C')
    knn_bc = _knn_pair(features, labels, 'B', 'C')
    lines.append('')
    lines.append('== 5-fold kNN(k=5,cosine) domain-id accuracy ==')
    lines.append(f'  3-way A/B/C       (chance=0.333): {knn3:.4f}')
    lines.append(f'  binary A vs B     (chance=0.500): {knn_ab:.4f}  '
                 f'(target -- should approach 0.500)')
    lines.append(f'  binary A vs C     (chance=0.500): {knn_ac:.4f}')
    lines.append(f'  binary B vs C     (chance=0.500): {knn_bc:.4f}')

    # Comparison block vs. baseline (no DANN) and MixStyle, when available.
    for name, path in (('baseline (no DANN)', BASELINE_EMB_PATH),
                       ('MixStyle', MIXSTYLE_EMB_PATH)):
        if not path.exists():
            continue
        ref = np.load(path, allow_pickle=True)
        rf = ref['features'].astype(np.float32)
        rl = ref['labels']
        keep = np.isin(rl, list(ORDER))
        rf, rl = rf[keep], rl[keep]
        rdist = _centroid_dist(rf, rl)
        r_ab = float(rdist[ORDER.index('A'), ORDER.index('B')])
        r_ac = float(rdist[ORDER.index('A'), ORDER.index('C')])
        r_bc = float(rdist[ORDER.index('B'), ORDER.index('C')])
        r_knn3 = _knn_domain_id(rf, rl)
        r_knn_ab = _knn_pair(rf, rl, 'A', 'B')
        lines.append('')
        lines.append(f'== Comparison vs. {name} ==')
        lines.append(f'  d(A, B):  {name}={r_ab:.4f}  '
                     f'dann={d_ab:.4f}   delta={d_ab - r_ab:+.4f}')
        lines.append(f'  d(A, C):  {name}={r_ac:.4f}  '
                     f'dann={d_ac:.4f}   delta={d_ac - r_ac:+.4f}')
        lines.append(f'  d(B, C):  {name}={r_bc:.4f}  '
                     f'dann={d_bc:.4f}   delta={d_bc - r_bc:+.4f}')
        lines.append(f'  3-way kNN dom-id:  {name}={r_knn3:.4f}  '
                     f'dann={knn3:.4f}   delta={knn3 - r_knn3:+.4f} '
                     f'(lower = better mixing)')
        lines.append(f'  binary kNN A vs B: {name}={r_knn_ab:.4f}  '
                     f'dann={knn_ab:.4f}   delta={knn_ab - r_knn_ab:+.4f}')

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
             'DANN — Domain Gap (t-SNE, 3 clusters)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'tsne_dann.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'tsne_dann.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DIR / "tsne_dann.png"} / .pdf')

    if emb_umap is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=200)
        _scatter(axes[0], emb_tsne, labels, 't-SNE')
        _scatter(axes[1], emb_umap, labels, 'UMAP')
        fig2.suptitle('DANN — Domain Gap (3 clusters)', fontsize=15)
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / 'tsne_umap_dann.png', dpi=300,
                     bbox_inches='tight')
        fig2.savefig(OUT_DIR / 'tsne_umap_dann.pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f'saved {OUT_DIR / "tsne_umap_dann.png"} / .pdf')

    np.savez_compressed(OUT_DIR / 'embeddings_2d_dann.npz',
                        tsne=emb_tsne,
                        umap=emb_umap if emb_umap is not None
                        else np.zeros((0, 2)),
                        labels=labels)

    report = _quantitative_report(features, labels)
    print('\n' + report)
    (OUT_DIR / 'cluster_distances_dann.txt').write_text(report + '\n')


if __name__ == '__main__':
    main()
