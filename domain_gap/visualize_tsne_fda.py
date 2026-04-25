"""4-cluster t-SNE/UMAP for the FDA re-evaluation:

    A   - Original 2D            (blue)
    A*  - FDA-Adapted 2D (->B)   (purple)
    B   - Internal Video         (orange)
    C   - External Video         (green)

Saves:
    domain_gap/figures/tsne_fda.{png,pdf}
    domain_gap/figures/tsne_umap_fda.{png,pdf}
    domain_gap/figures/embeddings_2d_fda.npz
    domain_gap/figures/cluster_distances_fda.txt
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
EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_fda.npz'

# preserve consistent ordering of clusters across the figure / report.
ORDER = ('A', 'A*', 'B', 'C')
DATASET_META = {
    'A':  ('Original 2D — Dataset A (stenosis_arcade)',          '#1f77b4'),
    'A*': ('FDA-Adapted 2D — Dataset A -> B (cadica style)',      '#9467bd'),
    'B':  ('Internal Video — Dataset B (cadica_50plus_new)',     '#ff7f0e'),
    'C':  ('External Video — Dataset C (dataset2_split)',        '#2ca02c'),
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
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.35)


def _quantitative_report(features: np.ndarray, labels: np.ndarray) -> str:
    # L2-normalize for cosine geometry, then compute centroid pairwise
    # cosine distances. Also report mean within-pair sample distance for A*-B
    # vs A-B vs A*-C as the headline numbers for the prompt.
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    centroids = {tag: f[labels == tag].mean(axis=0) for tag in ORDER}
    cm = np.stack([centroids[t] for t in ORDER], axis=0)
    cm = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-12)
    dist = cosine_distances(cm)

    lines = ['== Centroid cosine distance (lower = closer) ==', '']
    header = '       ' + ' '.join(f'{t:>8}' for t in ORDER)
    lines.append(header)
    for i, t in enumerate(ORDER):
        row = ' '.join(f'{dist[i, j]:8.4f}' for j in range(len(ORDER)))
        lines.append(f'{t:>6} {row}')

    def _pair(a, b):
        return float(dist[ORDER.index(a), ORDER.index(b)])

    lines.append('')
    lines.append('== Headline: did FDA pull A toward B? ==')
    lines.append(f'  d(A,  B) = {_pair("A", "B"):.4f}')
    lines.append(f'  d(A*, B) = {_pair("A*", "B"):.4f}   '
                 f'(delta = {_pair("A*", "B") - _pair("A", "B"):+.4f})')
    lines.append('')
    lines.append('== Zero-shot to external video (C) ==')
    lines.append(f'  d(A,  C) = {_pair("A", "C"):.4f}')
    lines.append(f'  d(A*, C) = {_pair("A*", "C"):.4f}   '
                 f'(delta = {_pair("A*", "C") - _pair("A", "C"):+.4f})')
    lines.append('')

    # Domain-id kNN: chance = 0.25.
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(f, labels):
        clf = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
        clf.fit(f[tr], labels[tr])
        accs.append((clf.predict(f[te]) == labels[te]).mean())
    lines.append(f'5-fold kNN(k=5,cosine) domain-id acc: '
                 f'{np.mean(accs):.4f} (chance=0.25)')
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
             'FDA Re-evaluation — Domain Alignment (t-SNE, 4 clusters)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'tsne_fda.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'tsne_fda.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DIR / "tsne_fda.png"} / .pdf')

    if emb_umap is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=200)
        _scatter(axes[0], emb_tsne, labels, 't-SNE')
        _scatter(axes[1], emb_umap, labels, 'UMAP')
        fig2.suptitle('FDA Re-evaluation — Domain Alignment (4 clusters)',
                      fontsize=15)
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / 'tsne_umap_fda.png', dpi=300,
                     bbox_inches='tight')
        fig2.savefig(OUT_DIR / 'tsne_umap_fda.pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f'saved {OUT_DIR / "tsne_umap_fda.png"} / .pdf')

    np.savez_compressed(OUT_DIR / 'embeddings_2d_fda.npz',
                        tsne=emb_tsne,
                        umap=emb_umap if emb_umap is not None
                        else np.zeros((0, 2)),
                        labels=labels)

    report = _quantitative_report(features, labels)
    print('\n' + report)
    (OUT_DIR / 'cluster_distances_fda.txt').write_text(report + '\n')


if __name__ == '__main__':
    main()
