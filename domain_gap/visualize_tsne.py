"""2D visualization (t-SNE + optional UMAP) of ResNet-50 embeddings from
three datasets to illustrate the domain gap."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path('/home/dsa/stenosis')
OUT_DIR = ROOT / 'domain_gap' / 'figures'
EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings.npz'

DATASET_META = {
    'A': ('Dataset A — stenosis_arcade (2D)', '#1f77b4'),       # blue
    'B': ('Dataset B — cadica_50plus_new (video)', '#ff7f0e'),  # orange
    'C': ('Dataset C — dataset2_split (video, external)', '#2ca02c'),  # green
    'D': ('Dataset D — CD584 (DICOM video)', '#d62728'),        # red
}


def _scatter(ax, emb2: np.ndarray, labels: np.ndarray, title: str):
    for tag, (name, color) in DATASET_META.items():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pca-dim', type=int, default=50,
                    help='PCA pre-reduction before t-SNE (0 to disable)')
    ap.add_argument('--perplexity', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-umap', action='store_true')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(EMB_PATH, allow_pickle=True)
    features = data['features'].astype(np.float32)
    labels = data['labels']
    print(f'loaded {features.shape} labels counts: '
          f'{dict(zip(*np.unique(labels, return_counts=True)))}')

    x = features
    if args.pca_dim and args.pca_dim < x.shape[1]:
        x = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(x)
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

    # t-SNE only figure (publication-ready)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    _scatter(ax, emb_tsne, labels,
             'Feature Representation Domain Gap across Datasets (t-SNE)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'tsne.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'tsne.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DIR / "tsne.png"} / .pdf')

    if emb_umap is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
        _scatter(axes[0], emb_tsne, labels, 't-SNE')
        _scatter(axes[1], emb_umap, labels, 'UMAP')
        fig2.suptitle('Feature Representation Domain Gap across Datasets',
                      fontsize=15)
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / 'tsne_umap.png', dpi=300, bbox_inches='tight')
        fig2.savefig(OUT_DIR / 'tsne_umap.pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f'saved {OUT_DIR / "tsne_umap.png"} / .pdf')

    # Save embeddings for reproducibility
    np.savez_compressed(OUT_DIR / 'embeddings_2d.npz',
                        tsne=emb_tsne,
                        umap=emb_umap if emb_umap is not None else np.zeros((0, 2)),
                        labels=labels)


if __name__ == '__main__':
    main()
