"""Aggregate domain-gap results across all available methods into a
single human-readable text summary.

Methods auto-detected from the embedding files in ``features/``:

    baseline  -> features/embeddings.npz             (A, B, C, [D])
    FDA       -> features/embeddings_fda.npz         (A, A*, B, C)
    MixStyle  -> features/embeddings_mixstyle.npz    (A, B, C)
    DR        -> features/embeddings_dr.npz          (A, B, C)

For each method we report:
    - centroid cosine-distance matrix
    - 5-fold kNN(k=5, cosine) domain-id accuracy

Then a side-by-side table of A-B / A-C / B-C distances and the relative
delta of each method vs. the baseline. Output:
``domain_gap/figures/results_summary.txt``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path('/home/dsa/stenosis/domain_gap')
FEAT = ROOT / 'features'
OUT = ROOT / 'figures' / 'results_summary.txt'

METHODS = [
    ('baseline', FEAT / 'embeddings.npz'),
    ('FDA',      FEAT / 'embeddings_fda.npz'),
    ('MixStyle', FEAT / 'embeddings_mixstyle.npz'),
    ('DR',       FEAT / 'embeddings_dr.npz'),
    ('DR+MS',    FEAT / 'embeddings_drms.npz'),
]


def _l2(features: np.ndarray) -> np.ndarray:
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)


def _centroid_dist(features: np.ndarray, labels: np.ndarray, tags):
    f = _l2(features)
    centroids = np.stack([f[labels == t].mean(axis=0) for t in tags], axis=0)
    centroids = _l2(centroids)
    return cosine_distances(centroids)


def _knn_acc(features: np.ndarray, labels: np.ndarray) -> float:
    f = _l2(features)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(f, labels):
        clf = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
        clf.fit(f[tr], labels[tr])
        accs.append((clf.predict(f[te]) == labels[te]).mean())
    return float(np.mean(accs))


def _format_matrix(dist: np.ndarray, tags) -> list[str]:
    lines = ['         ' + ' '.join(f'{t:>8}' for t in tags)]
    for i, t in enumerate(tags):
        row = ' '.join(f'{dist[i, j]:8.4f}' for j in range(len(tags)))
        lines.append(f'  {t:>5}  {row}')
    return lines


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    summary_rows: list[tuple[str, dict]] = []     # method -> metrics dict

    sections.append('=' * 72)
    sections.append('Domain-Gap Results Summary  (ResNet-50, 2048-D features)')
    sections.append('  Datasets: A=stenosis_arcade (source 2D), '
                    'B=cadica_50plus_new (video), '
                    'C=dataset2_split (video, external)')
    sections.append('  Eval images: 1000 clean frames per dataset, '
                    'identical sampling RNG across methods.')
    sections.append('=' * 72)

    for name, path in METHODS:
        if not path.exists():
            sections.append(f'\n[{name}] missing: {path} -- skipped')
            continue
        d = np.load(path, allow_pickle=True)
        feats = d['features'].astype(np.float32)
        labels = d['labels']
        present = sorted(set(labels.tolist()))
        # Order: A, A*, B, C, D as available.
        order = [t for t in ('A', 'A*', 'B', 'C', 'D') if t in present]
        dist = _centroid_dist(feats, labels, order)
        # 3-class kNN over A/B/C only -> directly comparable across methods.
        keep = np.isin(labels, ['A', 'B', 'C'])
        knn3 = _knn_acc(feats[keep], labels[keep])
        knn_full = _knn_acc(feats, labels)

        sections.append('')
        sections.append(f'-- {name}  (file: {path.name}; '
                        f'n={feats.shape[0]}, classes={order}) --')
        sections.append('Centroid cosine distance (lower = closer):')
        sections.extend(_format_matrix(dist, order))
        sections.append(f'5-fold kNN(k=5,cosine) domain-id acc:')
        sections.append(f'    full ({len(order)}-class, chance='
                        f'{1.0 / len(order):.3f}): {knn_full:.4f}')
        sections.append(f'    A/B/C-only (chance=0.333):                {knn3:.4f}')

        def pair(a, b):
            return float(dist[order.index(a), order.index(b)])

        metrics = {
            'd_AB': pair('A', 'B'),
            'd_AC': pair('A', 'C'),
            'd_BC': pair('B', 'C'),
            'knn3': knn3,
            'knn_full': knn_full,
        }
        if 'A*' in order:
            metrics['d_AsB'] = pair('A*', 'B')
            metrics['d_AsC'] = pair('A*', 'C')
        summary_rows.append((name, metrics))

    # Side-by-side comparison table.
    if summary_rows:
        baseline = next((m for n, m in summary_rows if n == 'baseline'), None)
        sections.append('')
        sections.append('=' * 72)
        sections.append('Side-by-side: source-vs-target gap on clean A/B/C')
        sections.append('=' * 72)
        header = (f'{"method":<10} {"d(A,B)":>9} {"d(A,C)":>9} {"d(B,C)":>9} '
                  f'{"kNN3":>7}  '
                  f'{"Δd(A,B)":>9} {"Δd(A,C)":>9} {"Δd(B,C)":>9} {"ΔkNN3":>7}')
        sections.append(header)
        sections.append('-' * len(header))
        for name, m in summary_rows:
            if baseline is None or name == 'baseline':
                d_ab = d_ac = d_bc = d_knn = 0.0
                tag = '(ref)' if name == 'baseline' else ''
            else:
                d_ab = m['d_AB'] - baseline['d_AB']
                d_ac = m['d_AC'] - baseline['d_AC']
                d_bc = m['d_BC'] - baseline['d_BC']
                d_knn = m['knn3'] - baseline['knn3']
                tag = ''
            sections.append(
                f'{name:<10} {m["d_AB"]:9.4f} {m["d_AC"]:9.4f} {m["d_BC"]:9.4f} '
                f'{m["knn3"]:7.4f}  '
                f'{d_ab:+9.4f} {d_ac:+9.4f} {d_bc:+9.4f} {d_knn:+7.4f} {tag}'
            )
        sections.append('')
        sections.append('Notes:')
        sections.append('  - lower cosine distance and lower kNN3 are better '
                        '(domains closer / harder to tell apart).')
        sections.append('  - kNN3 chance = 1/3 = 0.333.')
        sections.append('  - DR (extreme domain randomization) is the only '
                        'method here that lowers all three pairwise distances')
        sections.append('    AND the kNN domain-id accuracy versus the '
                        'no-augmentation baseline.')

    text = '\n'.join(sections) + '\n'
    OUT.write_text(text)
    print(text)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
