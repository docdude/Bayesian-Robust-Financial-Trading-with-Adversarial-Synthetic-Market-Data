"""PCA / t-SNE visualisation for real vs synthetic time-series sequences.

Framework-agnostic — requires only numpy, sklearn, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualization(ori_data, generated_data, analysis, n_samples=1000,
                  save_path=None, feature_indices=None):
    """PCA or t-SNE scatter plot comparing real and synthetic windows.

    Args:
        ori_data:       (N, seq_len, dim) real sequences
        generated_data: (M, seq_len, dim) synthetic sequences
        analysis:       'pca' or 'tsne'
        n_samples:      max points to plot (for speed)
        save_path:      if given, save figure to this path instead of plt.show()
        feature_indices: optional list of channel indices to average over.
                         If None, averages all channels.  For multi-stock data
                         pass close-return indices (e.g. range(0, dim, 5)) to
                         avoid mixing heterogeneous feature types.
    """
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    n = min(n_samples, len(ori_data), len(generated_data))
    idx_o = np.random.permutation(len(ori_data))[:n]
    idx_g = np.random.permutation(len(generated_data))[:n]

    # Collapse feature dim → mean across features per timestep → (n, seq_len)
    if ori_data.ndim == 3:
        if feature_indices is not None:
            prep_real = np.mean(ori_data[idx_o][:, :, feature_indices], axis=2)
            prep_fake = np.mean(generated_data[idx_g][:, :, feature_indices], axis=2)
        else:
            prep_real = np.mean(ori_data[idx_o], axis=2)
            prep_fake = np.mean(generated_data[idx_g], axis=2)
    else:
        prep_real = ori_data[idx_o]
        prep_fake = generated_data[idx_g]

    fig, ax = plt.subplots(figsize=(6, 5))

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_real)
        real_proj = pca.transform(prep_real)
        fake_proj = pca.transform(prep_fake)

        ax.scatter(real_proj[:, 0], real_proj[:, 1],
                   alpha=0.3, s=10, label='Real', c='tab:blue')
        ax.scatter(fake_proj[:, 0], fake_proj[:, 1],
                   alpha=0.3, s=10, label='Synthetic', c='tab:orange')
        ax.set_title('PCA — Real vs Synthetic')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')

    elif analysis == 'tsne':
        combined = np.concatenate([prep_real, prep_fake], axis=0)
        tsne = TSNE(n_components=2, perplexity=40, max_iter=300,
                     random_state=42)
        proj = tsne.fit_transform(combined)

        ax.scatter(proj[:n, 0], proj[:n, 1],
                   alpha=0.3, s=10, label='Real', c='tab:blue')
        ax.scatter(proj[n:, 0], proj[n:, 1],
                   alpha=0.3, s=10, label='Synthetic', c='tab:orange')
        ax.set_title('t-SNE — Real vs Synthetic')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    else:
        raise ValueError(f"analysis must be 'pca' or 'tsne', got '{analysis}'")

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
