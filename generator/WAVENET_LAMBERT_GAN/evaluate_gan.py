"""
Evaluate trained WaveNet Lambert GAN (multi-stock) using standard GAN quality metrics.

Metrics:
1. Summary Statistics - marginal distribution comparison (mean, std, skew, kurtosis)
2. Autocorrelation Score - temporal dependency comparison
3. Cross-Feature Correlation - inter-feature correlation comparison
4. Discriminative Score - post-hoc MLP classifier (real vs fake)
5. Predictive Score - train on fake, test on real
6. Visualization - PCA and t-SNE plots

Multi-stock model:
- Generator takes dual inputs: [z_latent (N, 120, 125), macro (N, 120, 46)]
- Uses half-real/half-noise latent strategy from TimeGAN
- Real data: pre-processed NPY arrays from datasets/output_data_lambert/
- Evaluation in Lambert-transformed space (same basis as training)

Usage:
    python evaluate_gan.py --model_dir output/dj30 --data_dir datasets/output_data_lambert
    python evaluate_gan.py --model_dir output/dj30 --n_samples 1000 --skip_disc --skip_pred
"""
import os
import sys
import pickle
import argparse

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from generator.WAVENET_LAMBERT_GAN.metrics import (
    summary_statistics,
    autocorrelation_score,
    cross_correlation_score,
    discriminative_score,
    predictive_score,
)
from generator.WAVENET_LAMBERT_GAN.metrics.visualization import visualization as vis_fn


def _save_visualization(real_data, fake_data, save_dir):
    """PCA + t-SNE side-by-side saved to disk."""
    vis_fn(real_data, fake_data, 'pca',
           save_path=os.path.join(save_dir, 'gan_eval_pca.png'))
    vis_fn(real_data, fake_data, 'tsne',
           save_path=os.path.join(save_dir, 'gan_eval_tsne.png'))
    print(f"Saved visualizations to {save_dir}/")


# ---------------------------------------------------------------------------
# Load real data from pre-processed NPYs
# ---------------------------------------------------------------------------

def load_real_data(data_dir, train_tickers=None, expected_feature_dim=None):
    """Load pre-processed windowed (N, seq_len, feature_dim) arrays.

    If train_tickers is provided, slice and reorder the channel axis so it
    matches the order/subset the generator was trained on.  Channels are
    interleaved per-ticker (5 features each).
    """
    stock_data = np.load(os.path.join(data_dir, 'output_data.npy')).astype(np.float32)
    macro_data = np.load(os.path.join(data_dir, 'output_macro_data.npy')).astype(np.float32)
    ticker_list_full = np.load(os.path.join(data_dir, 'ticker_list.npy'))
    print(f"  Loaded stock_data:  {stock_data.shape}")
    print(f"  Loaded macro_data:  {macro_data.shape}")
    print(f"  Dataset tickers ({len(ticker_list_full)}): {list(ticker_list_full)}")

    if train_tickers is None:
        tickers = [str(t) for t in ticker_list_full]
    else:
        pos = {str(t): i for i, t in enumerate(ticker_list_full)}
        missing = [t for t in train_tickers if t not in pos]
        if missing:
            raise ValueError(f"train_tickers not found in dataset: {missing}")
        ch_idx = np.concatenate(
            [np.arange(pos[t] * 5, pos[t] * 5 + 5) for t in train_tickers])
        stock_data = stock_data[:, :, ch_idx]
        tickers = list(train_tickers)
        print(f"  Realigned to {len(tickers)} train tickers: {stock_data.shape}")

    if expected_feature_dim is not None and stock_data.shape[-1] != expected_feature_dim:
        raise ValueError(
            f"stock_data has feature_dim={stock_data.shape[-1]} but config expects "
            f"{expected_feature_dim}.  Pass --train_tickers matching the generator.")

    return stock_data, macro_data, tickers


# ---------------------------------------------------------------------------
# Generate synthetic data from trained model (half-real / half-noise)
# ---------------------------------------------------------------------------

def generate_synthetic(generator, real_stock, real_macro, n_samples,
                       seq_len, latent_dim, batch_size=256, full_noise=False,
                       idx=None, rng=None):
    """Generate synthetic sequences.

    Args:
        full_noise: If True, feed pure Gaussian noise (no real prefix).
                    If False, use half-real/half-noise (training strategy).
        idx: Optional pre-computed window indices for conditioning. If None,
             a random subset of size ``n_samples`` is drawn. Returned alongside
             the synthetic array so callers can pair each fake window with the
             same real window used for conditioning.
        rng: Optional ``np.random.Generator``.  If None, falls back to the
             legacy global ``np.random`` state (kept for backward compat).

    Returns
    -------
    synthetic : ndarray (n_samples, seq_len, feature_dim)
    idx       : ndarray (n_samples,) — windows selected from ``real_stock``
    """
    half = seq_len // 2
    _choice = rng.choice if rng is not None else np.random.choice
    _randn = (lambda *s: rng.standard_normal(s).astype(np.float32)) \
        if rng is not None else (lambda *s: np.random.randn(*s).astype(np.float32))

    if idx is None:
        idx = _choice(len(real_stock), n_samples,
                      replace=n_samples > len(real_stock))
    selected_stock = real_stock[idx]
    selected_macro = real_macro[idx]

    if full_noise:
        z = _randn(n_samples, seq_len, latent_dim)
    else:
        noise = _randn(n_samples, half, latent_dim)
        z = np.concatenate([
            selected_stock[:, :half, :latent_dim],
            noise,
        ], axis=1).astype(np.float32)

    # Generate in batches to avoid OOM on large n_samples
    parts = []
    for i in range(0, n_samples, batch_size):
        parts.append(
            generator([z[i:i+batch_size], selected_macro[i:i+batch_size]],
                      training=False).numpy()
        )
    synthetic = np.concatenate(parts, axis=0)
    return synthetic, idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_eval(real_sequences, fake_sequences, args, label, save_dir):
    """Run all requested metrics for a (real, fake) pair."""
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"# real={real_sequences.shape}  fake={fake_sequences.shape}")
    print(f"{'#'*70}")

    summary_statistics(real_sequences, fake_sequences)
    cross_correlation_score(real_sequences, fake_sequences)
    autocorrelation_score(real_sequences, fake_sequences, max_lag=10)

    if not args.skip_disc:
        print("\n=== Discriminative Score ===")
        acc, disc_score = discriminative_score(
            real_sequences, fake_sequences, epochs=args.disc_epochs)
        print(f"  Classifier accuracy: {acc:.4f}")
        print(f"  Discriminative score (|acc - 0.5|): {disc_score:.4f}")
        quality = ('Excellent' if disc_score < 0.1 else
                   'Good' if disc_score < 0.2 else
                   'Fair' if disc_score < 0.3 else 'Poor')
        print(f"  Interpretation: {quality}")

    if not args.skip_pred:
        print("\n=== Predictive Score ===")
        mae = predictive_score(
            real_sequences, fake_sequences, epochs=args.pred_epochs)
        print(f"  MAE (trained on fake, tested on real): {mae:.6f}")

    if not args.skip_vis:
        print("\n=== Generating Visualization ===")
        _save_visualization(real_sequences, fake_sequences, save_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained WaveNet Lambert GAN model (multi-stock)")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model output directory (contains generator.keras, config.pkl)')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/output_data_lambert',
                        help='Path to pre-processed NPY data directory')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of synthetic samples to generate for evaluation '
                             '(matches notebook default of 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for window selection, latent noise, and '
                             'classifier splits.  Set to match the notebook.')
    parser.add_argument('--disc_epochs', type=int, default=200,
                        help='Max iterations for discriminative score MLP')
    parser.add_argument('--pred_epochs', type=int, default=200,
                        help='Max iterations for predictive score MLP')
    parser.add_argument('--skip_disc', action='store_true',
                        help='Skip discriminative score')
    parser.add_argument('--skip_pred', action='store_true',
                        help='Skip predictive score')
    parser.add_argument('--skip_vis', action='store_true',
                        help='Skip visualization')
    parser.add_argument('--full_noise', action='store_true',
                        help='Also evaluate with pure noise input (no real prefix)')
    parser.add_argument('--second_half', action='store_true',
                        help='Also evaluate only the generated second half (timesteps 60-119)')
    parser.add_argument('--train_tickers', type=str, default=None,
                        help='Comma-separated ticker list the generator was trained on '
                             '(order matters). If omitted, all dataset tickers are used.')
    parser.add_argument('--checkpoint_epoch', type=int, default=None,
                        help='Evaluate a specific checkpoint (loads '
                             'checkpoints/generator_epoch{N}.keras instead of '
                             'generator.keras). If omitted, uses the final generator.keras.')
    args = parser.parse_args()

    # Seed all RNG sources so CLI results match the notebook.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model_dir = args.model_dir
    print(f"Evaluating WaveNet Lambert GAN at: {model_dir}  (seed={args.seed})\n")

    # Load config
    with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    seq_len = config['seq_len']
    latent_dim = config['latent_dim']
    feature_dim = config['feature_dim']
    macro_dim = config['macro_dim']
    half = seq_len // 2
    print(f"Config: seq_len={seq_len}, feature_dim={feature_dim}, "
          f"macro_dim={macro_dim}, latent_dim={latent_dim}, epochs={config['epochs']}")

    # Load real data from NPYs
    print("\nLoading real data...")
    train_tickers = (
        [t.strip() for t in args.train_tickers.split(',') if t.strip()]
        if args.train_tickers else None
    )
    stock_data, macro_data, tickers = load_real_data(
        args.data_dir, train_tickers=train_tickers,
        expected_feature_dim=feature_dim)

    # Register channel names for per-channel metrics (if metrics support it)
    try:
        from generator.WAVENET_LAMBERT_GAN.metrics.evaluation_metrics import (
            set_channel_names, build_multistock_channel_names)
        set_channel_names(build_multistock_channel_names(
            tickers, ['close_ret', 'open_ret', 'high_ret', 'low_ret', 'volume_ret']))
    except Exception as _e:
        print(f"  (channel-name registration skipped: {_e})")

    # Load generator (final by default, or a specific checkpoint if requested)
    print("\nLoading generator...")
    if args.checkpoint_epoch is not None:
        generator_path = os.path.join(
            model_dir, 'checkpoints',
            f'generator_epoch{args.checkpoint_epoch}.keras')
        if not os.path.exists(generator_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {generator_path}")
        print(f"  Using checkpoint epoch {args.checkpoint_epoch}")
    else:
        generator_path = os.path.join(model_dir, 'generator.keras')
    print(f"  Generator: {generator_path}")
    generator = tf.keras.models.load_model(generator_path)

    # Match sample count
    n_eval = min(args.n_samples, len(stock_data))

    # ---- Mode 1: Standard half-real/half-noise (always runs) ----
    print(f"\nGenerating {n_eval} synthetic sequences (half-real/half-noise)...")
    fake_sequences, eval_idx = generate_synthetic(
        generator, stock_data, macro_data, n_eval, seq_len, latent_dim,
        rng=rng)
    # Pair each fake window with the SAME real window used for conditioning,
    # so every metric compares apples-to-apples.
    real_sequences = stock_data[eval_idx]
    print(f"Real sequences:      {real_sequences.shape}")
    print(f"Synthetic sequences: {fake_sequences.shape}")

    _run_eval(real_sequences, fake_sequences, args,
              "FULL SEQUENCE — half-real/half-noise latent", model_dir)

    # ---- Mode 2: Second-half only (generated portion) ----
    if args.second_half:
        real_2nd = real_sequences[:, half:, :]
        fake_2nd = fake_sequences[:, half:, :]
        save_dir_2nd = os.path.join(model_dir, 'eval_second_half')
        os.makedirs(save_dir_2nd, exist_ok=True)

        _run_eval(real_2nd, fake_2nd, args,
                  f"SECOND HALF ONLY — timesteps [{half}:{seq_len}] (pure generation)",
                  save_dir_2nd)

    # ---- Mode 3: Full noise (no real prefix) ----
    if args.full_noise:
        print(f"\nGenerating {n_eval} synthetic sequences (full noise, no real prefix)...")
        # Reuse the same eval_idx so real/fake populations match Mode 1.
        fake_full_noise, _ = generate_synthetic(
            generator, stock_data, macro_data, n_eval, seq_len, latent_dim,
            full_noise=True, idx=eval_idx, rng=rng)
        print(f"Full-noise synthetic: {fake_full_noise.shape}")
        save_dir_fn = os.path.join(model_dir, 'eval_full_noise')
        os.makedirs(save_dir_fn, exist_ok=True)

        _run_eval(real_sequences, fake_full_noise, args,
                  "FULL NOISE — pure Gaussian latent (no real prefix)",
                  save_dir_fn)

    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
