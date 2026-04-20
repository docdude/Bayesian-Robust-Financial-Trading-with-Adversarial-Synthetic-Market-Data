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
                       seq_len, latent_dim, batch_size=256, full_noise=False):
    """Generate synthetic sequences.

    Args:
        full_noise: If True, feed pure Gaussian noise (no real prefix).
                    If False, use half-real/half-noise (training strategy).
    """
    half = seq_len // 2
    idx = np.random.choice(len(real_stock), n_samples, replace=n_samples > len(real_stock))
    selected_stock = real_stock[idx]
    selected_macro = real_macro[idx]

    if full_noise:
        z = np.random.randn(n_samples, seq_len, latent_dim).astype(np.float32)
    else:
        noise = np.random.randn(n_samples, half, latent_dim).astype(np.float32)
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
    return synthetic


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
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of synthetic samples to generate for evaluation')
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
    args = parser.parse_args()

    model_dir = args.model_dir
    print(f"Evaluating WaveNet Lambert GAN at: {model_dir}\n")

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

    # Load generator
    print("\nLoading generator...")
    generator = tf.keras.models.load_model(os.path.join(model_dir, 'generator.keras'))

    # Match sample count
    n_eval = min(args.n_samples, len(stock_data))
    real_sequences = stock_data[:n_eval]
    print(f"Real sequences: {real_sequences.shape}")

    # ---- Mode 1: Standard half-real/half-noise (always runs) ----
    print(f"\nGenerating {n_eval} synthetic sequences (half-real/half-noise)...")
    fake_sequences = generate_synthetic(
        generator, stock_data, macro_data, n_eval, seq_len, latent_dim)
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
        fake_full_noise = generate_synthetic(
            generator, stock_data, macro_data, n_eval, seq_len, latent_dim,
            full_noise=True)
        print(f"Full-noise synthetic: {fake_full_noise.shape}")
        save_dir_fn = os.path.join(model_dir, 'eval_full_noise')
        os.makedirs(save_dir_fn, exist_ok=True)

        _run_eval(real_sequences, fake_full_noise, args,
                  "FULL NOISE — pure Gaussian latent (no real prefix)",
                  save_dir_fn)

    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
