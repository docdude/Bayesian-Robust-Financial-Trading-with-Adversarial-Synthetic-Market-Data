"""Lambda calibration script — run on Lightning AI with ep 3000 checkpoint.

Measures raw (unweighted) loss magnitudes, then computes optimal lambdas
so each loss contributes a specified fraction of the total gradient signal.

Usage:
    python calibrate_lambdas.py --ckpt_epoch 3000
"""
import os, sys, argparse, pickle, numpy as np, tensorflow as tf

# Ensure imports work
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from generator.WAVENET_LAMBERT_GAN.models.train import (
    compute_moment_loss, compute_std_loss, compute_quantile_loss, compute_tail_loss
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_epoch', type=int, default=3000)
    parser.add_argument('--output_dir', type=str,
                        default='generator/WAVENET_LAMBERT_GAN/output/dj30')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/output_data_lambert')
    parser.add_argument('--n_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    gen_path = os.path.join(ckpt_dir, f'generator_epoch{args.ckpt_epoch}.keras')
    print(f"Loading generator from {gen_path}")
    generator = tf.keras.models.load_model(gen_path)

    stock_data = np.load(os.path.join(args.data_dir, 'output_data.npy')).astype(np.float32)
    macro_data = np.load(os.path.join(args.data_dir, 'output_macro_data.npy')).astype(np.float32)
    print(f"Data: stock={stock_data.shape}, macro={macro_data.shape}")

    seq_len = stock_data.shape[1]
    latent_dim = stock_data.shape[2]
    half_seq = seq_len // 2

    raw = {k: [] for k in ['recon', 'moment', 'tail', 'std', 'quantile']}
    # Also compute per-loss gradient norms to see actual gradient magnitude
    grad_norms = {k: [] for k in raw}

    for b in range(args.n_batches):
        idx = np.random.choice(len(stock_data), args.batch_size, replace=False)
        batch_stock = tf.constant(stock_data[idx])
        batch_macro = tf.constant(macro_data[idx])

        real_first_half = batch_stock[:, :half_seq, :latent_dim]
        noise = tf.random.normal((args.batch_size, half_seq, latent_dim))
        z_gen = tf.concat([real_first_half, noise], axis=1)

        # Forward pass
        with tf.GradientTape(persistent=True) as tape:
            fake = generator([z_gen, batch_macro], training=False)

            l_recon = tf.reduce_mean(tf.keras.losses.huber(batch_stock, fake, delta=0.5))
            l_moment = compute_moment_loss(batch_stock, fake)
            l_tail = compute_tail_loss(batch_stock, fake, tail_pct=0.05)
            l_std = compute_std_loss(batch_stock, fake)
            l_quantile = compute_quantile_loss(batch_stock, fake)

        losses = {'recon': l_recon, 'moment': l_moment, 'tail': l_tail,
                  'std': l_std, 'quantile': l_quantile}

        for k, loss in losses.items():
            raw[k].append(loss.numpy())
            grads = tape.gradient(loss, generator.trainable_variables)
            total_norm = 0.0
            for g in grads:
                if g is not None:
                    total_norm += tf.reduce_sum(tf.square(g)).numpy()
            grad_norms[k].append(np.sqrt(total_norm))

        del tape
        if (b + 1) % 5 == 0:
            print(f"  Batch {b+1}/{args.n_batches}")

    print("\n" + "=" * 72)
    print(f"RAW LOSS MAGNITUDES @ epoch {args.ckpt_epoch} (avg over {args.n_batches} batches)")
    print("=" * 72)
    means = {}
    for k in raw:
        vals = raw[k]
        means[k] = np.mean(vals)
        print(f"  {k:12s}: loss={np.mean(vals):.6f} ± {np.std(vals):.6f}  "
              f"grad_norm={np.mean(grad_norms[k]):.4f} ± {np.std(grad_norms[k]):.4f}")

    # --- Gradient-norm based calibration (more principled) ---
    print("\n" + "=" * 72)
    print("GRADIENT-NORM CALIBRATION")
    print("=" * 72)
    print("  (Lambdas that equalize gradient contribution per loss term)")

    gn_means = {k: np.mean(v) for k, v in grad_norms.items()}

    # Target: each loss should contribute a specified fraction of total gradient
    budgets = {
        'recon':    0.40,   # anchor — keeps temporal structure
        'std':      0.25,   # primary distributional fix
        'quantile': 0.15,   # secondary distributional fix
        'moment':   0.10,   # minor
        'tail':     0.10,   # minor
    }

    # lambda_i = budget_i / grad_norm_i (normalized so recon lambda = 1.0)
    raw_lambdas = {k: budgets[k] / gn_means[k] for k in gn_means}
    scale = 1.0 / raw_lambdas['recon']
    lambdas = {k: v * scale for k, v in raw_lambdas.items()}

    print(f"\n  {'Loss':12s} {'Grad norm':>12s} {'Budget':>8s} {'Lambda':>10s} "
          f"{'λ×GradNorm':>12s} {'Contrib%':>10s}")
    print(f"  {'-'*68}")

    total = sum(lambdas[k] * gn_means[k] for k in lambdas)
    for k in ['recon', 'moment', 'tail', 'std', 'quantile']:
        weighted = lambdas[k] * gn_means[k]
        frac = weighted / total
        print(f"  {k:12s} {gn_means[k]:12.4f} {budgets[k]:8.1%} {lambdas[k]:10.4f} "
              f"{weighted:12.4f} {frac:10.1%}")

    print(f"\n  RECOMMENDED LAMBDAS (gradient-norm calibrated):")
    for k in ['recon', 'moment', 'tail', 'std', 'quantile']:
        print(f"    --lambda_{k:8s} {lambdas[k]:.2f}")

    # --- Also show loss-magnitude calibration for reference ---
    print("\n" + "=" * 72)
    print("LOSS-MAGNITUDE CALIBRATION (alternative)")
    print("=" * 72)

    raw_lam2 = {k: budgets[k] / means[k] for k in means}
    scale2 = 1.0 / raw_lam2['recon']
    lam2 = {k: v * scale2 for k, v in raw_lam2.items()}

    total2 = sum(lam2[k] * means[k] for k in lam2)
    print(f"\n  {'Loss':12s} {'Raw loss':>12s} {'Budget':>8s} {'Lambda':>10s} "
          f"{'λ×Loss':>12s} {'Contrib%':>10s}")
    print(f"  {'-'*68}")
    for k in ['recon', 'moment', 'tail', 'std', 'quantile']:
        weighted = lam2[k] * means[k]
        frac = weighted / total2
        print(f"  {k:12s} {means[k]:12.6f} {budgets[k]:8.1%} {lam2[k]:10.4f} "
              f"{weighted:12.6f} {frac:10.1%}")

    print(f"\n  RECOMMENDED LAMBDAS (loss-magnitude calibrated):")
    for k in ['recon', 'moment', 'tail', 'std', 'quantile']:
        print(f"    --lambda_{k:8s} {lam2[k]:.2f}")


if __name__ == '__main__':
    main()
