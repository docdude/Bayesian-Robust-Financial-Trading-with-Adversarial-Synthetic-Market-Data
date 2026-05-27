"""WaveNet Lambert GAN — Training script (multi-stock NPY pipeline).

Loads pre-processed NPY arrays from the Lambert preprocessing notebook
(output_data_lambert/) and trains a WaveNet GAN on multi-stock PV features
with macro conditioning, using the half-real/half-noise strategy from TimeGAN.

Usage (CLI):
    python -m generator.WAVENET_LAMBERT_GAN.models.train \
        --data_dir datasets/output_data_lambert \
        --epochs 2000 \
        --output_dir generator/WAVENET_LAMBERT_GAN/output/dj30

Usage (Python):
    from generator.WAVENET_LAMBERT_GAN.models.train import train
    artifacts = train(data_dir='datasets/output_data_lambert', epochs=2000)
"""

import argparse
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from .wavenet_lambert_gan import build_wavenet_generator, build_wavenet_discriminator


# ---------------------------------------------------------------------------
# Loss utilities (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

def smooth_positive_labels(y):
    return y - 0.3 + (tf.random.uniform(tf.shape(y)) * 0.5)


def smooth_negative_labels(y):
    return y + tf.random.uniform(tf.shape(y)) * 0.3


def compute_moment_loss(real, fake):
    """Moment matching: penalise std / skew / kurtosis differences."""
    real_flat = tf.reshape(real, [-1])
    fake_flat = tf.reshape(fake, [-1])
    mu_r = tf.reduce_mean(real_flat)
    mu_f = tf.reduce_mean(fake_flat)
    std_r = tf.math.reduce_std(real_flat) + 1e-8
    std_f = tf.math.reduce_std(fake_flat) + 1e-8
    z_r = (real_flat - mu_r) / std_r
    z_f = (fake_flat - mu_f) / std_f
    skew_r = tf.reduce_mean(z_r ** 3)
    skew_f = tf.reduce_mean(z_f ** 3)
    kurt_r = tf.reduce_mean(z_r ** 4) - 3.0
    kurt_f = tf.reduce_mean(z_f ** 4) - 3.0
    return (tf.abs(std_r - std_f) / std_r
            + tf.abs(skew_r - skew_f)
            + tf.abs(kurt_r - kurt_f))


def compute_std_loss(real, fake):
    """Per-feature standard deviation matching across the batch.

    Computes std for each of the 125 features independently across
    (batch × time), then returns mean |std_real - std_fake| / std_real.
    This prevents the generator from collapsing variance per feature.
    """
    # real, fake: (batch, seq_len, features)
    real_2d = tf.reshape(real, [-1, tf.shape(real)[2]])   # (batch*seq, features)
    fake_2d = tf.reshape(fake, [-1, tf.shape(fake)[2]])   # (batch*seq, features)
    std_r = tf.math.reduce_std(real_2d, axis=0) + 1e-8    # (features,)
    std_f = tf.math.reduce_std(fake_2d, axis=0) + 1e-8    # (features,)
    return tf.reduce_mean(tf.abs(std_r - std_f) / std_r)


def compute_quantile_loss(real, fake, quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)):
    """Per-feature quantile matching loss.

    For each feature, sorts values across (batch × time), computes the
    requested quantiles for both real and fake, and returns the mean
    absolute relative error.  This forces the generator to reproduce
    the full marginal shape — not just std — per feature.
    """
    # (batch*seq, features)
    real_2d = tf.reshape(real, [-1, tf.shape(real)[2]])
    fake_2d = tf.reshape(fake, [-1, tf.shape(fake)[2]])
    n = tf.cast(tf.shape(real_2d)[0], tf.float32)
    loss = 0.0
    for q in quantiles:
        idx = tf.cast(tf.round(q * (n - 1.0)), tf.int32)
        real_sorted = tf.sort(real_2d, axis=0)
        fake_sorted = tf.sort(fake_2d, axis=0)
        r_q = real_sorted[idx]   # (features,)
        f_q = fake_sorted[idx]   # (features,)
        denom = tf.abs(r_q) + 1e-8
        loss += tf.reduce_mean(tf.abs(r_q - f_q) / denom)
    return loss / float(len(quantiles))


def compute_tail_loss(real, fake, tail_pct=0.05):
    """MSE between sorted bottom/top quantiles."""
    real_flat = tf.sort(tf.reshape(real, [-1]))
    fake_flat = tf.sort(tf.reshape(fake, [-1]))
    n = tf.shape(real_flat)[0]
    k = tf.maximum(tf.cast(tf.cast(n, tf.float32) * tail_pct, tf.int32), 1)
    lower = tf.reduce_mean(tf.square(real_flat[:k] - fake_flat[:k]))
    upper = tf.reduce_mean(tf.square(real_flat[-k:] - fake_flat[-k:]))
    return lower + upper


class BalancedAdaptiveLR:
    """Rebalances G/D learning rates to maintain training equilibrium."""

    def __init__(self, gen_lr, disc_lr, factor=1.1, tol=0.4,
                 min_lr=3e-6, max_lr=5e-4, max_ratio=5.0):
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.factor = factor
        self.tol = tol
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_ratio = max_ratio

    def __call__(self, d_loss, g_loss):
        ratio = d_loss / (g_loss + 1e-8)
        if abs(ratio - 1.0) < self.tol:
            return self.gen_lr, self.disc_lr
        if ratio > 1:
            self.disc_lr /= self.factor
            self.gen_lr *= self.factor
        else:
            self.disc_lr *= self.factor
            self.gen_lr /= self.factor
        self.gen_lr = max(min(self.gen_lr, self.max_lr), self.min_lr)
        self.disc_lr = max(min(self.disc_lr, self.max_lr), self.min_lr)
        lr_ratio = self.disc_lr / (self.gen_lr + 1e-12)
        if lr_ratio > self.max_ratio:
            self.disc_lr = self.gen_lr * self.max_ratio
        elif lr_ratio < 1.0 / self.max_ratio:
            self.gen_lr = self.disc_lr * self.max_ratio
        return self.gen_lr, self.disc_lr


# ---------------------------------------------------------------------------
# Data loading from pre-processed NPYs
# ---------------------------------------------------------------------------

def load_npy_data(data_dir):
    """Load pre-processed windowed arrays from the Lambert notebook output.

    Expected files in data_dir:
        output_data.npy         — (N, 120, 125) Lambert-transformed PV features
        output_macro_data.npy   — (N, 120, 46)  MinMax-scaled macro features
        output_history_data.npy — (N, 120, 125) history context windows

    Returns
    -------
    stock_data : ndarray (N, seq_len, feature_dim)
    macro_data : ndarray (N, seq_len, macro_dim)
    history_data : ndarray (N, seq_len, feature_dim)
    """
    stock_data = np.load(os.path.join(data_dir, 'output_data.npy')).astype(np.float32)
    macro_data = np.load(os.path.join(data_dir, 'output_macro_data.npy')).astype(np.float32)
    history_data = np.load(os.path.join(data_dir, 'output_history_data.npy')).astype(np.float32)

    print(f"  Loaded stock_data:   {stock_data.shape}")
    print(f"  Loaded macro_data:   {macro_data.shape}")
    print(f"  Loaded history_data: {history_data.shape}")

    # Validate shapes
    assert stock_data.ndim == 3, f"stock_data must be 3D, got {stock_data.ndim}D"
    assert macro_data.ndim == 3, f"macro_data must be 3D, got {macro_data.ndim}D"
    assert stock_data.shape[0] == macro_data.shape[0], "Sample count mismatch"
    assert stock_data.shape[1] == macro_data.shape[1], "Sequence length mismatch"

    # Check for NaN/Inf
    for name, arr in [('stock_data', stock_data), ('macro_data', macro_data)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"NaN or Inf found in {name}")

    return stock_data, macro_data, history_data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _save_checkpoint(output_dir, generator, discriminator, gen_opt, disc_opt,
                     adaptive_lr, history, epoch):
    """Save a training checkpoint (models + optimizers + state)."""
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    generator.save(os.path.join(ckpt_dir, f'generator_epoch{epoch}.keras'))
    discriminator.save(os.path.join(ckpt_dir, f'discriminator_epoch{epoch}.keras'))
    state = {
        'epoch': epoch,
        'gen_opt_vars': [v.numpy() for v in gen_opt.variables],
        'disc_opt_vars': [v.numpy() for v in disc_opt.variables],
        'adaptive_lr_gen': adaptive_lr.gen_lr,
        'adaptive_lr_disc': adaptive_lr.disc_lr,
        'history': history,
    }
    with open(os.path.join(ckpt_dir, f'state_epoch{epoch}.pkl'), 'wb') as f:
        pickle.dump(state, f)
    print(f"  Checkpoint saved: epoch {epoch}")


def _find_latest_checkpoint(output_dir):
    """Find the latest checkpoint epoch in output_dir/checkpoints/."""
    import re
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        return None, 0
    best_epoch = 0
    for f in os.listdir(ckpt_dir):
        m = re.match(r'state_epoch(\d+)\.pkl', f)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
    if best_epoch == 0:
        return None, 0
    return ckpt_dir, best_epoch


def train(
    data_dir=None,
    output_dir=None,
    epochs=2000,
    batch_size=256,
    seq_len=120,
    feature_dim=125,
    macro_dim=46,
    latent_dim=None,
    nfilt=256,
    n_stacks=3,
    dilation_rates=None,
    lr_gen=0.0002,
    lr_disc=0.0001,
    beta_1=0.5,
    huber_delta=0.5,
    lambda_adv=1.0,
    lambda_recon=1.0,
    lambda_moment=5.0,
    lambda_tail=50.0,
    lambda_std=15.0,
    lambda_quantile=0.001,
    use_spectral_norm=True,
    lr_adjust_every=50,
    dis_thresh=0.15,
    seed=42,
    resume=False,
    checkpoint_every=50,
    exp=None,
    verbose=True,
):
    """Train a WaveNet Lambert GAN on multi-stock NPY data.

    Uses half-real/half-noise latent strategy from TimeGAN:
    - First half of sequence: real stock data (conditioning context)
    - Second half of sequence: random noise (for generation)
    - Macro features: always real, concatenated as conditioning

    Parameters
    ----------
    data_dir : str
        Path to directory with pre-processed NPY files (output_data_lambert/).
    output_dir : str
        Directory to save model and config artifacts.
    feature_dim : int
        Number of stock PV features (25 tickers × 5 = 125).
    macro_dim : int
        Number of macro conditioning features (46).
    latent_dim : int or None
        Noise dimensionality. Must match feature_dim for the half-real latent
        construction; if None, inferred from the loaded data.
    nfilt : int
        WaveNet convolution filter width (256, matching TimeGAN hidden_dim).
    n_stacks : int
        Number of WaveNet dilation stacks (3, matching TimeGAN num_layers).
    dilation_rates : list
        Dilation rates per stack (default [1,2,4,8,16,32]).
    dis_thresh : float
        Discriminator loss threshold — D only updates when D_loss > threshold
        (matching TimeGAN's dis_thresh=0.15).

    Returns
    -------
    dict with keys: 'generator', 'discriminator', 'history', 'config', 'output_dir'
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8, 16, 32]

    # -- Resolve paths --
    if data_dir is None:
        data_dir = os.path.join('datasets', 'output_data_lambert')
    if output_dir is None:
        output_dir = os.path.join('generator', 'WAVENET_LAMBERT_GAN', 'output', 'dj30')
    os.makedirs(output_dir, exist_ok=True)

    # -- Load pre-processed NPYs --
    if verbose:
        print(f"Loading NPY data from {data_dir}")
    stock_data, macro_data, history_data = load_npy_data(data_dir)

    n_samples = stock_data.shape[0]
    seq_len = stock_data.shape[1]
    feature_dim = stock_data.shape[2]
    macro_dim = macro_data.shape[2]
    half_seq = seq_len // 2
    if latent_dim is None:
        latent_dim = feature_dim
    elif latent_dim != feature_dim:
        print(
            f"  Overriding latent_dim={latent_dim} to feature_dim={feature_dim} "
            "for half-real latent construction."
        )
        latent_dim = feature_dim

    if verbose:
        print(f"  {n_samples} windows, seq_len={seq_len}, "
              f"feature_dim={feature_dim}, macro_dim={macro_dim}")
        print(f"  Half-real/half-noise split at timestep {half_seq}")

    # -- Build models --
    generator = build_wavenet_generator(
        sequence_length=seq_len, feature_dim=feature_dim,
        latent_dim=latent_dim, macro_dim=macro_dim,
        nfilt=nfilt, n_stacks=n_stacks,
        dilation_rates=dilation_rates,
        latent_noise_std=0.01, residual_noise_std=0.01, seed=seed,
    )
    discriminator = build_wavenet_discriminator(
        sequence_length=seq_len, feature_dim=feature_dim,
        nfilt=nfilt, n_stacks=n_stacks,
        dilation_rates=dilation_rates,
        residual_noise_std=0.01, dropout_rate=0.1,
        seed=seed, use_spectral_norm=use_spectral_norm,
    )

    if verbose:
        print(f"  Generator params:     {generator.count_params():,}")
        print(f"  Discriminator params: {discriminator.count_params():,}")

    gen_opt = tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1)
    adaptive_lr = BalancedAdaptiveLR(lr_gen, lr_disc)

    # -- Resume from checkpoint --
    start_epoch = 0
    if resume:
        ckpt_dir, ckpt_epoch = _find_latest_checkpoint(output_dir)
        if ckpt_epoch > 0:
            print(f"  Resuming from checkpoint at epoch {ckpt_epoch}")
            generator = tf.keras.models.load_model(
                os.path.join(ckpt_dir, f'generator_epoch{ckpt_epoch}.keras'))
            discriminator = tf.keras.models.load_model(
                os.path.join(ckpt_dir, f'discriminator_epoch{ckpt_epoch}.keras'))
            with open(os.path.join(ckpt_dir, f'state_epoch{ckpt_epoch}.pkl'), 'rb') as f:
                state = pickle.load(f)
            # Build optimizer slots WITHOUT perturbing weights.
            # Keras 3 Adam exposes .build(var_list) which creates m/v/iter slots
            # without invoking apply_gradients (so weights are untouched).
            gen_opt.build(generator.trainable_variables)
            disc_opt.build(discriminator.trainable_variables)
            # Restore optimizer variable values (m, v, iterations, learning_rate)
            for var, val in zip(gen_opt.variables, state['gen_opt_vars']):
                var.assign(val)
            for var, val in zip(disc_opt.variables, state['disc_opt_vars']):
                var.assign(val)
            adaptive_lr.gen_lr = state['adaptive_lr_gen']
            adaptive_lr.disc_lr = state['adaptive_lr_disc']
            # Re-sync the live optimizer LR to the adapted value so the first
            # post-resume epochs don't run at the stale CLI default LR
            # (adaptive_lr would otherwise only re-assign it every lr_adjust_every).
            gen_opt.learning_rate.assign(adaptive_lr.gen_lr)
            disc_opt.learning_rate.assign(adaptive_lr.disc_lr)
            # Advance RNG past the already-completed epochs so the first
            # resumed batches are not identical to the very first training batches.
            tf.random.set_seed(seed + ckpt_epoch)
            np.random.seed(seed + ckpt_epoch)
            history = state['history']
            # Backfill new loss keys for checkpoints saved before they existed
            for new_key in ['gen_std', 'gen_quantile']:
                if new_key not in history:
                    history[new_key] = [0.0] * len(history['gen_loss'])
            start_epoch = ckpt_epoch
            print(f"  Restored optimizers and history. Continuing from epoch {start_epoch}.")
        else:
            print("  No checkpoint found, starting from scratch.")

    # -- TensorBoard (inside WAVENET_LAMBERT_GAN directory) --
    _wavenet_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tb_log_dir = os.path.join(_wavenet_dir, 'tensorboard', exp or os.path.basename(output_dir))
    tb_writer = tf.summary.create_file_writer(tb_log_dir)
    if verbose:
        print(f"  TensorBoard logs: {tb_log_dir}")

    # -- Train step --
    @tf.function
    def train_step(real_stock, real_macro, update_disc):
        bs = tf.shape(real_stock)[0]

        # Half-real / half-noise latent construction (TimeGAN strategy):
        # First half timesteps: real stock data, second half: random noise
        noise_second_half = tf.random.normal((bs, half_seq, latent_dim))
        real_first_half = real_stock[:, :half_seq, :latent_dim]
        z_gen = tf.concat([real_first_half, noise_second_half], axis=1)

        noise_second_half_d = tf.random.normal((bs, half_seq, latent_dim))
        z_disc = tf.concat([real_first_half, noise_second_half_d], axis=1)

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_gen = generator([z_gen, real_macro], training=True)
            fake_disc = generator([z_disc, real_macro], training=True)
            real_out = discriminator(real_stock, training=True)
            fake_out_d = discriminator(fake_disc, training=True)
            fake_out_g = discriminator(fake_gen, training=True)

            # D emits logits (no sigmoid). from_logits=True uses the stable
            # sigmoid_cross_entropy_with_logits kernel so D keeps producing a
            # usable adversarial gradient even when |logit| is large.
            d_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                smooth_positive_labels(tf.ones_like(real_out)), real_out,
                from_logits=True))
            d_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                smooth_negative_labels(tf.zeros_like(fake_out_d)), fake_out_d,
                from_logits=True))
            d_loss = d_real + d_fake

            g_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                smooth_positive_labels(tf.ones_like(fake_out_g)), fake_out_g,
                from_logits=True))
            g_recon = tf.reduce_mean(
                tf.keras.losses.huber(real_stock, fake_gen, delta=huber_delta))
            g_moment = compute_moment_loss(real_stock, fake_gen)
            g_tail = compute_tail_loss(real_stock, fake_gen, tail_pct=0.05)
            g_std = compute_std_loss(real_stock, fake_gen)
            g_quantile = compute_quantile_loss(real_stock, fake_gen)

            g_loss = (lambda_adv * g_adv + lambda_recon * g_recon
                      + lambda_moment * g_moment + lambda_tail * g_tail
                      + lambda_std * g_std
                      + lambda_quantile * g_quantile)

        # Always update generator
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g
                   for g in g_grads]
        gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

        # Only update discriminator when D_loss > threshold (TimeGAN strategy)
        if update_disc:
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g
                       for g in d_grads]
            disc_opt.apply_gradients(
                zip(d_grads, discriminator.trainable_variables))

        return {
            'disc_loss': d_loss, 'gen_loss': g_loss,
            'disc_real': d_real, 'disc_fake': d_fake,
            'gen_adv': g_adv, 'gen_recon': g_recon,
            'gen_moment': g_moment, 'gen_tail': g_tail,
            'gen_std': g_std,
            'gen_quantile': g_quantile,
        }

    # -- Dataset: paired (stock, macro) --
    dataset = (tf.data.Dataset.from_tensor_slices((stock_data, macro_data))
               .shuffle(n_samples, seed=seed)
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))

    if start_epoch == 0:
        history = {k: [] for k in [
            'disc_loss', 'gen_loss', 'disc_real', 'disc_fake',
            'gen_adv', 'gen_recon', 'gen_moment', 'gen_tail', 'gen_std', 'gen_quantile',
            'lr_gen', 'lr_disc',
        ]}

    # -- Build config early so it exists from the first checkpoint --
    config = {
        'data_dir': data_dir,
        'epochs': epochs,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'feature_dim': feature_dim,
        'macro_dim': macro_dim,
        'latent_dim': latent_dim,
        'nfilt': nfilt,
        'n_stacks': n_stacks,
        'dilation_rates': dilation_rates,
        'lr_gen': lr_gen,
        'lr_disc': lr_disc,
        'beta_1': beta_1,
        'huber_delta': huber_delta,
        'lambda_adv': lambda_adv,
        'lambda_recon': lambda_recon,
        'lambda_moment': lambda_moment,
        'lambda_tail': lambda_tail,
        'lambda_std': lambda_std,
        'lambda_quantile': lambda_quantile,
        'use_spectral_norm': use_spectral_norm,
        'dis_thresh': dis_thresh,
        'seed': seed,
        'n_samples': n_samples,
        'training_time_s': None,  # updated after training completes
    }
    with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # -- Training loop --
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_m = {k: [] for k in history if k not in ('lr_gen', 'lr_disc')}

        # Check if D should update this epoch based on last epoch's D_loss
        update_d = True
        if epoch > 0 and history['disc_loss'][-1] < dis_thresh:
            update_d = False

        for batch_stock, batch_macro in dataset:
            m = train_step(batch_stock, batch_macro, update_d)
            for k in epoch_m:
                epoch_m[k].append(m[k].numpy())
        for k in epoch_m:
            history[k].append(np.mean(epoch_m[k]))

        if (epoch + 1) % lr_adjust_every == 0:
            new_g, new_d = adaptive_lr(
                history['disc_loss'][-1] / 2.0, history['gen_adv'][-1])
            gen_opt.learning_rate.assign(new_g)
            disc_opt.learning_rate.assign(new_d)

        history['lr_gen'].append(float(gen_opt.learning_rate))
        history['lr_disc'].append(float(disc_opt.learning_rate))

        # -- TensorBoard logging --
        with tb_writer.as_default(step=epoch):
            tf.summary.scalar('Joint/Discriminator_Loss', history['disc_loss'][-1])
            tf.summary.scalar('Joint/Generator_Loss', history['gen_loss'][-1])
            tf.summary.scalar('Joint/Disc_Real', history['disc_real'][-1])
            tf.summary.scalar('Joint/Disc_Fake', history['disc_fake'][-1])
            tf.summary.scalar('Joint/Gen_Adv', history['gen_adv'][-1])
            tf.summary.scalar('Joint/Gen_Recon', history['gen_recon'][-1])
            tf.summary.scalar('Joint/Gen_Moment', history['gen_moment'][-1])
            tf.summary.scalar('Joint/Gen_Tail', history['gen_tail'][-1])
            tf.summary.scalar('Joint/Gen_Std', history['gen_std'][-1])
            tf.summary.scalar('Joint/Gen_Quantile', history['gen_quantile'][-1])
            tf.summary.scalar('LR/Generator', history['lr_gen'][-1])
            tf.summary.scalar('LR/Discriminator', history['lr_disc'][-1])
        tb_writer.flush()

        if verbose and (epoch % 10 == 0 or epoch == start_epoch):
            d_status = "" if update_d else " (D frozen)"
            epoch_time = (time.time() - t0) / (epoch - start_epoch + 1)
            print(f"  Epoch {epoch:4d}/{epochs}"
                  f" | D: {history['disc_loss'][-1]:.4f}"
                  f" | G: {history['gen_loss'][-1]:.4f}"
                  f" | {epoch_time:.1f}s/ep{d_status}")

        # -- Periodic checkpoint --
        if (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(output_dir, generator, discriminator,
                             gen_opt, disc_opt, adaptive_lr, history, epoch + 1)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Training complete in {elapsed:.0f}s")

    # -- Save artifacts --
    generator.save(os.path.join(output_dir, 'generator.keras'))
    discriminator.save(os.path.join(output_dir, 'discriminator.keras'))

    config['training_time_s'] = elapsed
    with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # Save training history
    with open(os.path.join(output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    if verbose:
        print(f"  Saved to {output_dir}/")

    return {
        'generator': generator,
        'discriminator': discriminator,
        'history': history,
        'config': config,
        'output_dir': output_dir,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train WaveNet Lambert GAN on multi-stock NPY data')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/output_data_lambert')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Noise dimension; defaults to loaded feature_dim')
    parser.add_argument('--nfilt', type=int, default=256)
    parser.add_argument('--n_stacks', type=int, default=3)
    parser.add_argument('--lr_gen', type=float, default=0.0002)
    parser.add_argument('--lr_disc', type=float, default=0.0001)
    parser.add_argument('--dis_thresh', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device index or "cpu"')
    args = parser.parse_args()

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        nfilt=args.nfilt,
        n_stacks=args.n_stacks,
        lr_gen=args.lr_gen,
        lr_disc=args.lr_disc,
        dis_thresh=args.dis_thresh,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
