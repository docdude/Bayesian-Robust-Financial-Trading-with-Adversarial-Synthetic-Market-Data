import numpy as np
import tensorflow as tf


def save_model(model, path):
    model.save(path)


# Add to discriminator loss calculation:
def compute_gradient_penalty(real_samples, fake_samples, discriminator):
    # Cast to float32 to resolve dtype mismatch
    real_samples = tf.cast(real_samples, tf.float32)
    fake_samples = tf.cast(fake_samples, tf.float32)
    
    # Compute interpolated samples
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1], 0., 1., dtype=tf.float32)
    interpolated = alpha * real_samples + (1. - alpha) * fake_samples
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    
    gradients = gp_tape.gradient(pred, interpolated)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-8)
    return tf.reduce_mean((gradients_norm - 1.0) ** 2)

def gradient_penalty(real_samples, fake_samples, discriminator):
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred = discriminator(interpolates)
    grads = tape.gradient(pred, [interpolates])[0]
    penalty = tf.reduce_mean((tf.norm(grads, axis=[1,2]) - 1.0) ** 2)
    return penalty


def smooth_positive_labels(y):
	return y - 0.3 + (tf.random.uniform(tf.shape(y)) * 0.5)

def smooth_negative_labels(y):
	return y + tf.random.uniform(tf.shape(y)) * 0.3


def progressive_label_smoothing(epoch, labels, min_smooth_real=0.8, max_smooth_real=0.95, 
                                min_smooth_fake=0.05, max_smooth_fake=0.2):
    """
    Applies progressive label smoothing to both real (1s) and fake (0s) labels over epochs.

    Args:
        epoch (int): Current training epoch.
        labels (tensor): Tensor of real (1s) or fake (0s) labels.
        min_smooth_real (float): Minimum smoothing factor for real labels.
        max_smooth_real (float): Maximum smoothing factor for real labels.
        min_smooth_fake (float): Minimum smoothing factor for fake labels.
        max_smooth_fake (float): Maximum smoothing factor for fake labels.

    Returns:
        Tensor: Smoothed labels.
    """
    # Progressive smoothing factor using exponential decay
    smoothing_factor = min_smooth_real + (max_smooth_real - min_smooth_real) * (1 - np.exp(-epoch / 50))
    
    # Apply smoothing to real labels (1s) and fake labels (0s) separately
    smoothed_labels = tf.where(
        labels > 0.5,  # Real labels (1s)
        labels * smoothing_factor + (1 - smoothing_factor) * 0.5,  # Shift 1s down slightly
        labels * (1 - smoothing_factor) + (smoothing_factor * min_smooth_fake)  # Shift 0s up slightly
    )
    
    return smoothed_labels


def compute_moment_loss(real, fake):
    """Moment matching loss: penalise differences in std, skewness, kurtosis.

    All statistics are computed over the flattened batch so every
    generated window contributes to a single set of moments.
    Operates in the same [0,1] scaled space as the rest of training.

    Args:
        real: Tensor of real samples (any shape, will be flattened).
        fake: Tensor of generated samples (same shape as real).

    Returns:
        Scalar loss = relative_std_gap + |skew_diff| + |kurt_diff|.
    """
    real_flat = tf.reshape(real, [-1])
    fake_flat = tf.reshape(fake, [-1])

    # Mean & std
    mu_r, mu_f = tf.reduce_mean(real_flat), tf.reduce_mean(fake_flat)
    std_r = tf.math.reduce_std(real_flat) + 1e-8
    std_f = tf.math.reduce_std(fake_flat) + 1e-8

    # Standardised residuals
    z_r = (real_flat - mu_r) / std_r
    z_f = (fake_flat - mu_f) / std_f

    # Skewness (3rd standardised moment)
    skew_r = tf.reduce_mean(z_r ** 3)
    skew_f = tf.reduce_mean(z_f ** 3)

    # Kurtosis (4th standardised moment, excess)
    kurt_r = tf.reduce_mean(z_r ** 4) - 3.0
    kurt_f = tf.reduce_mean(z_f ** 4) - 3.0

    # L1 penalties — each term has natural scale ~O(1)
    loss_std = tf.abs(std_r - std_f) / std_r      # relative std gap
    loss_skew = tf.abs(skew_r - skew_f)
    loss_kurt = tf.abs(kurt_r - kurt_f)

    return loss_std + loss_skew + loss_kurt


def compute_tail_loss(real, fake, tail_pct=0.05):
    """Quantile matching loss: penalise mismatch at distribution tails.

    Sorts both real and fake samples, then computes MSE between the
    bottom and top ``tail_pct`` quantiles.  This directly targets
    under-representation of extreme values that moment matching alone
    can miss (moments summarize globally; this targets the tails).

    Args:
        real: Tensor of real samples (any shape, will be flattened).
        fake: Tensor of generated samples (same shape as real).
        tail_pct: Fraction of each tail to match (default 5%).

    Returns:
        Scalar loss = MSE(lower tails) + MSE(upper tails).
    """
    real_flat = tf.sort(tf.reshape(real, [-1]))
    fake_flat = tf.sort(tf.reshape(fake, [-1]))

    n = tf.shape(real_flat)[0]
    k = tf.maximum(tf.cast(tf.cast(n, tf.float32) * tail_pct, tf.int32), 1)

    # Lower tail (smallest k values)
    lower_loss = tf.reduce_mean(tf.square(real_flat[:k] - fake_flat[:k]))
    # Upper tail (largest k values)
    upper_loss = tf.reduce_mean(tf.square(real_flat[-k:] - fake_flat[-k:]))

    return lower_loss + upper_loss


class AdaptiveLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, target_ratio, adjustment_factor):
        self.initial_lr = initial_lr
        self.target_ratio = target_ratio  # Target ratio (e.g., D_loss should be ~0.8 * G_loss)
        self.adjustment_factor = adjustment_factor
    def __call__(self, d_loss, g_loss, name):
        ratio = d_loss / (g_loss + 1e-8)  # Avoid division by zero
        # Determine new learning rate based on ratio
        if ratio < self.target_ratio:
            new_lr = self.initial_lr * self.adjustment_factor
            print(f"Increasing {name} lr: {new_lr}")
        elif ratio > self.target_ratio:
            new_lr = self.initial_lr / self.adjustment_factor
            print(f"Decreasing {name} lr: {new_lr}")
        else:
            new_lr = self.initial_lr  # Keep LR constant
        
        # Update the initial_lr for the next epoch so that adjustments are cumulative
        self.initial_lr = new_lr  
        return new_lr

class BalancedAdaptiveLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, 
                 initial_gen_lr, 
                 initial_disc_lr, 
                 adjustment_factor=1.1, 
                 tolerance=0.4, 
                 min_lr=1e-5,
                 max_lr=5e-4,
                 max_lr_ratio=5.0):
        self.gen_lr = initial_gen_lr
        self.disc_lr = initial_disc_lr
        self.adjustment_factor = adjustment_factor
        self.tolerance = tolerance
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_lr_ratio = max_lr_ratio  # max allowed ratio between disc/gen LRs

    def __call__(self, d_loss, g_loss):
        ratio = d_loss / (g_loss + 1e-8)
        print(f"Current d_loss/g_loss ratio: {ratio:.2f}")

        # If within the tolerance band [1-tol, 1+tol], no adjustment needed
        if abs(ratio - 1.0) < self.tolerance:
            print(f"  → In equilibrium band. LRs unchanged: "
                  f"gen={self.gen_lr:.2e}, disc={self.disc_lr:.2e}")
            return self.gen_lr, self.disc_lr

        if ratio > 1:
            # Discriminator loss is higher: REDUCE disc LR to stabilize it
            new_disc_lr = self.disc_lr / self.adjustment_factor
            new_gen_lr = self.gen_lr * self.adjustment_factor
            print(f"  → D losing: disc_lr ↓ {new_disc_lr:.2e}, gen_lr ↑ {new_gen_lr:.2e}")
        else:
            # Generator loss is higher: REDUCE gen LR to stabilize it
            new_disc_lr = self.disc_lr * self.adjustment_factor
            new_gen_lr = self.gen_lr / self.adjustment_factor
            print(f"  → G losing: disc_lr ↑ {new_disc_lr:.2e}, gen_lr ↓ {new_gen_lr:.2e}")

        # Enforce per-LR min/max constraints
        new_gen_lr = max(min(new_gen_lr, self.max_lr), self.min_lr)
        new_disc_lr = max(min(new_disc_lr, self.max_lr), self.min_lr)

        # Enforce max ratio constraint — prevent runaway divergence
        lr_ratio = new_disc_lr / (new_gen_lr + 1e-12)
        if lr_ratio > self.max_lr_ratio:
            print(f"  → Clamping LR ratio {lr_ratio:.1f}x → {self.max_lr_ratio}x")
            new_disc_lr = new_gen_lr * self.max_lr_ratio
        elif lr_ratio < 1.0 / self.max_lr_ratio:
            print(f"  → Clamping LR ratio {lr_ratio:.2f}x → {1.0/self.max_lr_ratio:.2f}x")
            new_gen_lr = new_disc_lr * self.max_lr_ratio

        self.gen_lr = new_gen_lr
        self.disc_lr = new_disc_lr

        return self.gen_lr, self.disc_lr
