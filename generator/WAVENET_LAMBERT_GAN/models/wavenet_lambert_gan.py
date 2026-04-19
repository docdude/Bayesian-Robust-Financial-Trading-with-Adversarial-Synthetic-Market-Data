"""WaveNet Lambert GAN — Generator and Discriminator architectures.

WaveNet-style dilated causal convolution GAN with:
- Gated activation units (tanh * sigmoid)
- Residual + skip connections
- SpectralNormalization on discriminator
- Linear output on generator (unbounded, like QuantGAN)
- Macro conditioning via concatenation (macro is always real, never noised)
- Half-real / half-noise latent strategy (matching GRT_GAN TimeGAN)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, Multiply, Add, Dense, Concatenate,
    Dropout, GaussianNoise,
    SpectralNormalization,
)
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _maybe_sn(layer, use_spectral_norm=False):
    """Optionally wrap a layer in SpectralNormalization."""
    return SpectralNormalization(layer) if use_spectral_norm else layer


def wavenet_residual_block(
    input_tensor, nfilt, dilation_rate,
    residual_noise_std=None, seed=None,
    use_spectral_norm=False,
):
    """Single gated residual block with dilated causal convolution."""
    x = input_tensor
    if x.shape[-1] != nfilt:
        x = _maybe_sn(
            Conv1D(filters=nfilt, kernel_size=1, padding='same'),
            use_spectral_norm,
        )(x)

    if residual_noise_std:
        x = GaussianNoise(stddev=residual_noise_std, seed=seed)(x)

    tanh_out = _maybe_sn(
        Conv1D(nfilt, 3, dilation_rate=dilation_rate,
               padding='causal', activation='tanh'),
        use_spectral_norm,
    )(x)
    sigm_out = _maybe_sn(
        Conv1D(nfilt, 3, dilation_rate=dilation_rate,
               padding='causal', activation='sigmoid'),
        use_spectral_norm,
    )(x)

    gated = Multiply()([tanh_out, sigm_out])

    skip_out = _maybe_sn(
        Conv1D(nfilt, 1, padding='same'), use_spectral_norm,
    )(gated)
    residual = _maybe_sn(
        Conv1D(nfilt, 1, padding='same'), use_spectral_norm,
    )(gated)
    residual_out = Add()([x, residual])

    return residual_out, skip_out


def wavenet_block(
    input_tensor, nfilt,
    dilation_rates=None,
    residual_noise_std=None, seed=None,
    use_spectral_norm=False,
):
    """One stack of WaveNet residual blocks with exponentially growing dilations."""
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8, 16, 32]
    skip_connections = []
    x = input_tensor
    for i, dilation in enumerate(dilation_rates):
        x, skip = wavenet_residual_block(
            x, nfilt, dilation,
            residual_noise_std=residual_noise_std,
            seed=(seed + i) if seed else None,
            use_spectral_norm=use_spectral_norm,
        )
        skip_connections.append(skip)
    return Add()(skip_connections)


def deep_wavenet(
    input_tensor, nfilt, n_stacks=3,
    dilation_rates=None,
    residual_noise_std=None, seed=None,
    use_spectral_norm=False,
):
    """Multiple stacks of WaveNet blocks."""
    x = input_tensor
    for i in range(n_stacks):
        x = wavenet_block(
            x, nfilt,
            dilation_rates=dilation_rates,
            residual_noise_std=residual_noise_std,
            seed=(seed + 100 + i) if seed else None,
            use_spectral_norm=use_spectral_norm,
        )
    return x


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def build_wavenet_generator(
    sequence_length=120, feature_dim=125, latent_dim=125,
    macro_dim=46,
    nfilt=256, n_stacks=3,
    dilation_rates=None,
    latent_noise_std=0.01, residual_noise_std=0.01,
    seed=None,
):
    """WaveNet generator with macro conditioning and linear (unbounded) output.

    The generator receives two inputs:
    - latent_input: (B, seq_len, latent_dim) — noise (or half-real/half-noise)
    - macro_input:  (B, seq_len, macro_dim)  — always real macro features

    These are concatenated along the feature axis before entering the WaveNet.
    Output is (B, seq_len, feature_dim) — only stock PV features.

    Parameters
    ----------
    sequence_length : int   Window length (default 120 trading days).
    feature_dim : int       Output channels (125 = 25 tickers × 5 PV features).
    latent_dim : int        Noise vector dimensionality per timestep (125).
    macro_dim : int         Macro conditioning features (46).
    nfilt : int             Convolution filter width (256).
    n_stacks : int          Number of WaveNet dilation stacks (3).
    dilation_rates : list   Dilation rates per stack (default [1,2,4,8,16,32]).

    Returns
    -------
    tf.keras.Model with inputs [latent_input, macro_input]
    """
    latent_in = Input(shape=(sequence_length, latent_dim), name='latent_input')
    macro_in = Input(shape=(sequence_length, macro_dim), name='macro_input')

    x = latent_in
    if latent_noise_std:
        x = GaussianNoise(latent_noise_std, name='latent_noise', seed=seed)(x)

    # Concatenate noise + macro conditioning
    x = Concatenate(axis=-1, name='concat_latent_macro')([x, macro_in])

    x = deep_wavenet(
        x, nfilt, n_stacks=n_stacks,
        dilation_rates=dilation_rates,
        residual_noise_std=residual_noise_std, seed=seed,
    )
    output = Dense(feature_dim, name='generator_output')(x)
    return Model(inputs=[latent_in, macro_in], outputs=output,
                 name='WaveNet_Generator')


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

def build_wavenet_discriminator(
    sequence_length=120, feature_dim=125,
    nfilt=256, n_stacks=3,
    dilation_rates=None,
    residual_noise_std=None, dropout_rate=None,
    seed=None, use_spectral_norm=True,
):
    """WaveNet discriminator with optional SpectralNorm and dropout.

    Judges only stock PV features (not macro), producing per-timestep
    real/fake scores like TimeGAN's discriminator.

    Parameters
    ----------
    use_spectral_norm : bool
        Apply SpectralNormalization to all Conv/Dense layers (recommended).

    Returns
    -------
    tf.keras.Model with output (B, seq_len, 1)
    """
    data_in = Input(shape=(sequence_length, feature_dim), name='data_input')
    x = deep_wavenet(
        data_in, nfilt, n_stacks=n_stacks,
        dilation_rates=dilation_rates,
        residual_noise_std=residual_noise_std, seed=seed,
        use_spectral_norm=use_spectral_norm,
    )
    if dropout_rate:
        x = Dropout(dropout_rate, seed=seed)(x)
    output = _maybe_sn(
        Dense(1, activation='sigmoid', name='real_fake_output'),
        use_spectral_norm,
    )(x)
    return Model(inputs=data_in, outputs=output, name='WaveNet_Discriminator')
