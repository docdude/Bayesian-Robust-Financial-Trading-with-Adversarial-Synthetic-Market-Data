"""WaveNet Lambert GAN — drop-in generator replacement for GRT_GAN TimeGAN."""

from .models.wavenet_lambert_gan import (
    build_wavenet_generator,
    build_wavenet_discriminator,
)
from .models.train import train as train_wavenet
from .models.API import GeneratorAPI
