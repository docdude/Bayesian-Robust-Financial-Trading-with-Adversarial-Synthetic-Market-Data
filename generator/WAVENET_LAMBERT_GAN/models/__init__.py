"""WaveNet Lambert GAN models package."""
from .wavenet_lambert_gan import build_wavenet_generator, build_wavenet_discriminator
from .gaussianize import Gaussianize
from .train import train
from .API import GeneratorAPI
