"""WaveNet Lambert GAN — Main entry point.

Trains a multi-stock WaveNet Lambert GAN via ``models.train.train()``.

Usage:
    # Default DJ30 training
    python main.py --exp dj30

    # Resume from checkpoint
    python main.py --exp dj30 --resume

    # Custom data directory
    python main.py --exp dj30 --data_dir datasets/output_data_lambert
"""

import argparse
import os
import sys

# Ensure the package root is importable when running as a script
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from generator.WAVENET_LAMBERT_GAN.models.train import train


def main(args):
    """Train WaveNet Lambert GAN on multi-stock NPY data."""

    print(f"\n{'='*60}")
    print(f"  Training WaveNet Lambert GAN")
    print(f"  data_dir: {args.data_dir}")
    print(f"  output:   {args.output_dir}")
    print(f"  exp:      {args.exp}")
    print(f"  epochs:   {args.epochs}")
    print(f"  resume:   {args.resume}")
    print(f"{'='*60}")

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
        beta_1=args.beta_1,
        huber_delta=args.huber_delta,
        lambda_adv=args.lambda_adv,
        lambda_recon=args.lambda_recon,
        lambda_moment=args.lambda_moment,
        lambda_tail=args.lambda_tail,
        lambda_std=args.lambda_std,
        lambda_quantile=args.lambda_quantile,
        use_spectral_norm=not args.no_spectral_norm,
        lr_adjust_every=args.lr_adjust_every,
        dis_thresh=args.dis_thresh,
        seed=args.seed,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        exp=args.exp,
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train WaveNet Lambert GAN on multi-stock NPY data')

    # Data paths
    parser.add_argument('--exp', type=str, default='dj30',
                        help='Experiment name (sets output_dir and TensorBoard tag)')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(_project_root, 'datasets', 'output_data_lambert'),
                        help='Path to pre-processed NPY data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: output/{exp})')

    # Training hyperparameters (defaults match TimeGAN-aligned config)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=125)
    parser.add_argument('--nfilt', type=int, default=256)
    parser.add_argument('--n_stacks', type=int, default=3)
    parser.add_argument('--lr_gen', type=float, default=0.0002)
    parser.add_argument('--lr_disc', type=float, default=0.0001)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--huber_delta', type=float, default=0.5)
    parser.add_argument('--lambda_adv', type=float, default=1.0)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_moment', type=float, default=5.0)
    parser.add_argument('--lambda_tail', type=float, default=50.0)
    parser.add_argument('--lambda_std', type=float, default=15.0)
    parser.add_argument('--lambda_quantile', type=float, default=0.001)
    parser.add_argument('--no_spectral_norm', action='store_true',
                        help='Disable spectral normalisation on discriminator')
    parser.add_argument('--lr_adjust_every', type=int, default=50)
    parser.add_argument('--dis_thresh', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    # Checkpoint / resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in output_dir')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50)')

    # GPU control
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device index or "cpu"')

    args = parser.parse_args()

    # Resolve output_dir from --exp if not explicitly set
    if args.output_dir is None:
        args.output_dir = os.path.join(
            _this_dir, 'output', args.exp)

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    main(args)
