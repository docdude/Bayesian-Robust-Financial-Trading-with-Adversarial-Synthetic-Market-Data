"""
Evaluate the trained TimeGAN model using standard metrics:
1. Discriminative Score - Can a post-hoc classifier tell real from fake?
2. Predictive Score - Can a model trained on fake predict real equally well?
3. Visualization - PCA and t-SNE plots of real vs synthetic distributions

Usage:
    python evaluate_gan.py --model_path output/dj30 [--device cpu]
"""
import os
import sys
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# --- Discriminative Score -----------------------------------------------

class Discriminator(nn.Module):
    """GRU-based binary classifier: real (1) vs fake (0)."""
    def __init__(self, in_dim, h_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, h_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(h_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        out = self.fc(h[-1])
        return out.squeeze(-1)


def discriminative_score(real_data, real_time, fake_data, fake_time, device, epochs=30, lr=1e-3):
    """Train a post-hoc GRU discriminator on real vs fake, return accuracy on held-out set.
    
    Lower accuracy (closer to 0.5) = better GAN.
    """
    n_real, seq_len, dim = real_data.shape
    n_fake = fake_data.shape[0]

    # Create labels
    real_labels = np.ones(n_real)
    fake_labels = np.zeros(n_fake)

    # Combine and shuffle
    all_data = np.concatenate([real_data, fake_data], axis=0).astype(np.float32)
    all_time = np.concatenate([real_time, fake_time], axis=0)
    all_labels = np.concatenate([real_labels, fake_labels])

    idx = np.random.permutation(len(all_data))
    all_data, all_time, all_labels = all_data[idx], all_time[idx], all_labels[idx]

    # 80/20 split
    split = int(0.8 * len(all_data))
    train_x, test_x = all_data[:split], all_data[split:]
    train_t, test_t = all_time[:split], all_time[split:]
    train_y, test_y = all_labels[:split], all_labels[split:]

    train_ds = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_t), torch.FloatTensor(train_y))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = Discriminator(dim, h_dim=64, n_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    for epoch in trange(epochs, desc="Discriminative Score"):
        model.train()
        for xb, tb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, tb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_x).to(device)
        test_time_tensor = torch.LongTensor(test_t)
        logits = model(test_tensor, test_time_tensor)
        preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(float)
        acc = accuracy_score(test_y, preds)

    # Score: |accuracy - 0.5| - lower is better
    disc_score = np.abs(acc - 0.5)
    return acc, disc_score


# --- Predictive Score ---------------------------------------------------

class Predictor(nn.Module):
    """GRU that predicts next timestep features."""
    def __init__(self, in_dim, h_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, h_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(h_dim, in_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)


def predictive_score(real_data, real_time, fake_data, fake_time, device, epochs=30, lr=1e-3):
    """Train on synthetic, test on real. Measures whether fake captures temporal dynamics.
    
    Returns MAE on real test data. Lower = better GAN.
    """
    dim = real_data.shape[2]

    # Train on fake: input=X[:,:-1,:], target=X[:,1:,:]
    train_x = fake_data[:, :-1, :].astype(np.float32)
    train_y = fake_data[:, 1:, :].astype(np.float32)

    train_ds = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = Predictor(dim, h_dim=64, n_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in trange(epochs, desc="Predictive Score"):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test on real
    model.eval()
    test_x = real_data[:, :-1, :].astype(np.float32)
    test_y = real_data[:, 1:, :].astype(np.float32)

    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_x).to(device)
        preds = model(test_tensor).cpu().numpy()
        mae = np.mean(np.abs(test_y - preds))

    # Also compute baseline: train on real, test on real
    train_ds_real = TensorDataset(
        torch.FloatTensor(real_data[:, :-1, :].astype(np.float32)),
        torch.FloatTensor(real_data[:, 1:, :].astype(np.float32))
    )
    train_dl_real = DataLoader(train_ds_real, batch_size=128, shuffle=True)

    model_real = Predictor(dim, h_dim=64, n_layers=2).to(device)
    optimizer_real = torch.optim.Adam(model_real.parameters(), lr=lr)

    for epoch in trange(epochs, desc="Predictive Score (baseline)"):
        model_real.train()
        for xb, yb in train_dl_real:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_real(xb)
            loss = criterion(pred, yb)
            optimizer_real.zero_grad()
            loss.backward()
            optimizer_real.step()

    model_real.eval()
    with torch.no_grad():
        preds_real = model_real(torch.FloatTensor(test_x).to(device)).cpu().numpy()
        mae_real = np.mean(np.abs(test_y - preds_real))

    return mae, mae_real


# --- Visualization ------------------------------------------------------

def visualization(real_data, fake_data, save_dir):
    """Generate PCA and t-SNE plots comparing real vs synthetic distributions."""
    n_samples = min(1000, len(real_data), len(fake_data))
    idx_r = np.random.permutation(len(real_data))[:n_samples]
    idx_f = np.random.permutation(len(fake_data))[:n_samples]

    # Reduce each sample to (seq_len,) by averaging across features
    real_2d = np.mean(real_data[idx_r], axis=2)  # (n, seq_len)
    fake_2d = np.mean(fake_data[idx_f], axis=2)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(real_2d)
    real_pca = pca.transform(real_2d)
    fake_pca = pca.transform(fake_2d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, label='Real', c='tab:blue', s=10)
    ax1.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.3, label='Synthetic', c='tab:orange', s=10)
    ax1.set_title('PCA: Real vs Synthetic')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()

    # t-SNE
    combined = np.concatenate([real_2d, fake_2d], axis=0)
    tsne = TSNE(n_components=2, perplexity=40, max_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(combined)

    ax2.scatter(tsne_results[:n_samples, 0], tsne_results[:n_samples, 1],
                alpha=0.3, label='Real', c='tab:blue', s=10)
    ax2.scatter(tsne_results[n_samples:, 0], tsne_results[n_samples:, 1],
                alpha=0.3, label='Synthetic', c='tab:orange', s=10)
    ax2.set_title('t-SNE: Real vs Synthetic')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'gan_eval_visualization.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved visualization to {path}")


# --- Summary Statistics -------------------------------------------------

def summary_statistics(real_data, fake_data):
    """Compare marginal distributions: mean, std, skew, kurtosis per feature."""
    from scipy import stats

    real_flat = real_data.reshape(-1, real_data.shape[-1])
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1])

    n_feat = real_flat.shape[1]
    results = {
        'mean_diff': np.abs(real_flat.mean(0) - fake_flat.mean(0)),
        'std_diff': np.abs(real_flat.std(0) - fake_flat.std(0)),
        'skew_diff': np.abs(stats.skew(real_flat, axis=0) - stats.skew(fake_flat, axis=0)),
        'kurt_diff': np.abs(stats.kurtosis(real_flat, axis=0) - stats.kurtosis(fake_flat, axis=0)),
    }

    print("\n=== Marginal Distribution Comparison (avg |diff| across features) ===")
    for name, vals in results.items():
        print(f"  {name:15s}: avg={np.mean(vals):.6f}, max={np.max(vals):.6f}, median={np.median(vals):.6f}")

    return results


# --- Autocorrelation Score ----------------------------------------------

def autocorrelation_score(real_data, fake_data, max_lag=10):
    """Compare temporal autocorrelation structure between real and fake."""
    def compute_acf(data, max_lag):
        """Compute average autocorrelation across samples and features."""
        n, seq_len, dim = data.shape
        acfs = np.zeros((max_lag,))
        for lag in range(1, max_lag + 1):
            corrs = []
            for d in range(dim):
                series = data[:, :, d].flatten()
                if len(series) > lag:
                    c = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
            acfs[lag - 1] = np.mean(corrs) if corrs else 0
        return acfs

    real_acf = compute_acf(real_data, max_lag)
    fake_acf = compute_acf(fake_data, max_lag)
    acf_diff = np.abs(real_acf - fake_acf)

    print(f"\n=== Autocorrelation Comparison (lags 1-{max_lag}) ===")
    print(f"  {'Lag':>4s}  {'Real ACF':>10s}  {'Fake ACF':>10s}  {'|Diff|':>10s}")
    for lag in range(max_lag):
        print(f"  {lag+1:4d}  {real_acf[lag]:10.6f}  {fake_acf[lag]:10.6f}  {acf_diff[lag]:10.6f}")
    print(f"  Mean |diff|: {np.mean(acf_diff):.6f}")

    return real_acf, fake_acf, acf_diff


# --- Cross-Feature Correlation ------------------------------------------

def cross_correlation_score(real_data, fake_data):
    """Compare cross-feature correlation matrices."""
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1])

    real_corr = np.corrcoef(real_flat.T)
    fake_corr = np.corrcoef(fake_flat.T)

    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(real_corr, k=1)
    diff = np.abs(real_corr[triu_idx] - fake_corr[triu_idx])

    print(f"\n=== Cross-Feature Correlation Comparison ===")
    print(f"  Correlation pairs:  {len(diff)}")
    print(f"  Mean |diff|:        {np.mean(diff):.6f}")
    print(f"  Median |diff|:      {np.median(diff):.6f}")
    print(f"  Max |diff|:         {np.max(diff):.6f}")
    print(f"  % pairs |diff|<0.1: {100 * np.mean(diff < 0.1):.1f}%")

    return diff


# --- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TimeGAN model")
    parser.add_argument('--model_path', type=str, default='output/dj30', help='Path to model output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--disc_epochs', type=int, default=30, help='Epochs for discriminative score')
    parser.add_argument('--pred_epochs', type=int, default=30, help='Epochs for predictive score')
    parser.add_argument('--skip_disc', action='store_true', help='Skip discriminative score')
    parser.add_argument('--skip_pred', action='store_true', help='Skip predictive score')
    parser.add_argument('--skip_vis', action='store_true', help='Skip visualization')
    args = parser.parse_args()

    device = torch.device(args.device)
    model_path = args.model_path
    print(f"Evaluating GAN model at: {model_path}")
    print(f"Device: {device}\n")

    # Load saved data
    print("Loading data...")
    with open(os.path.join(model_path, 'args.pickle'), 'rb') as f:
        gan_args = pickle.load(f)

    Z_dim = gan_args.Z_dim  # 140 = synthetic feature dims

    with open(os.path.join(model_path, 'train_data.pickle'), 'rb') as f:
        train_data_full = pickle.load(f)  # (4131, 120, 186)
    with open(os.path.join(model_path, 'train_time.pickle'), 'rb') as f:
        train_time = pickle.load(f)
    with open(os.path.join(model_path, 'test_data.pickle'), 'rb') as f:
        test_data_full = pickle.load(f)   # (1033, 120, 186)
    with open(os.path.join(model_path, 'test_time.pickle'), 'rb') as f:
        test_time = pickle.load(f)
    with open(os.path.join(model_path, 'fake_data.pickle'), 'rb') as f:
        fake_data = pickle.load(f)        # (4131, 120, 140)
    with open(os.path.join(model_path, 'fake_time.pickle'), 'rb') as f:
        fake_time = pickle.load(f)

    # Extract only the synthetic features (first Z_dim dims) from real data
    train_real = train_data_full[:, :, :Z_dim].astype(np.float32)
    test_real = test_data_full[:, :, :Z_dim].astype(np.float32)
    fake_data = fake_data.astype(np.float32)

    print(f"Real train: {train_real.shape}")
    print(f"Real test:  {test_real.shape}")
    print(f"Fake data:  {fake_data.shape}")
    print(f"Features per sample: {Z_dim} ({Z_dim // 5} stocks x 5 features)")

    # --- Summary Statistics ---
    summary_statistics(train_real, fake_data)

    # --- Cross-Feature Correlation ---
    cross_correlation_score(train_real, fake_data)

    # --- Autocorrelation ---
    autocorrelation_score(train_real, fake_data, max_lag=10)

    # --- Discriminative Score ---
    if not args.skip_disc:
        print("\n=== Discriminative Score ===")
        acc, disc_score = discriminative_score(
            train_real, train_time, fake_data, fake_time,
            device, epochs=args.disc_epochs
        )
        print(f"  Classifier accuracy: {acc:.4f}")
        print(f"  Discriminative score (|acc - 0.5|): {disc_score:.4f}")
        print(f"  Interpretation: {'Excellent' if disc_score < 0.1 else 'Good' if disc_score < 0.2 else 'Fair' if disc_score < 0.3 else 'Poor'}")
    
    # --- Predictive Score ---
    if not args.skip_pred:
        print("\n=== Predictive Score ===")
        mae_fake, mae_real = predictive_score(
            test_real, test_time, fake_data, fake_time,
            device, epochs=args.pred_epochs
        )
        print(f"  MAE (trained on fake, tested on real):  {mae_fake:.6f}")
        print(f"  MAE (trained on real, tested on real):  {mae_real:.6f}")
        print(f"  Ratio (fake/real MAE): {mae_fake/mae_real:.4f}")
        print(f"  Interpretation: {'Excellent' if mae_fake/mae_real < 1.1 else 'Good' if mae_fake/mae_real < 1.3 else 'Fair' if mae_fake/mae_real < 1.5 else 'Poor'}")

    # --- Visualization ---
    if not args.skip_vis:
        print("\n=== Generating Visualization ===")
        visualization(train_real, fake_data, model_path)

    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
