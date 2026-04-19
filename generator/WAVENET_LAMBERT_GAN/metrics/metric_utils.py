"""GAN quality metrics for WaveNet Lambert GAN evaluation.

All metrics are framework-agnostic (numpy + sklearn). No PyTorch dependency.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def summary_statistics(real_data, fake_data):
    """Compare marginal distributions: mean, std, skew, kurtosis.

    Args:
        real_data: (N, seq_len, dim) or (N, dim)
        fake_data: (M, seq_len, dim) or (M, dim)

    Returns:
        dict with mean_diff, std_diff, skew_diff, kurt_diff arrays (per feature)
    """
    real_flat = real_data.reshape(-1, real_data.shape[-1]) if real_data.ndim > 2 else real_data.reshape(-1, 1)
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1]) if fake_data.ndim > 2 else fake_data.reshape(-1, 1)

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


# ---------------------------------------------------------------------------
# Autocorrelation Score
# ---------------------------------------------------------------------------

def autocorrelation_score(real_data, fake_data, max_lag=10):
    """Compare temporal autocorrelation structure between real and fake.

    Args:
        real_data: (N, seq_len, dim)
        fake_data: (M, seq_len, dim)
        max_lag:   number of lags to compare

    Returns:
        (real_acf, fake_acf, acf_diff) — arrays of shape (max_lag,)
    """
    def compute_acf(data, max_lag):
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
            acfs[lag - 1] = np.mean(corrs) if corrs else 0.0
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


# ---------------------------------------------------------------------------
# Cross-Feature Correlation
# ---------------------------------------------------------------------------

def cross_correlation_score(real_data, fake_data):
    """Compare cross-feature correlation matrices.

    Args:
        real_data: (N, seq_len, dim) with dim >= 2
        fake_data: (M, seq_len, dim)

    Returns:
        array of |diff| for each upper-triangular correlation pair
    """
    real_flat = real_data.reshape(-1, real_data.shape[-1]) if real_data.ndim > 2 else real_data.reshape(-1, 1)
    fake_flat = fake_data.reshape(-1, fake_data.shape[-1]) if fake_data.ndim > 2 else fake_data.reshape(-1, 1)

    if real_flat.shape[1] < 2:
        print("\n=== Cross-Feature Correlation: skipped (single feature) ===")
        return np.array([])

    real_corr = np.corrcoef(real_flat.T)
    fake_corr = np.corrcoef(fake_flat.T)

    triu_idx = np.triu_indices_from(real_corr, k=1)
    diff = np.abs(real_corr[triu_idx] - fake_corr[triu_idx])

    print(f"\n=== Cross-Feature Correlation Comparison ===")
    print(f"  Correlation pairs:  {len(diff)}")
    print(f"  Mean |diff|:        {np.mean(diff):.6f}")
    print(f"  Median |diff|:      {np.median(diff):.6f}")
    print(f"  Max |diff|:         {np.max(diff):.6f}")

    return diff


# ---------------------------------------------------------------------------
# Discriminative Score (sklearn MLP — no PyTorch)
# ---------------------------------------------------------------------------

def discriminative_score(real_data, fake_data, epochs=200):
    """Train an MLP classifier on real vs fake.

    Returns:
        (accuracy, |accuracy - 0.5|) — lower score → better GAN
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    real_flat = real_data.reshape(len(real_data), -1)
    fake_flat = fake_data.reshape(len(fake_data), -1)

    X = np.concatenate([real_flat, fake_flat], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(real_flat)), np.zeros(len(fake_flat))])

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=epochs,
                        early_stopping=True, n_iter_no_change=15,
                        validation_fraction=0.15, random_state=42)
    clf.fit(X_train, y_train)

    converged = clf.n_iter_ < epochs
    print(f"  MLP trained {clf.n_iter_}/{epochs} iters ({'converged' if converged else 'hit max_iter'})")

    acc = accuracy_score(y_test, clf.predict(X_test))

    disc_score = np.abs(acc - 0.5)
    return acc, disc_score


# ---------------------------------------------------------------------------
# Predictive Score (train on fake, test on real — sklearn MLP)
# ---------------------------------------------------------------------------

def predictive_score(real_data, fake_data, epochs=200):
    """Train one-step-ahead predictor on synthetic, test on real.

    Returns:
        MAE on real test data
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    dim = real_data.shape[-1] if real_data.ndim == 3 else 1

    if fake_data.ndim == 3:
        train_x = fake_data[:, :-1, :].reshape(-1, dim)
        train_y = fake_data[:, 1:, :].reshape(-1, dim)
    else:
        train_x = fake_data[:, :-1].reshape(-1, 1)
        train_y = fake_data[:, 1:].reshape(-1, 1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    train_x_s = scaler_x.fit_transform(train_x)
    train_y_s = scaler_y.fit_transform(train_y)

    reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs,
                       early_stopping=True, n_iter_no_change=15,
                       validation_fraction=0.15, random_state=42)
    reg.fit(train_x_s, train_y_s)

    converged = reg.n_iter_ < epochs
    print(f"  MLP trained {reg.n_iter_}/{epochs} iters ({'converged' if converged else 'hit max_iter'})")

    if real_data.ndim == 3:
        test_x = real_data[:, :-1, :].reshape(-1, dim)
        test_y = real_data[:, 1:, :].reshape(-1, dim)
    else:
        test_x = real_data[:, :-1].reshape(-1, 1)
        test_y = real_data[:, 1:].reshape(-1, 1)

    test_x_s = scaler_x.transform(test_x)
    preds_s = reg.predict(test_x_s)
    preds = scaler_y.inverse_transform(preds_s)
    mae = np.mean(np.abs(test_y - preds))

    return mae
