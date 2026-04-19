"""Lambert W × Gaussian transform for Gaussianizing heavy-tailed data.

Based on Goerg (2011, 2015):
  - "Lambert W random variables — a new family of generalized skewed
    distributions with applications to risk estimation" (Annals of Applied Statistics)
  - "The Lambert Way to Gaussianize heavy-tailed data with the inverse
    of Tukey's h transformation as a special case" (Scientific World Journal)

Implementation adapted from Henrywils0n/QuantGAN, a TensorFlow implementation of
Wiese et al. (2020) "Quant GANs: Deep Generation of Financial Time Series"
(Quantitative Finance, arXiv:1907.06673).

Usage:
    >>> from utils.gaussianize import Gaussianize
    >>> g = Gaussianize()
    >>> z = g.fit_transform(heavy_tailed_data)    # approximately Gaussian
    >>> y = g.inverse_transform(z)                 # back to original scale
"""

import numpy as np
from scipy.special import lambertw
from sklearn.base import TransformerMixin, BaseEstimator


# ─── Core Lambert W functions ────────────────────────────────────────

def _w_d(z, delta):
    """Heavy-tail Lambert W forward transform: z → y = z·exp(δz²/2).
    
    For δ = 0, this is the identity. For δ > 0, it inflates tails.
    """
    if abs(delta) < 1e-12:
        return z
    return z * np.exp(delta * z ** 2 / 2.0)


def _w_d_inv(y, delta):
    """Inverse heavy-tail Lambert W transform: y → z.
    
    Uses z = sign(y)·√(W(δy²)/δ) where W is the Lambert W function
    (principal branch, k=0).
    """
    if abs(delta) < 1e-12:
        return y
    u = delta * y ** 2
    w_val = np.real(lambertw(u, k=0))
    return np.sign(y) * np.sqrt(np.clip(w_val / delta, 0, None))


# ─── IGMM (Iterative Generalized Method of Moments) ─────────────────

def _delta_init(y):
    """Initial δ estimate from excess kurtosis: δ₀ ≈ κ_excess / 6."""
    m2 = np.mean(y ** 2)
    m4 = np.mean(y ** 4)
    kurt_excess = m4 / (m2 ** 2 + 1e-10) - 3.0
    return np.clip(kurt_excess / 6.0, 0, None)


def _delta_gmm(z):
    """GMM moment condition for δ: match 4th-moment ratio to Gaussian (=3)."""
    m2 = np.mean(z ** 2)
    m4 = np.mean(z ** 4)
    kurt_excess = m4 / (m2 ** 2 + 1e-10) - 3.0
    return np.clip(kurt_excess / 6.0, 0, None)


def igmm(y, max_iter=100, tol=1e-6):
    """Iterative GMM to estimate heavy-tail parameter δ.
    
    Algorithm:
        1. Initialize δ from excess kurtosis of y
        2. Back-transform: z = W_δ⁻¹(y)
        3. Re-estimate δ from moments of z
        4. Repeat until convergence
    
    Args:
        y: Standardized data (mean≈0, std≈1).
        max_iter: Maximum iterations.
        tol: Convergence tolerance for |δ_new − δ_old|.
    
    Returns:
        δ ≥ 0 (heavy-tail parameter; 0 = Gaussian).
    """
    delta = _delta_init(y)
    
    for _ in range(max_iter):
        z = _w_d_inv(y, delta)
        delta_new = _delta_gmm(z)
        if abs(delta_new - delta) < tol:
            break
        delta = delta_new
    
    return max(delta, 0.0)


# ─── Sklearn-compatible transformer ─────────────────────────────────

class Gaussianize(TransformerMixin, BaseEstimator):
    """Lambert W × Gaussian transform to Gaussianize heavy-tailed data.
    
    Fits a Lambert W × Gaussian distribution to each feature independently,
    then applies the inverse transform to produce approximately Gaussian data.
    
    Preprocessing pipeline (from QuantGAN paper, Section 3):
        StandardScaler → Gaussianize → StandardScaler
    
    The first StandardScaler centers and normalizes (needed for IGMM).
    Gaussianize removes heavy tails via inverse Lambert W.
    The second StandardScaler re-standardizes the Gaussianized output.
    
    Args:
        max_iter: Maximum IGMM iterations per feature.
        tol: Convergence tolerance for IGMM.
    """
    
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y=None):
        """Estimate heavy-tail parameter δ for each feature."""
        X = np.asarray(X, dtype=np.float64)
        self._single = X.ndim == 1
        if self._single:
            X = X.reshape(-1, 1)
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0
        
        Z = (X - self.mean_) / self.std_
        
        self.delta_ = np.array([
            igmm(Z[:, j], max_iter=self.max_iter, tol=self.tol)
            for j in range(Z.shape[1])
        ])
        
        return self
    
    def transform(self, X, y=None):
        """Gaussianize: standardize → inverse Lambert W (remove heavy tails)."""
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(-1, 1)
        
        Z = (X - self.mean_) / self.std_
        
        result = np.column_stack([
            _w_d_inv(Z[:, j], self.delta_[j]) for j in range(Z.shape[1])
        ])
        
        return result.ravel() if single else result
    
    def inverse_transform(self, X, y=None):
        """De-Gaussianize: forward Lambert W (re-add heavy tails) → rescale."""
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(-1, 1)
        
        result = np.column_stack([
            _w_d(X[:, j], self.delta_[j]) for j in range(X.shape[1])
        ])
        
        out = result * self.std_ + self.mean_
        return out.ravel() if single else out
