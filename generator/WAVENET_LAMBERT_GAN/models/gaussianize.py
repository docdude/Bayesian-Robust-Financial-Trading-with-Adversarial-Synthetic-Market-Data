"""Lambert W x Gaussian transform for Gaussianizing heavy-tailed data.

Implements the IGMM (Iterative Generalized Method of Moments) estimator
from Goerg (2011) to find the heavy-tail parameter delta, then applies
the inverse Lambert W transform to Gaussianize the data.
"""

import numpy as np
from scipy.special import lambertw
from sklearn.base import TransformerMixin, BaseEstimator


def _w_d(z, delta):
    """Heavy-tail Lambert W forward: z -> y = z * exp(delta * z^2 / 2)."""
    if abs(delta) < 1e-12:
        return z
    return z * np.exp(delta * z ** 2 / 2.0)


def _w_d_inv(y, delta):
    """Inverse heavy-tail Lambert W: y -> z = sign(y) * sqrt(W(delta*y^2) / delta)."""
    if abs(delta) < 1e-12:
        return y
    u = delta * y ** 2
    w_val = np.real(lambertw(u, k=0))
    return np.sign(y) * np.sqrt(np.clip(w_val / delta, 0, None))


def _delta_init(y):
    """Initial delta estimate from excess kurtosis: delta_0 ~ kappa_excess / 6."""
    m2 = np.mean(y ** 2)
    m4 = np.mean(y ** 4)
    kurt_excess = m4 / (m2 ** 2 + 1e-10) - 3.0
    return np.clip(kurt_excess / 6.0, 0, None)


def _delta_gmm(z):
    """GMM moment condition for delta."""
    m2 = np.mean(z ** 2)
    m4 = np.mean(z ** 4)
    kurt_excess = m4 / (m2 ** 2 + 1e-10) - 3.0
    return np.clip(kurt_excess / 6.0, 0, None)


def igmm(y, max_iter=100, tol=1e-6):
    """Iterative GMM to estimate heavy-tail parameter delta."""
    delta = _delta_init(y)
    for _ in range(max_iter):
        z = _w_d_inv(y, delta)
        delta_new = _delta_gmm(z)
        if abs(delta_new - delta) < tol:
            break
        delta = delta_new
    return max(delta, 0.0)


class Gaussianize(TransformerMixin, BaseEstimator):
    """Lambert W x Gaussian transform.

    Fits a heavy-tail parameter delta per column via IGMM, then removes
    heavy tails via the inverse Lambert W transform.  The forward
    (inverse_transform) re-introduces heavy tails.

    Parameters
    ----------
    max_iter : int
        Maximum IGMM iterations per column.
    tol : float
        Convergence tolerance for delta.
    """

    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
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
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(-1, 1)
        result = np.column_stack([
            _w_d(X[:, j], self.delta_[j]) for j in range(X.shape[1])
        ])
        out = result * self.std_ + self.mean_
        return out.ravel() if single else out
