"""WaveNet Lambert GAN — Inference API (multi-stock, GRT_GAN-compatible).

Drop-in replacement for ``GRT_GAN.models.API.GeneratorAPI``.
Loads a trained WaveNet Lambert GAN generator, pre-processed NPY arrays,
and Lambert fit parameters.  Implements the same ``call(timestamp, macro_epsilon)``
interface expected by the DQN RL pipeline.

Usage:
    from generator.WAVENET_LAMBERT_GAN.models.API import GeneratorAPI

    api = GeneratorAPI(
        model_path='generator/WAVENET_LAMBERT_GAN/output/dj30',
        ticker_name='AAPL',
        obs_features=cfg.dataset.features_name,
        temporal_features=cfg.dataset.temporals_name,
        data_dir='datasets/output_data_lambert_derived',
    )
    feature_df = api.call(timestamp='2020-01-02', macro_epsilon=noise)
"""

import os
import pickle
import time
import random
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Lambert fit params were pickled against a top-level ``gaussianize`` module.
# Make the local implementation importable under that module name before unpickling.
CURRENT = str(Path(__file__).resolve().parent)
if CURRENT not in sys.path:
    sys.path.append(CURRENT)
from gaussianize import Gaussianize  # noqa: F401


class GeneratorAPI:
    """Inference wrapper for a trained WaveNet Lambert GAN (GRT_GAN-compatible)."""

    def __init__(self, model_path, ticker_name, obs_features, temporal_features,
                 feature_method="derived", data_dir=None):
        """Load the model and NPY data for inference.

        Parameters
        ----------
        model_path : str
            Path to trained model directory (contains generator.keras, config.pkl).
        ticker_name : str
            Target ticker for feature extraction (e.g. 'AAPL').
        obs_features : list[str]
            Feature column names expected by the downstream DQN.
        temporal_features : list[str]
            Temporal feature names (passed through, not generated).
        feature_method : str
            'derived' (default, recursive compounding from initial close,
            matches the preprocessing notebook and GRT_GAN's API default) or
            'log_returns' (independent per-channel exp-cumsum).
        data_dir : str or None
            Path to the preprocessed NPY data folder. If omitted, uses the
            model config's data_dir when present, then falls back to the legacy
            derived DJ30 data directory.
        """
        self.feature_method = feature_method
        self.ticker_name = ticker_name
        self.obs_features = obs_features
        self.temporal_features = temporal_features

        # -- Load model config first; it records the training data directory. --
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)

        self.seq_len = self.config['seq_len']
        self.latent_dim = self.config['latent_dim']
        self.feature_dim = self.config['feature_dim']
        self.macro_dim = self.config['macro_dim']

        # -- Load NPY data (same layout as GRT_GAN) --
        repo_root = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..'))
        default_data_dir = os.path.join(
            repo_root, 'datasets', 'output_data_lambert_derived')
        data_dir = data_dir or self.config.get('data_dir') or default_data_dir
        if not os.path.isabs(data_dir):
            cwd_candidate = os.path.abspath(data_dir)
            repo_candidate = os.path.normpath(os.path.join(repo_root, data_dir))
            data_dir = cwd_candidate if os.path.exists(cwd_candidate) else repo_candidate
        data_dir = os.path.normpath(data_dir)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        self.data_dir = data_dir

        self.output_data = np.load(os.path.join(data_dir, 'output_data.npy'))
        self.output_history_data = np.load(os.path.join(data_dir, 'output_history_data.npy'))
        self.output_macro_data = np.load(os.path.join(data_dir, 'output_macro_data.npy'))
        self.output_mask_data = np.load(os.path.join(data_dir, 'output_mask_data.npy'))
        self.output_starting_date = np.load(
            os.path.join(data_dir, 'output_starting_date.npy'), allow_pickle=True)
        self.ticker_list = np.load(os.path.join(data_dir, 'ticker_list.npy'))
        self.time = np.load(os.path.join(data_dir, 'time.npy'))
        self.original_close = np.load(os.path.join(data_dir, 'output_original_close.npy'))
        self.original_open = np.load(os.path.join(data_dir, 'output_original_open.npy'))
        self.output_adj_factor = np.load(os.path.join(data_dir, 'output_adj_factor.npy'))

        if feature_method == "log_returns":
            self.original_high = np.load(os.path.join(data_dir, 'output_initial_high.npy'))
            self.original_low = np.load(os.path.join(data_dir, 'output_initial_low.npy'))
            self.original_volume = np.load(os.path.join(data_dir, 'output_initial_volume.npy'))

        if self.output_data.shape[1] != self.seq_len:
            raise ValueError(
                f"Data seq_len={self.output_data.shape[1]} but model expects {self.seq_len}"
            )
        if self.output_data.shape[2] != self.feature_dim:
            raise ValueError(
                f"Data feature_dim={self.output_data.shape[2]} but model expects {self.feature_dim}"
            )
        if self.output_macro_data.shape[2] != self.macro_dim:
            raise ValueError(
                f"Data macro_dim={self.output_macro_data.shape[2]} but model expects {self.macro_dim}"
            )
        if self.latent_dim != self.feature_dim:
            raise ValueError(
                f"Model latent_dim={self.latent_dim} must equal feature_dim={self.feature_dim} "
                "for half-real WaveNet API inference."
            )
        if self.ticker_name not in set(self.ticker_list.astype(str)):
            raise ValueError(
                f"Ticker {self.ticker_name!r} not found in GAN data tickers: "
                f"{list(self.ticker_list.astype(str))}"
            )

        # Lambert inverse transform params: {ticker: [(scaler1, gauss, scaler2) × 5]}
        with open(os.path.join(data_dir, 'lambert_fit_params.pkl'), 'rb') as f:
            self.lambert_fit_params = pickle.load(f)

        # Date index for fast timestamp lookup
        self._date_array = pd.DatetimeIndex(self.output_starting_date[:, 0])
        self._date_set = set(self._date_array)
        print(f"GAN date range: {self._date_array[0]} → {self._date_array[-1]}"
              f"  ({len(self._date_array)} dates)")

        self.generator = tf.keras.models.load_model(
            os.path.join(model_path, 'generator.keras'))

        print(f"WaveNet Lambert GAN loaded: ticker={ticker_name}, "
              f"seq_len={self.seq_len}, feature_dim={self.feature_dim}, "
              f"macro_dim={self.macro_dim}, data_dir={self.data_dir}")

    # ------------------------------------------------------------------
    # Model inference (matches GRT_GAN's half-real / half-noise strategy)
    # ------------------------------------------------------------------

    def model_inference(self, real_data, target_macro, T):
        """Generate synthetic PV features using half-real / half-noise.

        Parameters
        ----------
        real_data : ndarray (batch, seq_len, feature_dim)
        target_macro : ndarray (batch, seq_len, macro_dim)
        T : ndarray (batch,)  — sequence length (unused by WaveNet, kept for API compat)

        Returns
        -------
        ndarray (batch, seq_len, feature_dim)
        """
        half = self.seq_len // 2
        batch_size = real_data.shape[0]
        # Half-real / half-noise latent (TimeGAN strategy)
        noise = np.random.randn(batch_size, half, self.latent_dim).astype(np.float32)
        z = np.concatenate([
            real_data[:, :half, :self.latent_dim],
            noise,
        ], axis=1).astype(np.float32)

        macro = target_macro.astype(np.float32)
        generated = self.generator([z, macro], training=False)
        return generated.numpy()

    # ------------------------------------------------------------------
    # Lambert inverse transform (per-ticker, per-feature)
    # ------------------------------------------------------------------

    def _inverse_lambert_per_feature(self, data, ticker):
        """Invert Lambert 3-stage transform for one ticker's 5 features.

        Parameters
        ----------
        data : ndarray (seq_len, 5) — Lambert-scaled PV features
        ticker : str

        Returns
        -------
        ndarray (seq_len, 5) — original-scale PV features
        """
        params = self.lambert_fit_params[ticker]  # list of 5 (scaler1, gauss, scaler2)
        result = np.zeros_like(data)
        for col_idx in range(5):
            scaler1, gaussianizer, scaler2 = params[col_idx]
            col = data[:, col_idx:col_idx + 1]  # (seq_len, 1)
            # Inverse: scaler2⁻¹ → gaussianize⁻¹ → scaler1⁻¹
            stage2 = scaler2.inverse_transform(col)
            stage1 = gaussianizer.inverse_transform(stage2)
            original = scaler1.inverse_transform(stage1)
            result[:, col_idx] = original.ravel()
        return result

    # ------------------------------------------------------------------
    # OHLCV reconstruction from PV features
    # ------------------------------------------------------------------

    def transform_generated_pv_feature_to_data(self, df_features,
                                                original_close, original_open,
                                                underlying_close_caj):
        """Reconstruct OHLCV from derived PV features (recursive compounding)."""
        smoothed_close = [original_close]
        for t in range(len(df_features) - 1):
            smoothed_close.append(smoothed_close[-1] * (1 + df_features.iloc[t, 0]))
        smoothed_close = pd.Series(smoothed_close)

        smoothed_open = (smoothed_close.shift(1) *
                         (df_features.iloc[:, 1] + 1).shift(1)).fillna(original_open)
        smoothed_high = smoothed_close * (1 + df_features.iloc[:, 2])
        smoothed_low = smoothed_close * (1 + df_features.iloc[:, 3])
        volume = np.exp(df_features.iloc[:, 4]) / smoothed_close

        underlying_close = underlying_close_caj * smoothed_close

        return pd.DataFrame({
            'open': smoothed_open,
            'adj_close': smoothed_close,
            'high': smoothed_high,
            'low': smoothed_low,
            'volume': volume,
            'close': underlying_close,
        })

    def _inverse_log_returns(self, df_features, initial_close, initial_open,
                             initial_high, initial_low, initial_volume):
        """Reconstruct OHLCV from log-return PV features (independent channels)."""
        lr = df_features.values if isinstance(df_features, pd.DataFrame) else np.asarray(df_features)

        close = initial_close * np.exp(np.cumsum(lr[:, 0]))
        open_ = initial_open * np.exp(np.cumsum(lr[:, 1]))
        high = initial_high * np.exp(np.cumsum(lr[:, 2]))
        low = initial_low * np.exp(np.cumsum(lr[:, 3]))
        volume = initial_volume * np.exp(np.cumsum(lr[:, 4]))

        # OHLC bar validity clamp
        max_oc = np.maximum(open_, close)
        min_oc = np.minimum(open_, close)
        high = np.maximum(high, max_oc)
        low = np.minimum(low, min_oc)
        volume = np.maximum(volume, 0)

        return pd.DataFrame({
            'open': open_,
            'adj_close': close,
            'high': high,
            'low': low,
            'volume': volume,
            'close': close,
        })

    # ------------------------------------------------------------------
    # Technical feature computation (identical to GRT_GAN)
    # ------------------------------------------------------------------

    def transform_data_to_feature(self, df):
        """Transform OHLCV DataFrame to ~158 Alpha158-style technical features."""

        def my_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        df = df.copy()
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        adj_close = df["adj_close"]
        volume = df["volume"]

        oc_vals = df[["open", "close"]].to_numpy()
        max_oc = np.maximum(oc_vals[:, 0], oc_vals[:, 1])
        min_oc = np.minimum(oc_vals[:, 0], oc_vals[:, 1])
        high_low_spread = high - low

        ret1 = close.pct_change(1)
        abs_ret1 = ret1.abs()
        pos_ret1 = ret1.clip(lower=0)
        log_volume = np.log(volume + 1)

        vchg1 = volume - volume.shift(1)
        abs_vchg1 = vchg1.abs()
        pos_vchg1 = vchg1.clip(lower=0)

        close_chg_ratio = close / close.shift(1)
        vol_chg_log = np.log(volume / volume.shift(1) + 1)

        retpos = (ret1 > 0).astype(float)
        retneg = (ret1 < 0).astype(float)

        features = {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj_close,
            "kmid": (close - open_) / close,
            "kmid2": (close - open_) / (high_low_spread + 1e-12),
            "klen": high_low_spread / open_,
            "kup": (high - max_oc) / open_,
            "kup2": (high - max_oc) / (high_low_spread + 1e-12),
            "klow": (min_oc - low) / open_,
            "klow2": (min_oc - low) / (high_low_spread + 1e-12),
            "ksft": (2.0 * close - high - low) / open_,
            "ksft2": (2.0 * close - high - low) / (high_low_spread + 1e-12),
            "log_volume": log_volume,
        }

        windows = [5, 10, 20, 30, 60]
        for w in windows:
            shifted_close = close.shift(w)
            features[f"roc_{w}"] = shifted_close / close
            features[f"beta_{w}"] = (shifted_close - close) / (w * close)

            shifted_ = close.shift(w)
            mn_ = low.where(low < shifted_, shifted_)
            mx_ = high.where(high > shifted_, shifted_)
            features[f"rsv_{w}"] = (close - mn_) / (mx_ - mn_ + 1e-12)

            c_rolling = close.rolling(w)
            features[f"ma_{w}"] = c_rolling.mean() / close
            features[f"std_{w}"] = c_rolling.std() / close
            features[f"max_{w}"] = c_rolling.max() / close
            features[f"min_{w}"] = c_rolling.min() / close
            features[f"qtlu_{w}"] = c_rolling.quantile(0.8) / close
            features[f"qtld_{w}"] = c_rolling.quantile(0.2) / close
            features[f"rank_{w}"] = c_rolling.apply(my_rank) / w

            h_rolling = high.rolling(w)
            l_rolling = low.rolling(w)
            h_argmax = h_rolling.apply(np.argmax)
            l_argmin = l_rolling.apply(np.argmin)
            features[f"imax_{w}"] = h_argmax / w
            features[f"imin_{w}"] = l_argmin / w
            features[f"imxd_{w}"] = (h_argmax - l_argmin) / w

            cntp = retpos.rolling(w).sum() / w
            cntn = retneg.rolling(w).sum() / w
            features[f"cntp_{w}"] = cntp
            features[f"cntn_{w}"] = cntn
            features[f"cntd_{w}"] = cntp - cntn

            # Match processor.cal_factor's pairwise rolling call so generated features stay on the training manifold.
            features[f"corr_{w}"] = close.rolling(w).corr(pairwise=log_volume.rolling(w))
            features[f"cord_{w}"] = close_chg_ratio.rolling(w).corr(pairwise=vol_chg_log.rolling(w))

            sum_abs = abs_ret1.rolling(w).sum()
            sum_pos = pos_ret1.rolling(w).sum()
            sump = sum_pos / (sum_abs + 1e-12)
            features[f"sump_{w}"] = sump
            features[f"sumn_{w}"] = 1.0 - sump
            features[f"sumd_{w}"] = 2.0 * sump - 1.0

            v_rolling = volume.rolling(w)
            features[f"vma_{w}"] = v_rolling.mean() / (volume + 1e-12)
            features[f"vstd_{w}"] = v_rolling.std() / (volume + 1e-12)

            shift_serie = np.abs(close / close.shift(1) - 1) * volume
            df1_ = shift_serie.rolling(w).std()
            df2_ = shift_serie.rolling(w).mean()
            features[f"wvma_{w}"] = df1_ / (df2_ + 1e-12)

            sum_abs_v = abs_vchg1.rolling(w).sum()
            sum_pos_v = pos_vchg1.rolling(w).sum()
            vsump = sum_pos_v / (sum_abs_v + 1e-12)
            features[f"vsump_{w}"] = vsump
            features[f"vsumn_{w}"] = 1.0 - vsump
            features[f"vsumd_{w}"] = 2.0 * vsump - 1.0

        feature_df = pd.DataFrame(features, index=df.index)
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.fillna(0, inplace=True)
        return feature_df[self.obs_features]

    # ------------------------------------------------------------------
    # Main entry point for DQN pipeline
    # ------------------------------------------------------------------

    def _lookup_start_index(self, timestamp):
        """Map a timestamp to the closest available generator window start."""
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        if timestamp in self._date_set:
            ts_idx = self._date_array.get_loc(timestamp)
        else:
            ts_idx = self._date_array.get_indexer([timestamp], method='nearest')[0]

        start_idx = ts_idx - self.seq_len // 2
        return max(0, min(start_idx, len(self.output_data) - 1))

    def _lookup_start_indices(self, timestamps):
        """Vectorized timestamp lookup for batched generation."""
        return np.asarray([self._lookup_start_index(timestamp) for timestamp in timestamps], dtype=np.int64)

    def _coerce_macro_epsilon(self, macro_epsilon):
        """Normalize macro noise into a shape compatible with the generator.

        The DQN pipeline supplies either a flat macro vector, an exact
        generator-length trajectory, or a shorter tail trajectory aligned to the
        observed RL window. Shorter trajectories are padded on the left so the
        most recent steps remain perturbed.
        """
        macro_epsilon = np.asarray(macro_epsilon, dtype=self.output_macro_data.dtype)

        if macro_epsilon.ndim == 1:
            if macro_epsilon.shape[0] != self.macro_dim:
                raise ValueError(
                    f"Expected macro epsilon with {self.macro_dim} features, "
                    f"got shape {macro_epsilon.shape}"
                )
            return macro_epsilon.reshape(1, 1, self.macro_dim)

        if macro_epsilon.ndim == 2:
            if macro_epsilon.shape[1] != self.macro_dim:
                raise ValueError(
                    f"Expected macro epsilon with trailing dim {self.macro_dim}, "
                    f"got shape {macro_epsilon.shape}"
                )
            if macro_epsilon.shape[0] > self.seq_len:
                raise ValueError(
                    f"Expected at most {self.seq_len} macro steps, got "
                    f"shape {macro_epsilon.shape}"
                )
            padded = np.zeros((1, self.seq_len, self.macro_dim), dtype=macro_epsilon.dtype)
            padded[:, -macro_epsilon.shape[0]:, :] = macro_epsilon.reshape(1, *macro_epsilon.shape)
            return padded

        if macro_epsilon.ndim == 3:
            if macro_epsilon.shape[0] != 1 or macro_epsilon.shape[2] != self.macro_dim:
                raise ValueError(
                    f"Expected macro epsilon shaped (1, steps, {self.macro_dim}), "
                    f"got {macro_epsilon.shape}"
                )
            if macro_epsilon.shape[1] > self.seq_len:
                raise ValueError(
                    f"Expected at most {self.seq_len} macro steps, got "
                    f"shape {macro_epsilon.shape}"
                )
            if macro_epsilon.shape[1] == self.seq_len:
                return macro_epsilon

            padded = np.zeros((1, self.seq_len, self.macro_dim), dtype=macro_epsilon.dtype)
            padded[:, -macro_epsilon.shape[1]:, :] = macro_epsilon
            return padded

        raise ValueError(
            f"Unsupported macro epsilon shape {macro_epsilon.shape}; expected 1D, 2D, or 3D input"
        )

    def _coerce_macro_epsilon_batch(self, macro_epsilon, batch_size):
        """Normalize batched macro noise into a generator-compatible shape."""
        macro_epsilon = np.asarray(macro_epsilon, dtype=self.output_macro_data.dtype)

        if batch_size == 1:
            return self._coerce_macro_epsilon(macro_epsilon)

        if macro_epsilon.ndim == 2:
            if macro_epsilon.shape != (batch_size, self.macro_dim):
                raise ValueError(
                    f"Expected batched macro epsilon shaped ({batch_size}, {self.macro_dim}), "
                    f"got {macro_epsilon.shape}"
                )
            return macro_epsilon.reshape(batch_size, 1, self.macro_dim)

        if macro_epsilon.ndim == 3:
            if macro_epsilon.shape[0] != batch_size or macro_epsilon.shape[2] != self.macro_dim:
                raise ValueError(
                    f"Expected batched macro epsilon shaped ({batch_size}, steps, {self.macro_dim}), "
                    f"got {macro_epsilon.shape}"
                )
            if macro_epsilon.shape[1] > self.seq_len:
                raise ValueError(
                    f"Expected at most {self.seq_len} macro steps, got shape {macro_epsilon.shape}"
                )
            if macro_epsilon.shape[1] == self.seq_len:
                return macro_epsilon

            padded = np.zeros((batch_size, self.seq_len, self.macro_dim), dtype=macro_epsilon.dtype)
            padded[:, -macro_epsilon.shape[1]:, :] = macro_epsilon
            return padded

        raise ValueError(
            f"Unsupported batched macro epsilon shape {macro_epsilon.shape}; expected 2D or 3D input"
        )

    def _postprocess_generated_sample(self, generated_all, start_idx, history_data, cal_factor, ticker_index):
        """Convert one generated PV trajectory back into downstream features."""
        pv_raw = generated_all[:, ticker_index * 5:(ticker_index + 1) * 5]
        history_ticker = history_data[:, ticker_index * 5:(ticker_index + 1) * 5]

        h_mean = history_ticker.mean(axis=0)
        h_std = history_ticker.std(axis=0)
        pv_denorm = (pv_raw * h_std) + h_mean

        pv_original = self._inverse_lambert_per_feature(pv_denorm, self.ticker_name)
        pv_df = pd.DataFrame(pv_original)

        close_init = self.original_close[start_idx, ticker_index]
        close_init = np.asarray(close_init)
        if close_init.ndim > 0:
            close_init = close_init[0]

        open_init = self.original_open[start_idx, ticker_index]

        if self.feature_method == "log_returns":
            high_init = self.original_high[start_idx, ticker_index]
            low_init = self.original_low[start_idx, ticker_index]
            vol_init = self.original_volume[start_idx, ticker_index]
            pv_data = self._inverse_log_returns(
                pv_df, close_init, open_init,
                high_init, low_init, vol_init)
        else:
            caj_ticker = cal_factor[ticker_index].reshape(-1)
            pv_data = self.transform_generated_pv_feature_to_data(
                pv_df, close_init, open_init, caj_ticker)

        return self.transform_data_to_feature(pv_data)

    def call_batch(self, timestamps, macro_epsilon):
        """Generate synthetic features for multiple timestamps in one model call.

        Parameters
        ----------
        timestamps : Sequence[str | pd.Timestamp]
            Target dates for generation.
        macro_epsilon : ndarray
            Batched macro perturbations. Accepts either one flat macro vector per
            sample with shape ``(batch, macro_dim)`` or a batched trajectory with
            shape ``(batch, steps, macro_dim)``.

        Returns
        -------
        list[pd.DataFrame]
            Technical features for each requested timestamp.
        """
        timestamps = list(timestamps)
        if not timestamps:
            return []

        batch_size = len(timestamps)
        start_indices = self._lookup_start_indices(timestamps)
        ticker_index = int(np.where(self.ticker_list == self.ticker_name)[0][0])

        real_data = self.output_data[start_indices, :, :]
        history_data = self.output_history_data[start_indices, :, :]
        cal_factor = self.output_adj_factor[start_indices, :, :]
        macro = self.output_macro_data[start_indices, :, :]
        T = self.time[start_indices]

        target_macro = macro + self._coerce_macro_epsilon_batch(macro_epsilon, batch_size)
        generated_batch = self.model_inference(real_data, target_macro, T)

        return [
            self._postprocess_generated_sample(
                generated_batch[i],
                start_indices[i],
                history_data[i],
                cal_factor[i],
                ticker_index,
            )
            for i in range(batch_size)
        ]

    def call(self, timestamp, macro_epsilon):
        """Generate synthetic features for one ticker at a given timestamp.

        Matches GRT_GAN's ``GeneratorAPI.call(timestamp, macro_epsilon)``
        interface exactly.

        Parameters
        ----------
        timestamp : str or pd.Timestamp
            Target date for generation.
        macro_epsilon : ndarray
            Noise to perturb macro conditioning. Accepts a flat macro vector,
            a full generator-length macro trajectory, or a shorter tail
            trajectory aligned to the observed RL window.

        Returns
        -------
        pd.DataFrame
            Technical features (columns = self.obs_features).
            Consumer slices ``result[-N:]`` for the DQN observation.
        """
        return self.call_batch([timestamp], macro_epsilon)[0]
