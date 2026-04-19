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
    )
    feature_df = api.call(timestamp='2020-01-02', macro_epsilon=noise)
"""

import os
import pickle
import time
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GeneratorAPI:
    """Inference wrapper for a trained WaveNet Lambert GAN (GRT_GAN-compatible)."""

    def __init__(self, model_path, ticker_name, obs_features, temporal_features,
                 feature_method="derived"):
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
        """
        self.feature_method = feature_method
        self.ticker_name = ticker_name
        self.obs_features = obs_features
        self.temporal_features = temporal_features

        # -- Load NPY data (same layout as GRT_GAN) --
        data_dir = os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'datasets', 'output_data_lambert')
        data_dir = os.path.normpath(data_dir)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

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

        # Lambert inverse transform params: {ticker: [(scaler1, gauss, scaler2) × 5]}
        with open(os.path.join(data_dir, 'lambert_fit_params.pkl'), 'rb') as f:
            self.lambert_fit_params = pickle.load(f)

        # Date index for fast timestamp lookup
        self._date_array = pd.DatetimeIndex(self.output_starting_date[:, 0])
        self._date_set = set(self._date_array)
        print(f"GAN date range: {self._date_array[0]} → {self._date_array[-1]}"
              f"  ({len(self._date_array)} dates)")

        # -- Load model config & generator --
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        with open(os.path.join(model_path, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)

        self.seq_len = self.config['seq_len']
        self.latent_dim = self.config['latent_dim']
        self.feature_dim = self.config['feature_dim']
        self.macro_dim = self.config['macro_dim']

        self.generator = tf.keras.models.load_model(
            os.path.join(model_path, 'generator.keras'))

        print(f"WaveNet Lambert GAN loaded: ticker={ticker_name}, "
              f"seq_len={self.seq_len}, feature_dim={self.feature_dim}, "
              f"macro_dim={self.macro_dim}")

    # ------------------------------------------------------------------
    # Model inference (matches GRT_GAN's half-real / half-noise strategy)
    # ------------------------------------------------------------------

    def model_inference(self, real_data, target_macro, T):
        """Generate synthetic PV features using half-real / half-noise.

        Parameters
        ----------
        real_data : ndarray (1, seq_len, feature_dim)
        target_macro : ndarray (1, seq_len, macro_dim)
        T : ndarray (1,)  — sequence length (unused by WaveNet, kept for API compat)

        Returns
        -------
        ndarray (1, seq_len, feature_dim)
        """
        half = self.seq_len // 2
        # Half-real / half-noise latent (TimeGAN strategy)
        noise = np.random.randn(1, half, self.latent_dim).astype(np.float32)
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

        oc_vals = df[["open", "close"]].to_numpy()
        df["max_oc"] = np.maximum(oc_vals[:, 0], oc_vals[:, 1])
        df["min_oc"] = np.minimum(oc_vals[:, 0], oc_vals[:, 1])

        df["kmid"]  = (df["close"] - df["open"]) / df["close"]
        df["kmid2"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)
        df["klen"]  = (df["high"] - df["low"]) / df["open"]
        df["kup"]   = (df["high"] - df["max_oc"]) / df["open"]
        df["kup2"]  = (df["high"] - df["max_oc"]) / (df["high"] - df["low"] + 1e-12)
        df["klow"]  = (df["min_oc"] - df["low"]) / df["open"]
        df["klow2"] = (df["min_oc"] - df["low"]) / (df["high"] - df["low"] + 1e-12)
        df["ksft"]  = (2.0 * df["close"] - df["high"] - df["low"]) / df["open"]
        df["ksft2"] = (2.0 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"] + 1e-12)

        df["ret1"]     = df["close"].pct_change(1)
        df["abs_ret1"] = df["ret1"].abs()
        df["pos_ret1"] = df["ret1"].clip(lower=0)
        df["log_volume"] = np.log(df["volume"] + 1)

        df["vchg1"]     = df["volume"] - df["volume"].shift(1)
        df["abs_vchg1"] = df["vchg1"].abs()
        df["pos_vchg1"] = df["vchg1"].clip(lower=0)

        df["close_chg_ratio"] = df["close"] / df["close"].shift(1)
        df["vol_chg_log"]     = np.log(df["volume"] / df["volume"].shift(1) + 1)

        retpos = (df["ret1"] > 0).astype(float)
        retneg = (df["ret1"] < 0).astype(float)

        windows = [5, 10, 20, 30, 60]
        for w in windows:
            shifted_close = df["close"].shift(w)
            df[f"roc_{w}"]  = shifted_close / df["close"]
            df[f"beta_{w}"] = (shifted_close - df["close"]) / (w * df["close"])

            shifted_ = df["close"].shift(w)
            mn_ = df["low"].where(df["low"] < shifted_, shifted_)
            mx_ = df["high"].where(df["high"] > shifted_, shifted_)
            df[f"rsv_{w}"] = (df["close"] - mn_) / (mx_ - mn_ + 1e-12)

            c_rolling = df["close"].rolling(w)
            df[f"ma_{w}"]   = c_rolling.mean() / df["close"]
            df[f"std_{w}"]  = c_rolling.std()  / df["close"]
            df[f"max_{w}"]  = c_rolling.max()  / df["close"]
            df[f"min_{w}"]  = c_rolling.min()  / df["close"]
            df[f"qtlu_{w}"] = c_rolling.quantile(0.8) / df["close"]
            df[f"qtld_{w}"] = c_rolling.quantile(0.2) / df["close"]
            df[f"rank_{w}"] = c_rolling.apply(my_rank) / w

            h_rolling = df["high"].rolling(w)
            l_rolling = df["low"].rolling(w)
            df[f"imax_{w}"] = h_rolling.apply(np.argmax) / w
            df[f"imin_{w}"] = l_rolling.apply(np.argmin) / w
            df[f"imxd_{w}"] = (h_rolling.apply(np.argmax)
                               - l_rolling.apply(np.argmin)) / w

            df[f"cntp_{w}"] = retpos.rolling(w).sum() / w
            df[f"cntn_{w}"] = retneg.rolling(w).sum() / w
            df[f"cntd_{w}"] = df[f"cntp_{w}"] - df[f"cntn_{w}"]

            df[f"corr_{w}"] = df["close"].rolling(w).corr(df["log_volume"])
            df[f"cord_{w}"] = df["close_chg_ratio"].rolling(w).corr(df["vol_chg_log"])

            sum_abs = df["abs_ret1"].rolling(w).sum()
            sum_pos = df["pos_ret1"].rolling(w).sum()
            df[f"sump_{w}"] = sum_pos / (sum_abs + 1e-12)
            df[f"sumn_{w}"] = 1.0 - df[f"sump_{w}"]
            df[f"sumd_{w}"] = 2.0 * df[f"sump_{w}"] - 1.0

            v_rolling = df["volume"].rolling(w)
            df[f"vma_{w}"]  = v_rolling.mean() / (df["volume"] + 1e-12)
            df[f"vstd_{w}"] = v_rolling.std()  / (df["volume"] + 1e-12)

            shift_serie = np.abs(df["close"] / df["close"].shift(1) - 1) * df["volume"]
            df[f"wvma_{w}"] = shift_serie.rolling(w).std() / (
                shift_serie.rolling(w).mean() + 1e-12)

            sum_abs_v = df["abs_vchg1"].rolling(w).sum()
            sum_pos_v = df["pos_vchg1"].rolling(w).sum()
            df[f"vsump_{w}"] = sum_pos_v / (sum_abs_v + 1e-12)
            df[f"vsumn_{w}"] = 1.0 - df[f"vsump_{w}"]
            df[f"vsumd_{w}"] = 2.0 * df[f"vsump_{w}"] - 1.0

        df.drop(columns=[
            "max_oc", "min_oc",
            "ret1", "abs_ret1", "pos_ret1",
            "vchg1", "abs_vchg1", "pos_vchg1",
            "volume",
            "close_chg_ratio",
            "vol_chg_log",
        ], inplace=True, errors="ignore")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        df = df[self.obs_features]
        return df

    # ------------------------------------------------------------------
    # Main entry point for DQN pipeline
    # ------------------------------------------------------------------

    def call(self, timestamp, macro_epsilon):
        """Generate synthetic features for one ticker at a given timestamp.

        Matches GRT_GAN's ``GeneratorAPI.call(timestamp, macro_epsilon)``
        interface exactly.

        Parameters
        ----------
        timestamp : str or pd.Timestamp
            Target date for generation.
        macro_epsilon : ndarray
            Noise to perturb macro conditioning.

        Returns
        -------
        pd.DataFrame
            Technical features (columns = self.obs_features).
            Consumer slices ``result[-N:]`` for the DQN observation.
        """
        # NOTE: do NOT reseed RNGs here. Reseeding on every call with a fixed
        # value makes every adversarial rollout identical, which defeats the
        # point of generator-based data augmentation. Global seed is set once
        # by the RL trainer; each ``call()`` must draw fresh noise.

        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        # Timestamp → window index
        if timestamp in self._date_set:
            ts_idx = self._date_array.get_loc(timestamp)
        else:
            ts_idx = self._date_array.get_indexer(
                [timestamp], method='nearest')[0]

        start_idx = ts_idx - self.seq_len // 2
        start_idx = max(0, min(start_idx, len(self.output_data) - 1))

        # Ticker → column index
        ticker_index = int(np.where(self.ticker_list == self.ticker_name)[0][0])

        # Slice data arrays at this window index
        real_data = self.output_data[start_idx:start_idx + 1, :, :]
        history_data = self.output_history_data[start_idx:start_idx + 1, :, :][0]
        cal_factor = self.output_adj_factor[start_idx:start_idx + 1, :, :]
        macro = self.output_macro_data[start_idx:start_idx + 1, :, :]
        T = self.time[start_idx:start_idx + 1]

        target_macro = macro + macro_epsilon

        # -- Generator inference --
        generated_all = self.model_inference(real_data, target_macro, T)

        # Extract this ticker's 5 features
        generated_all = generated_all[0]  # (seq_len, feature_dim)
        pv_raw = generated_all[:, ticker_index * 5:(ticker_index + 1) * 5]
        history_ticker = history_data[:, ticker_index * 5:(ticker_index + 1) * 5]

        # Denormalize with history mean/std
        h_mean = history_ticker.mean(axis=0)
        h_std = history_ticker.std(axis=0)
        pv_denorm = (pv_raw * h_std) + h_mean

        # Inverse Lambert transform (per-feature scaler2⁻¹ → gauss⁻¹ → scaler1⁻¹)
        pv_original = self._inverse_lambert_per_feature(pv_denorm, self.ticker_name)
        pv_df = pd.DataFrame(pv_original)

        # -- Reconstruct OHLCV --
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
            caj_ticker = cal_factor[:, ticker_index].reshape(-1)
            pv_data = self.transform_generated_pv_feature_to_data(
                pv_df, close_init, open_init, caj_ticker)

        # -- Compute technical features --
        feature = self.transform_data_to_feature(pv_data)
        return feature
