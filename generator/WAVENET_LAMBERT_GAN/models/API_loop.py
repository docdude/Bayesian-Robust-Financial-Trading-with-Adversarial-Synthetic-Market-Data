"""Legacy singleton WaveNet inference API for loop-vs-batch validation.

This compatibility shim restores the pre-batch `GeneratorAPI.call(...)`
behavior without modifying the current batched implementation in `API.py`.
"""

import numpy as np
import pandas as pd

from .API import GeneratorAPI as BatchGeneratorAPI


class GeneratorAPI(BatchGeneratorAPI):
    def model_inference(self, real_data, target_macro, T):
        """Generate synthetic PV features using the legacy singleton path.

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
        noise = np.random.randn(1, half, self.latent_dim).astype(np.float32)
        z = np.concatenate([
            real_data[:, :half, :self.latent_dim],
            noise,
        ], axis=1).astype(np.float32)

        macro = target_macro.astype(np.float32)
        generated = self.generator([z, macro], training=False)
        return generated.numpy()

    def call(self, timestamp, macro_epsilon):
        """Generate synthetic features for one ticker at a given timestamp.

        Matches the pre-batch `GeneratorAPI.call(timestamp, macro_epsilon)`
        implementation exactly.
        """
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        if timestamp in self._date_set:
            ts_idx = self._date_array.get_loc(timestamp)
        else:
            ts_idx = self._date_array.get_indexer(
                [timestamp], method='nearest')[0]

        start_idx = ts_idx - self.seq_len // 2
        start_idx = max(0, min(start_idx, len(self.output_data) - 1))

        ticker_index = int(np.where(self.ticker_list == self.ticker_name)[0][0])

        real_data = self.output_data[start_idx:start_idx + 1, :, :]
        history_data = self.output_history_data[start_idx:start_idx + 1, :, :][0]
        cal_factor = self.output_adj_factor[start_idx:start_idx + 1, :, :]
        macro = self.output_macro_data[start_idx:start_idx + 1, :, :]
        T = self.time[start_idx:start_idx + 1]

        target_macro = macro + self._coerce_macro_epsilon(macro_epsilon)
        generated_all = self.model_inference(real_data, target_macro, T)

        generated_all = generated_all[0]
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
            caj_ticker = cal_factor[:, ticker_index].reshape(-1)
            pv_data = self.transform_generated_pv_feature_to_data(
                pv_df, close_init, open_init, caj_ticker)

        return self.transform_data_to_feature(pv_data)