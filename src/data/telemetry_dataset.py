"""
Telemetry Dataset for loading historical satellite telemetry.

Expects CSV files with columns:
    timestamp, satellite_id, snr, csi_real, csi_imag, rssi, pos_x, pos_y, pos_z

Reference: Starlink telemetry specifications, 3GPP NR channel state information.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TelemetryDataset(Dataset):
    """
    PyTorch Dataset for historical Starlink satellite telemetry.

    Constructs sliding windows of size ``lookback`` over the time series.
    The feature columns used are: snr, rssi, pos_x, pos_y, pos_z.
    SNR and RSSI are z-score normalised using statistics computed from the
    entire CSV file.  The regression target is the SNR value at time step
    ``idx + lookback`` (useful for auxiliary supervised pre-training).

    Args:
        csv_file (str): Path to CSV file with telemetry data.
        lookback (int): Number of historical time steps per sample.
    """

    FEATURE_COLS = ["snr", "rssi", "pos_x", "pos_y", "pos_z"]

    def __init__(self, csv_file: str, lookback: int = 10) -> None:
        self.data = pd.read_csv(csv_file, parse_dates=["timestamp"])
        self.lookback = lookback

        # Pre-compute normalisation statistics
        self.snr_mean = float(self.data["snr"].mean())
        self.snr_std = float(self.data["snr"].std())
        self.rssi_mean = float(self.data["rssi"].mean())
        self.rssi_std = float(self.data["rssi"].std())

        if self.snr_std == 0:
            self.snr_std = 1.0
        if self.rssi_std == 0:
            self.rssi_std = 1.0

    def __len__(self) -> int:
        return max(0, len(self.data) - self.lookback)

    def __getitem__(self, idx: int):
        window = self.data.iloc[idx : idx + self.lookback]
        states = torch.tensor(
            window[self.FEATURE_COLS].values, dtype=torch.float32
        )
        # Normalise SNR and RSSI columns (indices 0 and 1)
        states[:, 0] = (states[:, 0] - self.snr_mean) / self.snr_std
        states[:, 1] = (states[:, 1] - self.rssi_mean) / self.rssi_std

        target_snr = float(self.data.iloc[idx + self.lookback]["snr"])
        target = torch.tensor(
            (target_snr - self.snr_mean) / self.snr_std, dtype=torch.float32
        )
        return states, target
