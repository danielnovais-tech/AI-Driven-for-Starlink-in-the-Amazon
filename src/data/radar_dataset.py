"""
Radar Dataset for loading meteorological rain-rate grids.

Assumes an HDF5 file with datasets:
    rain_rate – shape (time, lat, lon), rain rate in mm/h.
    lat       – 1-D latitude array.
    lon       – 1-D longitude array.
    time      – 1-D Unix timestamp array.

Data sources: CPTEC/INPE, GPM (NASA).
Reference: ITU-R P.838-3 – rain rate statistics for the Amazon region.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RadarDataset(Dataset):
    """
    PyTorch Dataset for gridded meteorological radar data.

    Rain-rate values are z-score normalised using the mean and standard
    deviation computed across the full dataset loaded into memory.

    Args:
        h5_file (str): Path to HDF5 file containing radar data.
    """

    def __init__(self, h5_file: str) -> None:
        with h5py.File(h5_file, "r") as f:
            self.rain_rate: np.ndarray = f["rain_rate"][:]   # (time, lat, lon)
            self.lat: np.ndarray = f["lat"][:]
            self.lon: np.ndarray = f["lon"][:]
            self.timestamps: np.ndarray = f["time"][:]       # Unix timestamps

        self.rain_mean = float(self.rain_rate.mean())
        self.rain_std = float(self.rain_rate.std())
        if self.rain_std == 0:
            self.rain_std = 1.0

    def __len__(self) -> int:
        return self.rain_rate.shape[0]

    def __getitem__(self, idx: int):
        rain_map = torch.tensor(self.rain_rate[idx], dtype=torch.float32)
        rain_map = (rain_map - self.rain_mean) / self.rain_std
        timestamp = int(self.timestamps[idx])
        return rain_map, timestamp
