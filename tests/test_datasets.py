"""
Tests for TelemetryDataset and RadarDataset using synthetic in-memory data.
"""

import io
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

# h5py is required; skip if not available
try:
    import h5py
    _H5PY = True
except ImportError:
    _H5PY = False


def _make_telemetry_csv(n_rows=50) -> str:
    """Write a synthetic telemetry CSV and return the file path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    )
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="s")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "satellite_id": ["SAT-1"] * n_rows,
            "snr": np.random.uniform(5, 25, n_rows),
            "csi_real": np.random.randn(n_rows),
            "csi_imag": np.random.randn(n_rows),
            "rssi": np.random.uniform(-90, -60, n_rows),
            "pos_x": np.random.uniform(-100, 100, n_rows),
            "pos_y": np.random.uniform(-100, 100, n_rows),
            "pos_z": np.random.uniform(500, 600, n_rows),
        }
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _make_radar_h5(n_time=20, lat=8, lon=8) -> str:
    """Write a synthetic radar HDF5 file and return the file path."""
    if not _H5PY:
        pytest.skip("h5py not installed")
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    with h5py.File(tmp.name, "w") as f:
        f.create_dataset("rain_rate", data=np.random.rand(n_time, lat, lon).astype(np.float32))
        f.create_dataset("lat", data=np.linspace(-10, 5, lat).astype(np.float32))
        f.create_dataset("lon", data=np.linspace(-70, -50, lon).astype(np.float32))
        f.create_dataset("time", data=np.arange(n_time, dtype=np.int64) * 300)
    return tmp.name


class TestTelemetryDataset:
    def setup_method(self):
        self.csv_path = _make_telemetry_csv(n_rows=30)

    def teardown_method(self):
        os.unlink(self.csv_path)

    def test_import(self):
        from data.telemetry_dataset import TelemetryDataset
        assert TelemetryDataset is not None

    def test_length(self):
        from data.telemetry_dataset import TelemetryDataset
        ds = TelemetryDataset(self.csv_path, lookback=5)
        assert len(ds) == 30 - 5

    def test_item_shapes(self):
        from data.telemetry_dataset import TelemetryDataset
        import torch
        ds = TelemetryDataset(self.csv_path, lookback=5)
        states, target = ds[0]
        assert states.shape == (5, 5)       # (lookback, features)
        assert target.shape == torch.Size([])  # scalar

    def test_normalisation(self):
        """SNR values in the window should be normalised (mean ~0, std ~1 after many samples)."""
        from data.telemetry_dataset import TelemetryDataset
        import torch
        ds = TelemetryDataset(self.csv_path, lookback=5)
        snr_values = []
        for i in range(len(ds)):
            s, _ = ds[i]
            snr_values.append(s[:, 0].tolist())
        flat = [v for row in snr_values for v in row]
        arr = np.array(flat)
        # Mean should be close to 0 and std close to 1 for normalised values
        assert abs(arr.mean()) < 1.5
        assert 0.5 < arr.std() < 1.5


@pytest.mark.skipif(not _H5PY, reason="h5py not installed")
class TestRadarDataset:
    def setup_method(self):
        self.h5_path = _make_radar_h5(n_time=20, lat=8, lon=8)

    def teardown_method(self):
        os.unlink(self.h5_path)

    def test_import(self):
        from data.radar_dataset import RadarDataset
        assert RadarDataset is not None

    def test_length(self):
        from data.radar_dataset import RadarDataset
        ds = RadarDataset(self.h5_path)
        assert len(ds) == 20

    def test_item_shapes(self):
        from data.radar_dataset import RadarDataset
        import torch
        ds = RadarDataset(self.h5_path)
        rain_map, timestamp = ds[0]
        assert rain_map.shape == (8, 8)
        assert isinstance(timestamp, int)
