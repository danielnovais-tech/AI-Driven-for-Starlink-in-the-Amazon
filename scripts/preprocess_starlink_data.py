#!/usr/bin/env python3
"""
Pre-processes raw Starlink telemetry (CSV) and aligns it with meteorological
radar grids and foliage density maps, producing a single aligned HDF5 file
ready for offline DRL training.

Expected CSV columns (telemetry):
    timestamp, sat_id, lat, lon, alt, snr, rssi, pos_x, pos_y, pos_z

Expected HDF5 schemas:

    radar_h5:
        time      (T,)       – Unix timestamps (float64)
        lat       (Rl,)      – Latitude grid (float32)
        lon       (Rl,)      – Longitude grid (float32)
        rain_rate (T, Rl, Rl) – Rain rate (mm/h, float32)

    foliage_h5:
        lat  (Fl,)           – Latitude grid (float32)
        lon  (Fl,)           – Longitude grid (float32)
        lai  (Fl, Fl)        – Leaf area index (float32); assumed static

Output HDF5 (output_h5):
    timestamp        (N,)   – Unix timestamps (int64)
    sat_id           (N,)   – Satellite ID strings
    snr              (N,)   – SNR (dB, float32)
    rssi             (N,)   – RSSI (dBm, float32)
    pos_x/y/z        (N,)   – ECEF position (km, float32)
    rain_rate        (N,)   – Rain rate interpolated at satellite location
    foliage_density  (N,)   – LAI interpolated at satellite location

Data sources:
    Radar: CPTEC/INPE, NASA GPM
    Foliage: INPE PRODES, MODIS LAI Product (MCD15A3H)

Usage:
    python scripts/preprocess_starlink_data.py \\
        --telemetry starlink_telemetry.csv \\
        --radar radar_amazon.h5 \\
        --foliage foliage_amazon.h5 \\
        --output aligned_data.h5

Requirements:
    pip install pandas numpy h5py scipy
"""

import argparse
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Core interpolation helper
# ---------------------------------------------------------------------------

def make_spatial_interpolator(
    grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> RegularGridInterpolator:
    """
    Build a 2-D bilinear interpolator over a (lat, lon) grid.

    Args:
        grid:      2-D array of shape (len(lat_grid), len(lon_grid)).
        lat_grid:  1-D ascending latitude array.
        lon_grid:  1-D ascending longitude array.

    Returns:
        Callable ``f(lat, lon) -> value`` that clamps to the grid boundary
        (``bounds_error=False, fill_value=None``).
    """
    interp = RegularGridInterpolator(
        (lat_grid, lon_grid),
        grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return interp


def interpolate_rain(
    rain_data: np.ndarray,
    rain_times: np.ndarray,
    rain_lats: np.ndarray,
    rain_lons: np.ndarray,
    query_time: float,
    query_lat: float,
    query_lon: float,
) -> float:
    """
    Nearest-time bilinear spatial interpolation of rain rate.

    Args:
        rain_data:   (T, Rl, Rl) rain rate array.
        rain_times:  (T,) Unix timestamps for each rain frame.
        rain_lats:   (Rl,) latitude array.
        rain_lons:   (Rl,) longitude array.
        query_time:  Unix timestamp of the telemetry row.
        query_lat:   Latitude of the satellite sub-point.
        query_lon:   Longitude of the satellite sub-point.

    Returns:
        Interpolated rain rate (mm/h).
    """
    t_idx = int(np.argmin(np.abs(rain_times - query_time)))
    interp = make_spatial_interpolator(rain_data[t_idx], rain_lats, rain_lons)
    value = float(interp([[query_lat, query_lon]])[0])
    return max(0.0, value)


def interpolate_foliage(
    foliage_data: np.ndarray,
    foliage_lats: np.ndarray,
    foliage_lons: np.ndarray,
    query_lat: float,
    query_lon: float,
) -> float:
    """
    Bilinear spatial interpolation of LAI (assumed static).

    Args:
        foliage_data: (Fl, Fl) LAI array.
        foliage_lats: (Fl,) latitude array.
        foliage_lons: (Fl,) longitude array.
        query_lat:    Latitude of the satellite sub-point.
        query_lon:    Longitude of the satellite sub-point.

    Returns:
        Interpolated LAI value (dimensionless, >= 0).
    """
    interp = make_spatial_interpolator(foliage_data, foliage_lats, foliage_lons)
    value = float(interp([[query_lat, query_lon]])[0])
    return max(0.0, value)


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    csv_path: str,
    radar_h5: str,
    foliage_h5: str,
    output_h5: str,
    verbose: bool = True,
) -> None:
    """
    Align telemetry, radar and foliage data and write a merged HDF5 file.

    Args:
        csv_path:   Path to Starlink telemetry CSV.
        radar_h5:   Path to meteorological radar HDF5.
        foliage_h5: Path to foliage/LAI HDF5.
        output_h5:  Destination HDF5 path.
        verbose:    Print progress to stdout.
    """
    if verbose:
        print(f"Loading telemetry from {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    if verbose:
        print(f"Loading radar data from {radar_h5} ...")
    with h5py.File(radar_h5, "r") as f:
        rain_times = f["time"][:].astype(np.float64)
        rain_lats = f["lat"][:].astype(np.float32)
        rain_lons = f["lon"][:].astype(np.float32)
        rain_data = f["rain_rate"][:].astype(np.float32)

    if verbose:
        print(f"Loading foliage data from {foliage_h5} ...")
    with h5py.File(foliage_h5, "r") as f:
        foliage_lats = f["lat"][:].astype(np.float32)
        foliage_lons = f["lon"][:].astype(np.float32)
        foliage_data = f["lai"][:].astype(np.float32)

    n = len(df)
    rain_interp = np.zeros(n, dtype=np.float32)
    foliage_interp = np.zeros(n, dtype=np.float32)

    if verbose:
        print(f"Interpolating rain and foliage for {n} rows ...")
    for idx in range(n):
        row = df.iloc[idx]
        t = row["timestamp"].timestamp()
        lat = float(row["lat"])
        lon = float(row["lon"])

        rain_interp[idx] = interpolate_rain(
            rain_data, rain_times, rain_lats, rain_lons, t, lat, lon
        )
        foliage_interp[idx] = interpolate_foliage(
            foliage_data, foliage_lats, foliage_lons, lat, lon
        )

    if verbose:
        print(f"Writing aligned dataset to {output_h5} ...")
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("timestamp", data=df["timestamp"].astype(np.int64).values)
        # Encode satellite IDs as variable-length ASCII strings
        sat_ids = df["sat_id"].astype(str).values
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("sat_id", data=sat_ids.astype("S"), dtype=dt)
        f.create_dataset("snr", data=df["snr"].values.astype(np.float32))
        f.create_dataset("rssi", data=df["rssi"].values.astype(np.float32))
        f.create_dataset("pos_x", data=df["pos_x"].values.astype(np.float32))
        f.create_dataset("pos_y", data=df["pos_y"].values.astype(np.float32))
        f.create_dataset("pos_z", data=df["pos_z"].values.astype(np.float32))
        # Combined position matrix expected by OfflineLEOEnv
        pos = np.stack(
            [
                df["pos_x"].values.astype(np.float32),
                df["pos_y"].values.astype(np.float32),
                df["pos_z"].values.astype(np.float32),
            ],
            axis=1,
        )
        f.create_dataset("pos", data=pos)
        f.create_dataset("rain_rate", data=rain_interp)
        f.create_dataset("foliage_density", data=foliage_interp)

    if verbose:
        print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Align Starlink telemetry with radar and foliage data."
    )
    parser.add_argument("--telemetry", required=True, help="Input telemetry CSV path.")
    parser.add_argument("--radar", required=True, help="Input radar HDF5 path.")
    parser.add_argument("--foliage", required=True, help="Input foliage HDF5 path.")
    parser.add_argument("--output", required=True, help="Output aligned HDF5 path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    preprocess(
        csv_path=args.telemetry,
        radar_h5=args.radar,
        foliage_h5=args.foliage,
        output_h5=args.output,
        verbose=not args.quiet,
    )
