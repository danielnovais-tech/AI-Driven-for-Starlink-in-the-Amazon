"""Data pipeline sub-package."""

from .telemetry_dataset import TelemetryDataset
from .weather_forecast import WeatherForecast, SyntheticWeatherForecast, make_forecast
from .realtime_adapters import (
    CptecRadarAdapter,
    SpaceTrackTLEAdapter,
    NetworkTrafficAdapter,
)

# h5py-dependent modules
try:
    from .radar_dataset import RadarDataset
    _RADAR_DATASET_AVAILABLE = True
except ImportError:
    _RADAR_DATASET_AVAILABLE = False

__all__ = [
    "TelemetryDataset",
    "RadarDataset",
    "WeatherForecast",
    "SyntheticWeatherForecast",
    "make_forecast",
    "CptecRadarAdapter",
    "SpaceTrackTLEAdapter",
    "NetworkTrafficAdapter",
]
