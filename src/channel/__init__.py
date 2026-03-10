"""Channel model sub-package."""

from .rain_attenuation import rain_specific_attenuation, slant_path_attenuation, ChannelModel
from .vegetation_attenuation import vegetation_specific_attenuation, vegetation_excess_attenuation
from .orbital_propagator import (
    SimplifiedPropagator,
    StarlinkConstellationTelemetry,
    geodetic_to_ecef,
    ecef_to_geodetic,
    elevation_angle,
    make_propagator,
    AMAZON_GS_LAT_DEG,
    AMAZON_GS_LON_DEG,
    AMAZON_GS_ALT_KM,
)

__all__ = [
    "rain_specific_attenuation",
    "slant_path_attenuation",
    "ChannelModel",
    "vegetation_specific_attenuation",
    "vegetation_excess_attenuation",
    "SimplifiedPropagator",
    "StarlinkConstellationTelemetry",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "elevation_angle",
    "make_propagator",
    "AMAZON_GS_LAT_DEG",
    "AMAZON_GS_LON_DEG",
    "AMAZON_GS_ALT_KM",
]
