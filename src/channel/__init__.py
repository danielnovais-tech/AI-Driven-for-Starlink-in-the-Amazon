"""Channel model sub-package."""

from .rain_attenuation import rain_specific_attenuation, slant_path_attenuation, ChannelModel
from .vegetation_attenuation import vegetation_specific_attenuation, vegetation_excess_attenuation

__all__ = [
    "rain_specific_attenuation",
    "slant_path_attenuation",
    "ChannelModel",
    "vegetation_specific_attenuation",
    "vegetation_excess_attenuation",
]
