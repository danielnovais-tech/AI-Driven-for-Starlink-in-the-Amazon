"""Channel model sub-package."""

from .rain_attenuation import rain_specific_attenuation, slant_path_attenuation, ChannelModel

__all__ = ["rain_specific_attenuation", "slant_path_attenuation", "ChannelModel"]
