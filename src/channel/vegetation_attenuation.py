"""
Vegetation (foliage) attenuation model based on ITU-R P.833-9.

Provides the excess attenuation due to the forest canopy for Earth-space
links, supplementing the rain attenuation computed by ITU-R P.838-3.

The model is based on the exponential decay approach described in
ITU-R P.833-9 Section 3 for in-leaf conditions.

Reference:
    ITU-R P.833-9 (09/2016) – "Attenuation in vegetation"
    https://www.itu.int/rec/R-REC-P.833
"""

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ITU-R P.833-9 vegetation-type coefficients (a, b) for γ = a · f^b (dB/m)
# Calibrated for frequencies above 1 GHz.
# ---------------------------------------------------------------------------
_VEG_COEFFICIENTS = {
    "tropical":   (0.39, 0.39),   # Dense tropical forest (Amazon)
    "temperate":  (0.25, 0.30),   # Temperate deciduous forest (in-leaf)
    "light":      (0.15, 0.20),   # Sparse vegetation / scrubland
}

# Maximum recommended attenuation to avoid physically unrealistic values
_MAX_VEG_ATTENUATION_DB = 30.0


def vegetation_specific_attenuation(f: float, vegetation_type: str = "tropical") -> float:
    """
    Compute specific vegetation attenuation γ (dB/m) per ITU-R P.833-9.

    Uses the power-law relation:  γ = a · f^b

    Args:
        f:               Carrier frequency in GHz (> 0).
        vegetation_type: Vegetation class: ``'tropical'``, ``'temperate'``,
                         or ``'light'``.

    Returns:
        Specific vegetation attenuation γ in dB/m.

    Raises:
        ValueError: If ``vegetation_type`` is not recognised.
    """
    if f <= 0:
        return 0.0
    if vegetation_type not in _VEG_COEFFICIENTS:
        raise ValueError(
            f"Unknown vegetation type '{vegetation_type}'. "
            f"Choose from: {list(_VEG_COEFFICIENTS)}"
        )
    a, b = _VEG_COEFFICIENTS[vegetation_type]
    return a * (f ** b)


def vegetation_excess_attenuation(
    f: float,
    elevation: float,
    depth: float = 15.0,
    vegetation_type: str = "tropical",
) -> float:
    """
    Compute total excess attenuation (dB) due to foliage along the slant path.

    The forest canopy is modelled as a horizontal slab of uniform thickness
    ``depth`` (metres).  The oblique path length through the slab is
    ``depth / sin(elevation)``.

    Args:
        f:               Carrier frequency in GHz.
        elevation:       Satellite elevation angle (degrees).  Must be > 0
                         for a satellite link; returns 0 for elevation ≤ 0.
        depth:           Effective vertical depth of the vegetation layer (m).
                         Typical value for tropical forest: 15 m.
        vegetation_type: Vegetation class (see :func:`vegetation_specific_attenuation`).

    Returns:
        Total excess attenuation A_veg in dB, capped at
        :data:`_MAX_VEG_ATTENUATION_DB` (30 dB).

    Reference:
        ITU-R P.833-9, Section 3 – Terrestrial and Earth-space paths through
        vegetation.
    """
    if elevation <= 0.0 or depth <= 0.0:
        return 0.0

    gamma = vegetation_specific_attenuation(f, vegetation_type)
    theta_rad = math.radians(elevation)
    sin_theta = math.sin(theta_rad)
    # Guard against near-zero sin (very low elevation angles already excluded above)
    if sin_theta < 1e-6:
        return _MAX_VEG_ATTENUATION_DB

    l_veg = depth / sin_theta        # slant path length through foliage (m)
    a_veg = gamma * l_veg            # dB
    return min(a_veg, _MAX_VEG_ATTENUATION_DB)
