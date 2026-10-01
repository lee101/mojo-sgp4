"""Public API, mirroring `sgp4.api`."""

__all__ = (
    'SGP4_ERRORS', 'Satrec', 'SatrecArray', 'WGS72OLD', 'WGS72', 'WGS84',
    'accelerated', 'jday', 'days2mdhms',
)

from .earth_gravity import wgs72, wgs72old, wgs84
from .functions import days2mdhms, jday
from .model import WGS72OLD, WGS72, WGS84, Satrec, SatrecArray

SGP4_ERRORS = {
    1: 'mean eccentricity is outside the range 0.0 to 1.0',
    2: 'nm is less than zero',
    3: 'perturbed eccentricity is outside the range 0.0 to 1.0',
    4: 'semilatus rectum is less than zero',
    5: '(error 5 no longer in use; it meant the satellite was underground)',
    6: 'mrt is less than 1.0 which indicates the satellite has decayed',
}

# Upstream sets this True when its C extension is importable.  The Mojo kernel
# is always in play here, so it is True unconditionally, and the three gravity
# models come from `earth_gravity` rather than a compiled `vallado_cpp`.
accelerated = True
