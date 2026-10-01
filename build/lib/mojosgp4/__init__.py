"""A Mojo port of the SGP4 orbit propagator from the Python `sgp4` package.

The public surface mirrors `sgp4.api`, so most code that imports `sgp4` can
import `mojosgp4` instead:

    >>> from mojosgp4.api import Satrec, WGS72
    >>> sat = Satrec.twoline2rv(line1, line2, WGS72)
    >>> error, position, velocity = sat.sgp4(2451722.5, 0.0)
"""

__version__ = '0.1.0'

__all__ = (
    'Satrec', 'SatrecArray', 'SGP4_ERRORS', 'WGS72OLD', 'WGS72', 'WGS84',
    'accelerated', 'jday', 'days2mdhms', 'wgs72', 'wgs72old', 'wgs84',
)

from .api import (
    SGP4_ERRORS,
    WGS72,
    WGS72OLD,
    WGS84,
    Satrec,
    SatrecArray,
    accelerated,
    days2mdhms,
    jday,
)
from .earth_gravity import wgs72, wgs72old, wgs84
