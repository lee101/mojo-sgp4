"""Three earth-gravity models for use with SGP4.

The numbers come from the Mojo `getgravconst()` in `src/ported.mojo`, which
mirrors upstream's `sgp4.propagation.getgravconst()`.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from ._lib import WGS72OLD as _WGS72OLD, WGS72 as _WGS72, WGS84 as _WGS84, lib

EarthGravity = namedtuple(
    'EarthGravity',
    'tumin mu radiusearthkm xke j2 j3 j4 j3oj2',
    )

_SCRATCH = np.zeros(8, dtype=np.float64)
_SCRATCH_ADDR = _SCRATCH.ctypes.data


def getgravconst(whichconst: int) -> EarthGravity:
    lib().ms_getgravconst(whichconst, _SCRATCH_ADDR)
    return EarthGravity(*_SCRATCH.tolist())


wgs72old = getgravconst(_WGS72OLD)
wgs72 = getgravconst(_WGS72)
wgs84 = getgravconst(_WGS84)

# indexed by the `model.WGS72OLD` / `WGS72` / `WGS84` enum values
gravity_constants = wgs72old, wgs72, wgs84

__all__ = ('EarthGravity', 'getgravconst', 'gravity_constants',
           'wgs72old', 'wgs72', 'wgs84')
