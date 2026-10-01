"""Utility routines from "sgp4ext.cpp".

`mag`, `cross`, `dot`, `angle`, `newtonnu` and `invjday` run in Mojo; see
`src/ported.mojo`.  `rv2coe` is not ported, see the README.
"""

from __future__ import annotations

import numpy as np

from ._lib import lib
from .functions import days2mdhms  # noqa: F401  (re-exported, as upstream)

_VEC = np.zeros(3, dtype=np.float64)
_SCRATCH = np.zeros(6, dtype=np.float64)
_VEC_ADDR = _VEC.ctypes.data
_SCRATCH_ADDR = _SCRATCH.ctypes.data


def _vec(x) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError('expected a 3-element vector')
    return arr


def mag(x) -> float:
    """Find the magnitude of a vector."""
    v = _vec(x)
    return lib().ms_mag(v.ctypes.data)


def cross(vec1, vec2, outvec):
    """Cross two vectors, writing the result into `outvec`."""
    a, b = _vec(vec1), _vec(vec2)
    lib().ms_cross(a.ctypes.data, b.ctypes.data, _VEC_ADDR)
    outvec[0], outvec[1], outvec[2] = _VEC.tolist()


def dot(x, y) -> float:
    """Find the dot product of two vectors."""
    a, b = _vec(x), _vec(y)
    return lib().ms_dot(a.ctypes.data, b.ctypes.data)


def angle(vec1, vec2) -> float:
    """Calculate the angle between two vectors, in radians."""
    a, b = _vec(vec1), _vec(vec2)
    return lib().ms_angle(a.ctypes.data, b.ctypes.data)


def newtonnu(ecc: float, nu: float):
    """Given eccentric anomaly and eccentric anomaly, return true anomaly."""
    lib().ms_newtonnu(ecc, nu, _SCRATCH_ADDR)
    out = _SCRATCH.tolist()
    return out[0], out[1]


def jday(year: int, mon: int, day: int, hr: int, minute: int, sec: float) -> float:
    """The Julian date as a single float, the way `sgp4io.cpp` wants it.

    Note this is a different routine from `functions.jday`, which returns the
    two-float `(jd, fr)` pair; upstream keeps both and `io.py` uses this one.
    """
    return lib().ms_ext_jday(year, mon, day, hr, minute, float(sec))


def invjday(jd: float):
    """Return (year, mon, day, hr, minute, sec) for the given Julian date."""
    lib().ms_invjday(jd, _SCRATCH_ADDR)
    out = _SCRATCH.tolist()
    return (int(out[0]), int(out[1]), int(out[2]), int(out[3]), int(out[4]),
            out[5])
