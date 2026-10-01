"""The SGP4 propagator, as a thin Python shell over `src/ported.mojo`.

Same names, argument order and defaults as `sgp4.propagation`.  The module
level `deg2rad`, `twopi` and `x2o3` constants are provided for callers that
reach for them, as upstream does.
"""

from __future__ import annotations

from math import pi

import numpy as np

from ._lib import OPSMODE_A, OPSMODE_I, lib

deg2rad = pi / 180.0
twopi = 2.0 * pi
x2o3 = 2.0 / 3.0

_nan = float('NaN')
false = (_nan, _nan, _nan)
true = True
# Module-level scratch, so a propagation allocates nothing.  That makes the
# three helpers here and `sgp4_array` non-reentrant, which is the same bargain
# upstream's C extension makes.  The addresses are taken once, at import: an
# `ndarray.ctypes.data` lookup costs more than the FFI call it feeds, and every
# exported kernel writes each slot it returns, so zeroing first is redundant.
_R = np.zeros(3, dtype=np.float64)
_V = np.zeros(3, dtype=np.float64)
_ERRVAL = np.zeros(1, dtype=np.float64)
_R_ADDR = _R.ctypes.data
_V_ADDR = _V.ctypes.data
_ERRVAL_ADDR = _ERRVAL.ctypes.data

# Upstream builds these sentences from the value that failed its range check.
_ERROR_MESSAGES = {
    1: 'mean eccentricity {0:f} not within range 0.0 <= e < 1.0',
    2: 'mean motion {0:f} is less than zero',
    3: 'perturbed eccentricity {0:f} not within range 0.0 <= e <= 1.0',
    4: 'semilatus rectum {0:f} is less than zero',
    6: 'mrt {0:f} is less than 1.0 indicating the satellite has decayed',
}


def set_error_message(satrec, error, errval):
    """Upstream's `satrec.error_message` bookkeeping, from a code and the value
    that failed its range check.  A call that did not fail clears the message,
    which is how upstream's `error_message = None` at the top of `sgp4` works.
    """
    template = _ERROR_MESSAGES.get(error)
    satrec.error_message = (None if template is None
                            else template.format(errval))
    return error


def gstime(jdut1: float) -> float:
    """Find the greenwich sidereal time for the given Julian date, in radians."""
    return lib().ms_gstime(jdut1)


def getgravconst(whichconst):
    """Return the (tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2) tuple.

    Accepts either the `model.WGS72` enum value or one of upstream's
    'wgs72old' / 'wgs72' / 'wgs84' names.
    """
    from .earth_gravity import getgravconst as _get
    if isinstance(whichconst, str):
        whichconst = {'wgs72old': 0, 'wgs72': 1, 'wgs84': 2}[whichconst]
    return _get(whichconst)


def sgp4init(whichconst, opsmode, satn, epoch, xbstar, xndot, xnddot, xecco,
             xargpo, xinclo, xmo, xno_kozai, xnodeo, satrec) -> bool:
    """Initialize variables for sgp4, in place on `satrec`.

    `whichconst` is the `model.WGS72`-style enum value (or an
    `earth_gravity.EarthGravity` instance), `opsmode` is 'a' or 'i', and
    `satn` is an int or an Alpha 5 string.
    """
    from . import earth_gravity as _eg
    from .alpha5 import from_alpha5, to_alpha5

    if not isinstance(whichconst, int):
        whichconst = _eg.gravity_constants.index(whichconst)
    if isinstance(opsmode, int):
        opsmode = 'a' if opsmode == OPSMODE_A else 'i'
    if not isinstance(satn, int):
        satn = int(satn) if str(satn).isdigit() else from_alpha5(str(satn))

    satrec.error = 0
    satrec.operationmode = opsmode
    satrec.satnum_str = to_alpha5(satn)
    satrec.classification = 'U'

    satrec.bstar = xbstar
    satrec.ndot = xndot
    satrec.nddot = xnddot
    satrec.ecco = xecco
    satrec.argpo = xargpo
    satrec.inclo = xinclo
    satrec.mo = xmo
    satrec.no_kozai = xno_kozai
    satrec.nodeo = xnodeo

    # single averaged mean elements
    satrec.am = 0.0
    satrec.em = 0.0
    satrec.im = 0.0
    satrec.Om = 0.0
    satrec.mm = 0.0
    satrec.nm = 0.0

    lib().ms_sgp4init(
        int(whichconst),
        OPSMODE_A if opsmode == 'a' else OPSMODE_I,
        int(satn), float(epoch),
        float(xbstar), float(xndot), float(xnddot), float(xecco),
        float(xargpo), float(xinclo), float(xmo), float(xno_kozai),
        float(xnodeo), satrec._address,
    )
    satrec.error_message = None
    return True


def _sgp4(satrec, tsince):
    """The `ms_sgp4` crossing, plus the error bookkeeping, without a layer of
    wrapper between them.  `Satrec.sgp4` and `Satrec.sgp4_tsince` call this
    directly; `sgp4` below is the upstream-shaped front end."""
    error = lib().ms_sgp4(satrec._address, tsince, _R_ADDR, _V_ADDR,
                          _ERRVAL_ADDR)
    # `ms_sgp4` returns `satrec.error`, which the kernel has already written
    # into the struct, so there is nothing to store back here.
    return set_error_message(satrec, error, _ERRVAL[0]), \
        tuple(_R.tolist()), tuple(_V.tolist())


def sgp4(satrec, tsince):
    """Propagate `satrec` to `tsince` minutes from epoch; return (r, v)."""
    return _sgp4(satrec, tsince)[1:]
