"""The Satellite class.

`Satrec` mirrors upstream's `sgp4.model.Satrec`: the same `__slots__`, the same
attribute names, the same methods.  The numeric state lives in a block of
`float64` that the Mojo `Satrec` struct in `src/ported.mojo` reads and writes
in place, so attribute access is a view, not a copy.
"""

from __future__ import annotations

import numpy as np

from . import propagation
from ._lib import (
    NFIELDS,
    SATREC_FIELDS,
    SATREC_INT_FIELDS,
    STRIDE,
    _CHAR_FIELDS,
    _CHAR_VALUES,
    lib,
)
from .alpha5 import from_alpha5, to_alpha5
from .earth_gravity import gravity_constants

WGS72OLD = 0
WGS72 = 1
WGS84 = 2
minutes_per_day = 1440.

_INDEX = {name: i for i, name in enumerate(SATREC_FIELDS)}

# Field index -> the int64 view, so a numeric attribute costs one dict lookup
# rather than a walk of the name-based tables below.  `twoline2rv` writes
# about forty of them per element set.
_INT_INDEX = frozenset(_INDEX[name] for name in SATREC_INT_FIELDS)
_CHAR_INDEX = frozenset(_INDEX[name] for name in _CHAR_FIELDS)

# Attributes upstream keeps outside the numeric struct: strings and datetimes
# that the propagator never reads.
_EXTRA = ('satnum_str', 'classification', 'intldesg', 'ephtype', 'elnum',
          'revnum', 'epoch', 'error_message')

_SLOTS = ('_buf', '_ints', '_extra', 'array', '_addr')


def _dates(jd, fr):
    """C-contiguous float64 `jd` and `fr` of equal length, and that length.

    The kernel walks both arrays with a single index, so a shorter `fr` would
    be read past its end.  Upstream's `zip` stops at the shorter one and then
    fails to reshape; this rejects the mismatch outright.  Only 1-D input is
    accepted, so a 2-D array cannot pass the length check by flattening.
    """
    jd = np.ascontiguousarray(jd, dtype=np.float64)
    fr = np.ascontiguousarray(fr, dtype=np.float64)
    if jd.ndim > 1 or fr.ndim > 1:
        raise ValueError('jd and fr must be one-dimensional, got %d and %d'
                         % (jd.ndim, fr.ndim))
    if jd.size != fr.size:
        raise ValueError('jd and fr must be the same length, got %d and %d'
                         % (jd.size, fr.size))
    return jd, fr, jd.size


def _check_buf(buf):
    """The struct is a packed row of 8-byte fields, so the block has to be
    exactly `NFIELDS` contiguous `float64`.  Anything else would be read by
    the kernel as the wrong layout, or not at all."""
    if buf.dtype != np.float64:
        raise ValueError('expected a float64 block, got %s' % buf.dtype)
    if buf.shape != (NFIELDS,) or not buf.flags.c_contiguous:
        raise ValueError('expected a contiguous %d-element float64 block'
                         % NFIELDS)
    if not buf.flags.aligned or buf.ctypes.data % 8:
        raise ValueError('expected an 8-byte aligned block')


class Satrec:
    """A satellite, with the SGP4 variables that go with it."""

    __slots__ = _SLOTS

    def __init__(self, buf: np.ndarray | None = None):
        if buf is None:
            buf = np.zeros(NFIELDS, dtype=np.float64)
        _check_buf(buf)
        object.__setattr__(self, '_buf', buf)
        # the kernels address the block by pointer, and taking that pointer
        # costs more than the call it feeds
        object.__setattr__(self, '_addr', buf.ctypes.data)
        object.__setattr__(self, '_ints', buf.view(np.int64))
        object.__setattr__(self, '_extra', dict.fromkeys(_EXTRA))
        self._extra.update(satnum_str='00000', classification='U',
                           ephtype='', revnum=0, error_message=None)
        object.__setattr__(self, 'array', None)
        self.revnum = 0  # for consistency, since sgp4init() leaves this unset

    # -- attribute plumbing -------------------------------------------------
    def __getattr__(self, name):
        if name == 'no':
            return self.no_kozai
        if name == 'satnum':
            return from_alpha5(self.satnum_str)
        i = _INDEX.get(name)
        if i is not None:
            if i in _CHAR_INDEX:
                return _CHAR_FIELDS[name][
                    int(object.__getattribute__(self, '_ints')[i])]
            if i in _INT_INDEX:
                return int(object.__getattribute__(self, '_ints')[i])
            return float(object.__getattribute__(self, '_buf')[i])
        extra = object.__getattribute__(self, '_extra')
        if name in extra:
            return extra[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == '_buf':
            # the kernels address the block by pointer, and taking that
            # pointer costs more than the call it feeds
            _check_buf(value)
            object.__setattr__(self, '_buf', value)
            object.__setattr__(self, '_addr', value.ctypes.data)
            # the int64 view is a second window onto the same bytes, so it has
            # to move with the block or the `Int` fields would keep writing
            # into the buffer that was just replaced
            object.__setattr__(self, '_ints', value.view(np.int64))
            return
        if name in _SLOTS:
            object.__setattr__(self, name, value)
            return
        if name == 'satnum':
            self.satnum_str = to_alpha5(int(value))
            return
        i = _INDEX.get(name)
        if i is not None:
            if i in _CHAR_INDEX:
                # the struct holds these as small integers, so the write has
                # to go through the int64 view; writing the float would put
                # the bit pattern there instead of the value
                object.__getattribute__(self, '_ints')[i] = \
                    _CHAR_VALUES[name][value]
            elif i in _INT_INDEX:
                object.__getattribute__(self, '_ints')[i] = int(value)
            else:
                object.__getattribute__(self, '_buf')[i] = float(value)
            return
        if name in _EXTRA:
            object.__getattribute__(self, '_extra')[name] = value
            return
        raise AttributeError(name)

    def __delattr__(self, name):
        if name in _EXTRA:
            del self._extra[name]
            return
        raise AttributeError(name)

    def __repr__(self):
        return '<Satrec %s>' % self.satnum_str

    @property
    def _address(self) -> int:
        return self._addr

    # -- upstream API -------------------------------------------------------
    @classmethod
    def twoline2rv(cls, line1: str, line2: str, whichconst: int = WGS72):
        from .io import twoline2rv
        self = cls()
        twoline2rv(line1, line2, gravity_constants[whichconst], 'i', self)

        # Expose the same attribute types as the C++ code.
        self.ephtype = int(self.ephtype.strip() or '0')
        self.revnum = int(self.revnum)

        # Install a fancy split JD of the kind the C++ natively supports.
        # We rebuild it from the TLE year and day to maintain precision.
        year = self.epochyr
        days, fraction = divmod(self.epochdays, 1.0)
        self.jdsatepoch = year * 365 + (year - 1) // 4 + days + 1721044.5
        self.jdsatepochF = round(fraction, 8)  # exact number of digits in TLE

        # Remove the legacy datetime "epoch", which is not provided by
        # the C++ version of the object.
        del self.epoch

        # Undo my non-standard 4-digit year
        self.epochyr %= 100
        return self

    def sgp4init(self, whichconst, opsmode, satnum, epoch, bstar,
                 ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo):
        from .ext import invjday, jday

        whichconst = gravity_constants[whichconst]
        whole, fraction = divmod(epoch, 1.0)
        whole_jd = whole + 2433281.5

        # Go out on a limb: if `epoch` has no decimal digits past the 8
        # decimal places stored in a TLE, then assume the user is trying
        # to specify an exact decimal fraction.
        if round(epoch, 8) == epoch:
            fraction = round(fraction, 8)

        self.jdsatepoch = whole_jd
        self.jdsatepochF = fraction

        y, m, d, H, M, S = invjday(whole_jd)
        jan0 = jday(y, 1, 0, 0, 0, 0.0)
        self.epochyr = y % 100
        self.epochdays = whole_jd - jan0 + fraction

        self.classification = 'U'

        propagation.sgp4init(
            whichconst, opsmode, satnum, epoch, bstar, ndot, nddot,
            ecco, argpo, inclo, mo, no_kozai, nodeo, self)

    def sgp4(self, jd, fr):
        tsince = ((jd - self.jdsatepoch) * minutes_per_day +
                  (fr - self.jdsatepochF) * minutes_per_day)
        error, r, v = propagation._sgp4(self, tsince)
        return error, r, v

    def sgp4_tsince(self, tsince):
        return propagation._sgp4(self, tsince)

    def sgp4_array(self, jd, fr):
        """Compute positions and velocities for the times in a NumPy array.

        Given NumPy arrays ``jd`` and ``fr`` of the same length that
        supply the whole part and the fractional part of one or more
        Julian dates, return a tuple ``(e, r, v)`` of three vectors:

        * ``e``: nonzero for any dates that produced errors, 0 otherwise.
        * ``r``: position vectors in kilometers.
        * ``v``: velocity vectors in kilometers per second.
        """
        jd, fr, n = _dates(jd, fr)
        e = np.zeros(n, dtype=np.uint8)
        r = np.zeros((n, 3), dtype=np.float64)
        v = np.zeros((n, 3), dtype=np.float64)
        errval = np.zeros(1, dtype=np.float64)
        if n:
            lib().ms_sgp4_array(self._address, STRIDE, 1,
                                jd.ctypes.data, fr.ctypes.data, n,
                                e.ctypes.data, r.ctypes.data, v.ctypes.data,
                                errval.ctypes.data)
        # The kernel leaves `error` in the struct, but `error_message` is
        # upstream's Python-side bookkeeping: it reflects the last date.
        propagation.set_error_message(self, self.error,
                                      errval[0] if n else 0.0)
        return e, r, v


class SatrecArray:
    """A whole catalogue of satellites, propagated in one call."""

    __slots__ = ('_satrecs',)

    def __init__(self, satrecs):
        self._satrecs = list(satrecs)

    def __len__(self):
        return len(self._satrecs)

    def __getitem__(self, i):
        return self._satrecs[i]

    def sgp4(self, jd, fr):
        """Compute positions and velocities for the satellites in this array.

        Given NumPy scalars or arrays ``jd`` and ``fr`` supplying the
        whole part and the fractional part of one or more Julian dates,
        return a tuple ``(e, r, v)`` of three vectors that are each as
        long as ``jd`` and ``fr``:

        * ``e``: nonzero for any dates that produced errors, 0 otherwise.
        * ``r``: (x,y,z) position vector in kilometers.
        * ``v``: (dx,dy,dz) velocity vector in kilometers per second.
        """
        jd, fr, n = _dates(jd, fr)
        m = len(self._satrecs)
        e = np.zeros((m, n), dtype=np.uint8)
        r = np.zeros((m, n, 3), dtype=np.float64)
        v = np.zeros((m, n, 3), dtype=np.float64)
        errval = np.zeros(max(m, 1), dtype=np.float64)
        if m and n:
            block = np.concatenate([s._buf for s in self._satrecs])
            lib().ms_sgp4_array(block.ctypes.data, STRIDE, m,
                                jd.ctypes.data, fr.ctypes.data, n,
                                e.ctypes.data, r.ctypes.data, v.ctypes.data,
                                errval.ctypes.data)
            # The kernel advances each satellite in the block copy, so the
            # deep space integrator state and `error` have to go back into the
            # caller's Satrecs, as they do in upstream's SatrecArray.
            for i, sat in enumerate(self._satrecs):
                sat._buf[:] = block[i * NFIELDS:(i + 1) * NFIELDS]
                propagation.set_error_message(sat, sat.error, errval[i])
        return e, r, v


class Satellite:
    """The old Satellite object, for compatibility with sgp4 1.x."""

    # Only `_satrec` is stored; every upstream attribute proxies to it, so a
    # `Satellite` and the `Satrec` behind it can never drift apart.
    __slots__ = ('_satrec',)

    jdsatepochF = 0.0  # for compatibility with new Satrec; makes tests simpler

    def __init__(self):
        object.__setattr__(self, '_satrec', Satrec())

    _PROXIED = _EXTRA + tuple(_INDEX) + ('no', 'satnum', '_buf', '_extra',
                                        '_addr', '_address')

    def __getattr__(self, name):
        if name in self._PROXIED:
            return getattr(self._satrec, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ('_satrec', 'jdsatepochF'):
            object.__setattr__(self, name, value)
            return
        if name in self._PROXIED:
            setattr(self._satrec, name, value)
            return
        raise AttributeError(name)

    def propagate(self, year, month=1, day=1, hour=0, minute=0, second=0.0):
        """Return a position and velocity vector for a given date and time."""
        from .functions import jday
        j = jday(year, month, day, hour, minute, second)[0]
        m = (j - self.jdsatepoch) * minutes_per_day
        return propagation.sgp4(self._satrec, m)

    no = property(lambda self: self._satrec.no)
    satnum = property(lambda self: self._satrec.satnum)
