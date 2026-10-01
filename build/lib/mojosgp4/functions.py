"""General-purpose routines.

Same names, argument order and defaults as `sgp4.functions`; the arithmetic
runs in the Mojo `jday()`, `days2mdhms()` and `_day_of_year_to_month_day()` in
`src/ported.mojo`.
"""

from __future__ import annotations

import numpy as np

from ._lib import lib

# The kernels write every slot they return, so this needs no zeroing, and its
# address is taken once: an `ndarray.ctypes.data` lookup costs more than the
# FFI call it feeds.
_SCRATCH = np.zeros(6, dtype=np.float64)
_SCRATCH_ADDR = _SCRATCH.ctypes.data


def jday(year: int, mon: int, day: int, hr: int, minute: int, sec: float):
    """Return two floats that, when added, produce the specified Julian date.

    The first float returned gives the date, while the second float
    provides an additional offset for the particular hour, minute, and
    second of that date.  Because the second float is much smaller in
    magnitude it can, unlike the first float, be accurate down to very
    small fractions of a second.

    >>> jd, fr = jday(2020, 2, 11, 13, 57, 0)
    >>> jd
    2458890.5
    >>> fr
    0.58125
    """
    lib().ms_jday(year, mon, day, hr, minute, float(sec), _SCRATCH_ADDR)
    out = _SCRATCH.tolist()
    return out[0], out[1]


def days2mdhms(year: int, days: float, round_to_microsecond=6):
    """Convert a floating point number of days into the year into date and time.

    Given the integer year plus the "day of the year" where 1.0 means
    the beginning of January 1, 2.0 means the beginning of January 2,
    and so forth, return the Gregorian calendar month, day, hour,
    minute, and floating point seconds.

    >>> days2mdhms(2000, 1.0)   # January 1
    (1, 1, 0, 0, 0.0)
    >>> days2mdhms(2000, 32.0)  # February 1
    (2, 1, 0, 0, 0.0)
    >>> days2mdhms(2000, 366.0)  # December 31, since 2000 was a leap year
    (12, 31, 0, 0, 0.0)

    The floating point seconds are rounded to an even number of
    microseconds if ``round_to_microsecond`` is true.
    """
    lib().ms_days2mdhms(year, days, int(round_to_microsecond or 0),
                        _SCRATCH_ADDR)
    out = _SCRATCH.tolist()
    return (int(out[0]), int(out[1]), int(out[2]), int(out[3]), out[4])


def _day_of_year_to_month_day(day_of_year: int, is_leap: bool):
    """Core logic for turning days into months, for easy testing."""
    lib().ms_day_of_year_to_month_day(int(day_of_year), int(bool(is_leap)),
                                     _SCRATCH_ADDR)
    out = _SCRATCH.tolist()
    return int(out[0]), int(out[1])
