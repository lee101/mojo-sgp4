"""Various conveniences.

Mirrors `sgp4.conveniences`.
"""

from __future__ import annotations

import datetime as dt
from math import pi

from .functions import days2mdhms, jday


class _UTC(dt.tzinfo):
    'UTC'
    zero = dt.timedelta(0)

    def __repr__(self):
        return 'UTC'

    def dst(self, datetime):
        return self.zero

    def tzname(self, datetime):
        return 'UTC'

    def utcoffset(self, datetime):
        return self.zero


UTC = _UTC()


def jday_datetime(datetime):
    """Return two floats that, when added, produce the specified Julian date.

    The input is a native `datetime` object. Timezone of the input is
    converted internally to UTC.
    """
    u = datetime.astimezone(UTC)
    return jday(u.year, u.month, u.day, u.hour, u.minute,
                u.second + u.microsecond * 1e-6)


def sat_epoch_datetime(sat):
    """Return the epoch of the given satellite as a Python datetime."""
    year = sat.epochyr
    year += 1900 + (year < 57) * 100
    days = sat.epochdays
    month, day, hour, minute, second = days2mdhms(year, days)
    if month == 12 and day > 31:  # for that time the ISS epoch was "Dec 32"
        year += 1
        month = 1
        day -= 31
    second, fraction = divmod(second, 1.0)
    return dt.datetime(year, month, day, hour, minute, int(second),
                       int(fraction * 1e6), UTC)


_ATTR_MAXES = {'argpo': '2pi', 'inclo': 'pi', 'mo': '2pi', 'nodeo': '2pi'}
_MAX_VALUES = {'2pi': 2 * pi, 'pi': pi}


def check_satrec(sat):
    """Check whether satellite orbital elements are within range."""
    e = []

    for name, max_name in sorted(_ATTR_MAXES.items()):
        value = getattr(sat, name)
        if 0.0 <= value < _MAX_VALUES[max_name]:
            continue
        e.append('  {0} = {1:f} is outside the range 0 <= {0} < {2}\n'
                 .format(name, value, max_name))

    if e:
        raise ValueError('satellite parameters out of range:\n' + '\n'.join(e))


_ATTRIBUTES = [
    'Identification', 'satnum_str', 'satnum', 'classification', 'ephtype',
    'elnum', 'revnum', 'Orbital Elements', 'epochyr', 'epochdays', 'ndot',
    'nddot', 'bstar', 'inclo', 'nodeo', 'ecco', 'argpo', 'mo', 'no_kozai',
    'no', 'jdsatepoch', 'jdsatepochF', 'Computed Orbit Properties', 'a',
    'altp', 'alta', 'argpdot', 'gsto', 'mdot', 'nodedot', 'Propagator Mode',
    'operationmode', 'method', 'Result of Most Recent Propagation', 't',
    'error', 'Mean Elements From Most Recent Propagation', 'am', 'em', 'im',
    'Om', 'om', 'mm', 'nm', 'Gravity Model Parameters', 'tumin', 'xke', 'mu',
    'radiusearthkm', 'j2', 'j3', 'j4', 'j3oj2']


def dump_satrec(sat, sat2=None):
    """Yield lines that list the attributes of one or two satellites."""

    for item in _ATTRIBUTES:
        if item[0].isupper():
            yield '\n'
            yield '# -------- {0} --------\n'.format(item)
        else:
            name = item
            value = getattr(sat, item, '(not set)')
            line = '{0} = {1!r}\n'.format(item, value)
            if sat2 is not None:
                value2 = getattr(sat2, name, '(not set)')
                verdict = '==' if (value == value2) else '!='
                line = '{0:39} {1} {2!r}\n'.format(line[:-1], verdict, value2)
            yield line
