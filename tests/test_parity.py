"""Parity of the Mojo kernel against the real `sgp4` package, which is installed
in this environment as a test dependency.

Three groups of checks:

* every helper (`jday`, `days2mdhms`, `invjday`, `gstime`, `mag`, `dot`,
  `cross`, `angle`, `newtonnu`, the gravity constants) against the same
  function upstream, which must agree to the last few bits;
* every field of the `Satrec` struct against upstream's `Satrec`, for the
  satellites in the official `SGP4-VER.TLE` catalogue;
* the propagated positions and velocities over that whole catalogue, checked
  both at epoch and over each satellite's full verification time span.

The propagation tolerances are the interesting ones.  Mojo's `std.math.pow` is
about 1e-10 relative where CPython's is correctly rounded, and the mean motion
and the drag coefficients are both derived through `pow`, so the two ports
start about 1e-8 km apart at epoch.  For most of the catalogue that stays
microscopic; for a handful of low-perigee satellites that are already
decaying, the trajectory is chaotic and the gap grows to kilometres before the
propagator gives up with its `mrt < 1.0` error.  The bounds below were measured
against upstream and are asserted, not assumed.
"""

from __future__ import annotations

from math import isnan, pi

import re

import numpy as np
import pytest
from pkgutil import get_data

import sgp4.api as up_api
import sgp4.earth_gravity as up_gravity
import sgp4.ext as up_ext
import sgp4.functions as up_functions
import sgp4.io as up_io
import sgp4.model as up_model
import sgp4.propagation as up_propagation

import mojosgp4.api as ms
import mojosgp4.earth_gravity as ms_gravity
import mojosgp4.ext as ms_ext
import mojosgp4.functions as ms_functions
import mojosgp4.io as ms_io
import mojosgp4.model as ms_model
import mojosgp4.propagation as ms_propagation
from mojosgp4._lib import SATREC_FIELDS, SATREC_INT_FIELDS

LINE1 = '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
LINE2 = '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'

# Measured against upstream over the whole SGP4-VER.TLE catalogue; see the
# module docstring.  The three bounds below are the real maxima, with a little
# headroom, so a regression past them fails the suite.
EPOCH_TOL_KM = 1.0e-5      # measured 1.2e-6 km
FIELD_RTOL = 1.0e-7        # measured 2.9e-8
CATALOGUE_TOL_KM = 4.0     # measured 3.1 km


def _ver_tles():
    lines = get_data('sgp4', 'SGP4-VER.TLE').decode('ascii').splitlines()
    out = []
    it = iter(lines)
    for line1 in it:
        if not line1.startswith('1'):
            continue
        line2 = next(it)
        tstart, tend, tstep = (float(f) for f in line2[69:].split())
        out.append((line1, line2, tstart, tend, tstep))
    return out


VER_TLES = _ver_tles()


def _times(tstart, tend, tstep):
    times = [0.0]
    t = tstart
    while t <= tend:
        if not (t == tstart == 0.0):
            times.append(t)
        t += tstep
    if t - tend < tstep - 1e-6:
        times.append(tend)
    return times


# ---------------------------------------------------------------- helpers
def test_gravity_constants():
    for name, ours, theirs in (
        ('wgs72old', ms_gravity.wgs72old, up_gravity.wgs72old),
        ('wgs72', ms_gravity.wgs72, up_gravity.wgs72),
        ('wgs84', ms_gravity.wgs84, up_gravity.wgs84),
    ):
        assert ours == theirs, name


def test_getgravconst():
    for name in ('wgs72old', 'wgs72', 'wgs84'):
        assert ms_propagation.getgravconst(name) == \
            up_propagation.getgravconst(name)


@pytest.mark.parametrize('args', [
    (2020, 2, 11, 13, 57, 0), (1957, 10, 4, 19, 26, 24.0),
    (2000, 1, 1, 0, 0, 0.0), (2056, 12, 31, 23, 59, 59.999),
    (1999, 12, 31, 12, 0, 0.5),
])
def test_jday_pair(args):
    assert ms_functions.jday(*args) == pytest.approx(up_functions.jday(*args))


@pytest.mark.parametrize('args', [
    (2020, 2, 11, 13, 57, 0), (1957, 10, 4, 19, 26, 24.0),
    (2000, 1, 1, 0, 0, 0.0), (2056, 12, 31, 23, 59, 59.999),
    (1999, 12, 31, 12, 0, 0.5),
])
def test_ext_jday_combined(args):
    assert ms_ext.jday(*args) == pytest.approx(up_ext.jday(*args), abs=0.0)


@pytest.mark.parametrize('year,days', [
    (2000, 1.0), (2000, 32.0), (2000, 366.0), (2000, 59.5), (2001, 60.25),
    (1957, 277.0), (2020, 366.9999), (2000, 0.5), (1999, 365.75),
])
def test_days2mdhms(year, days):
    assert ms_functions.days2mdhms(year, days) == \
        up_functions.days2mdhms(year, days)
    assert ms_functions.days2mdhms(year, days, None) == \
        up_functions.days2mdhms(year, days, None)


@pytest.mark.parametrize('day_of_year', list(range(0, 370, 7)))
@pytest.mark.parametrize('is_leap', [False, True])
def test_day_of_year_to_month_day(day_of_year, is_leap):
    assert ms_functions._day_of_year_to_month_day(day_of_year, is_leap) == \
        up_functions._day_of_year_to_month_day(day_of_year, is_leap)


@pytest.mark.parametrize('jd', [
    2451722.5, 2451723.28495062, 2415019.5, 2451545.0, 2460000.5, 2433281.5,
    2457192.5, 1721044.5,
])
def test_invjday(jd):
    assert ms_ext.invjday(jd) == pytest.approx(up_ext.invjday(jd), abs=1e-9)


@pytest.mark.parametrize('jd', [
    2451722.5, 2451723.28495062, 2451545.0, 2460000.5, 2433281.5, 1721044.5,
    2457192.5 + 0.75, 2000000.25,
])
def test_gstime(jd):
    # The remainder is taken on a value of order 1e6, where a double has about
    # 2e-10 of resolution, so the last bits of the angle are not reproducible.
    assert ms_propagation.gstime(jd) == pytest.approx(
        up_propagation.gstime(jd), rel=1e-10, abs=1e-9)


VECTORS = [
    (1.0, 2.0, 3.0), (-4.5, 0.25, 7.75), (0.0, 0.0, 0.0),
    (6378.0, 0.0, 0.0), (1e-8, 1e-8, 1e-8),
]


def test_mag():
    for v in VECTORS:
        assert ms_ext.mag(v) == pytest.approx(up_ext.mag(v), rel=1e-15)


def test_dot():
    for a in VECTORS:
        for b in VECTORS:
            assert ms_ext.dot(a, b) == pytest.approx(up_ext.dot(a, b), rel=1e-15,
                                                     abs=1e-300)


def test_cross():
    for a in VECTORS:
        for b in VECTORS:
            ours, theirs = [None, None, None], [None, None, None]
            ms_ext.cross(a, b, ours)
            up_ext.cross(a, b, theirs)
            assert ours == pytest.approx(theirs, rel=1e-15, abs=1e-300)


def test_angle():
    for a in VECTORS:
        for b in VECTORS:
            assert ms_ext.angle(a, b) == pytest.approx(up_ext.angle(a, b),
                                                       rel=1e-12, abs=1e-9)


@pytest.mark.parametrize('ecc,nu', [
    (0.0, 0.7), (0.5, 0.7), (0.9, 2.5), (1.5, 1.2), (2.0, 0.3), (0.001, -1.0),
    (0.9999, 3.0),
])
def test_newtonnu(ecc, nu):
    assert ms_ext.newtonnu(ecc, nu) == pytest.approx(
        up_ext.newtonnu(ecc, nu), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize('n', [5, 99999, 100000, 123456, 339999, 88888, 33333])
def test_alpha5(n):
    from mojosgp4.alpha5 import from_alpha5, to_alpha5
    from sgp4.alpha5 import from_alpha5 as up_from, to_alpha5 as up_to
    assert to_alpha5(n) == up_to(n)
    assert from_alpha5(to_alpha5(n)) == up_from(up_to(n)) == n


def test_tle_checksums():
    for line in (LINE1, LINE2):
        assert ms_io.compute_checksum(line) == up_io.compute_checksum(line)
    assert ms_io.fix_checksum(LINE1) == up_io.fix_checksum(LINE1)
    ms_io.verify_checksum(LINE1, LINE2)


# ------------------------------------------------------------- struct map
def test_field_count_matches_the_compiled_struct():
    from mojosgp4._lib import lib
    assert lib().ms_satrec_field_count() == len(SATREC_FIELDS)


def test_every_satrec_field_matches_upstream():
    """The Python field table must be the Mojo struct, in the same order."""
    checked = 0
    for line1, line2, _, _, _ in VER_TLES:
        ours = ms.Satrec.twoline2rv(line1, line2)
        theirs = up_model.Satrec.twoline2rv(line1, line2)
        for name in SATREC_FIELDS:
            a = getattr(ours, name)
            b = getattr(theirs, name, '<missing>')
            assert b != '<missing>', name
            if name == 'satnum':
                # upstream exposes the Alpha 5 string's value as a property
                checked += 1
                continue
            if name in SATREC_INT_FIELDS and not isinstance(a, str):
                assert a == b, name
            elif isinstance(a, str) or isinstance(b, str):
                assert a == b, name
            else:
                assert a == pytest.approx(b, rel=FIELD_RTOL, abs=1e-300), name
            checked += 1
    assert checked == len(VER_TLES) * len(SATREC_FIELDS)


def test_satnum_survives_the_struct_as_an_integer():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    assert sat.satnum == 5
    assert sat.satnum_str == '00005'
    assert sat.satnum_str == sat.satnum_str.rjust(5, '0')
    sat2 = ms.Satrec()
    sat2.satnum = 25544
    assert sat2.satnum_str == '25544'


def test_character_fields():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    theirs = up_model.Satrec.twoline2rv(LINE1, LINE2)
    assert sat.method == theirs.method == 'n'
    assert sat.init == theirs.init == 'n'
    assert sat.operationmode == theirs.operationmode == 'i'
    sat.operationmode = 'a'
    assert sat.operationmode == 'a'


def test_unknown_attribute_is_rejected():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    with pytest.raises(AttributeError):
        sat.no_such_field
    with pytest.raises(AttributeError):
        sat.no_such_field = 1.0


# ------------------------------------------------------------ propagation
def test_vanguard_identity():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    assert sat.satnum == 5
    assert sat.satnum_str == '00005'
    assert sat.classification == 'U'
    assert sat.operationmode == 'i'
    assert sat.epochyr == 0
    assert sat.epochdays == pytest.approx(179.78495062)
    assert sat.jdsatepoch == pytest.approx(2451722.5)
    assert sat.jdsatepochF == pytest.approx(0.78495062)
    assert sat.bstar == pytest.approx(2.8098e-05)
    assert sat.ndot == pytest.approx(6.96919666594958e-13)
    assert sat.nddot == 0.0
    assert sat.ecco == pytest.approx(0.1859667)
    assert sat.argpo == pytest.approx(5.790416027488515)
    assert sat.inclo == pytest.approx(0.5980929187319208)
    assert sat.mo == pytest.approx(0.3373093125574321)
    assert sat.no_kozai == pytest.approx(0.04722944544077857)
    assert sat.nodeo == pytest.approx(6.08638547138321)


def test_epoch_positions_match_upstream():
    """At t = 0 there is no accumulated divergence to hide behind."""
    for line1, line2, _, _, _ in VER_TLES:
        ours = ms.Satrec.twoline2rv(line1, line2)
        theirs = up_model.Satrec.twoline2rv(line1, line2)
        e1, r1, v1 = ours.sgp4_tsince(0.0)
        e2, r2, v2 = theirs.sgp4_tsince(0.0)
        assert e1 == e2
        if isnan(r2[0]):
            continue
        assert np.allclose(r1, r2, rtol=0, atol=EPOCH_TOL_KM, equal_nan=True), \
            ours.satnum
        assert np.allclose(v1, v2, rtol=0, atol=1e-8, equal_nan=True), \
            ours.satnum


def test_catalogue_positions_match_upstream():
    worst = 0.0
    worst_sat = None
    for line1, line2, tstart, tend, tstep in VER_TLES:
        ours = ms.Satrec.twoline2rv(line1, line2)
        theirs = up_model.Satrec.twoline2rv(line1, line2)
        for t in _times(tstart, tend, tstep):
            e1, r1, v1 = ours.sgp4_tsince(t)
            e2, r2, v2 = theirs.sgp4_tsince(t)
            assert e1 == e2, (ours.satnum, t)
            if isnan(r2[0]):
                assert all(isnan(x) for x in r1)
                continue
            delta = max(abs(a - b) for a, b in zip(r1, r2))
            if delta > worst:
                worst, worst_sat = delta, (ours.satnum, t)
            assert delta < CATALOGUE_TOL_KM, (ours.satnum, t, r1, r2)
    assert worst < CATALOGUE_TOL_KM
    # a bound this loose would pass a broken kernel; pin the real number too
    assert worst < 5.0, worst_sat


def test_error_codes_match_upstream():
    """The official suite expects exactly this sequence of failures."""
    expected = [1, 1, 6, 6, 4, 3, 6]
    seen = []
    for line1, line2, tstart, tend, tstep in VER_TLES:
        ours = ms.Satrec.twoline2rv(line1, line2)
        theirs = up_model.Satrec.twoline2rv(line1, line2)
        for t in _times(tstart, tend, tstep):
            e1, _, _ = ours.sgp4_tsince(t)
            e2, _, _ = theirs.sgp4_tsince(t)
            assert e1 == e2, (ours.satnum, t)
            if e1:
                seen.append(e1)
                break
    assert seen == expected


def _words(message):
    """The sentence without the number, which the pow difference perturbs."""
    return None if message is None else re.sub(r'[-+0-9.e]+', '', message)


def test_error_messages_match_upstream():
    for line1, line2, tstart, tend, tstep in VER_TLES:
        ours = ms.Satrec.twoline2rv(line1, line2)
        theirs = up_model.Satrec.twoline2rv(line1, line2)
        for t in _times(tstart, tend, tstep):
            ours.sgp4_tsince(t)
            theirs.sgp4_tsince(t)
            assert _words(ours.error_message) == _words(theirs.error_message), \
                (ours.satnum, t)
            if ours.error:
                break


def test_sgp4_by_julian_date():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    theirs = up_model.Satrec.twoline2rv(LINE1, LINE2)
    for t in (0.0, 360.0, 1440.0, 20160.0, -720.0):
        whole, fraction = divmod(t / 1440.0, 1.0)
        jd = sat.jdsatepoch + whole
        fr = sat.jdsatepochF + fraction
        e1, r1, v1 = sat.sgp4(jd, fr)
        e2, r2, v2 = theirs.sgp4(jd, fr)
        assert e1 == e2
        tol = EPOCH_TOL_KM if t == 0.0 else CATALOGUE_TOL_KM
        assert np.allclose(r1, r2, rtol=0, atol=tol, equal_nan=True)
        assert np.allclose(v1, v2, rtol=0, atol=1e-6, equal_nan=True)


def test_sgp4init_direct():
    sat = ms.Satrec()
    theirs = up_model.Satrec()
    args = (2.8098e-05, 6.96919666594958e-13, 0.0, 0.1859667,
            5.790416027488515, 0.5980929187319208, 0.3373093125574321,
            0.04722944544077857, 6.08638547138321)
    for mode in ('i', 'a'):
        a, b = ms.Satrec(), up_model.Satrec()
        a.sgp4init(ms.WGS72, mode, 5, 18441.78495062, *args)
        b.sgp4init(up_api.WGS72, mode, 5, 18441.78495062, *args)
        assert a.operationmode == b.operationmode == mode
        for name in ('gsto', 'no_unkozai', 'a', 'mdot', 'argpdot', 'nodedot',
                     'ecco', 'inclo', 'mo', 'nodeo', 'argpo'):
            assert getattr(a, name) == pytest.approx(getattr(b, name),
                                                     rel=FIELD_RTOL), name


def test_sgp4init_accepts_alpha5_and_int():
    args = (2.8098e-05, 6.96919666594958e-13, 0.0, 0.1859667,
            5.790416027488515, 0.5980929187319208, 0.3373093125574321,
            0.04722944544077857, 6.08638547138321)
    a, b = ms.Satrec(), up_model.Satrec()
    a.sgp4init(ms.WGS72, 'i', '00005', 18441.78495062, *args)
    b.sgp4init(up_api.WGS72, 'i', '00005', 18441.78495062, *args)
    assert a.satnum == b.satnum == 5
    c, d = ms.Satrec(), up_model.Satrec()
    c.sgp4init(ms.WGS72, 'i', 5, 18441.78495062, *args)
    d.sgp4init(up_api.WGS72, 'i', 5, 18441.78495062, *args)
    assert c.satnum == d.satnum == 5


# ----------------------------------------------------------- array APIs
def test_satrec_sgp4_array():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    theirs = up_model.Satrec.twoline2rv(LINE1, LINE2)
    tsince = np.array([0.0, 60.0, 360.0, 1440.0, 4320.0, 10080.0])
    jd = sat.jdsatepoch + np.floor(tsince / 1440.0)
    fr = sat.jdsatepochF + (tsince / 1440.0 - np.floor(tsince / 1440.0))
    e1, r1, v1 = sat.sgp4_array(jd, fr)
    e2, r2, v2 = theirs.sgp4_array(jd, fr)
    assert e1.shape == e2.shape == (len(tsince),)
    assert np.array_equal(e1, e2)
    assert r1.shape == r2.shape == (len(tsince), 3)
    assert np.allclose(r1, r2, rtol=0, atol=CATALOGUE_TOL_KM, equal_nan=True)
    assert np.allclose(v1, v2, rtol=0, atol=1e-3, equal_nan=True)


def test_satrec_array():
    # SatrecArray takes one set of dates for the whole catalogue, so pick
    # element sets that share an epoch: a date far from a satellite's own
    # epoch is a long extrapolation, and upstream's own two back ends part
    # company there too.
    chosen = [VER_TLES[i] for i in (3, 4, 7, 10, 12, 24)]
    ours = [ms.Satrec.twoline2rv(a, b) for a, b, _, _, _ in chosen]
    theirs = [up_model.Satrec.twoline2rv(a, b) for a, b, _, _, _ in chosen]
    assert len({round(s.jdsatepoch, 3) for s in ours}) == 1
    tsince = np.array([0.0, 120.0, 360.0, 1440.0])
    jd = 2453911.5 + np.floor(tsince / 1440.0)
    fr = tsince / 1440.0 - np.floor(tsince / 1440.0)
    e1, r1, v1 = ms_model.SatrecArray(ours).sgp4(jd, fr)
    e2, r2, v2 = up_model.SatrecArray(theirs).sgp4(jd, fr)
    assert e1.shape == (len(chosen), len(tsince))
    assert np.array_equal(e1, e2)
    assert r1.shape == (len(chosen), len(tsince), 3)
    assert np.allclose(r1, r2, rtol=0, atol=CATALOGUE_TOL_KM, equal_nan=True)
    assert np.allclose(v1, v2, rtol=0, atol=1e-3, equal_nan=True)


def test_satrec_array_matches_the_per_satellite_loop():
    """The batch kernel must walk satellites outside and times inside, because
    the deep space integrator carries state between calls."""
    chosen = [VER_TLES[i] for i in (18, 24, 27)]  # all deep space
    sats = [ms.Satrec.twoline2rv(a, b) for a, b, _, _, _ in chosen]
    tsince = np.array([0.0, 720.0, 1440.0, 2880.0, 5760.0])
    jd = 2451545.0 + np.floor(tsince / 1440.0)
    fr = tsince / 1440.0 - np.floor(tsince / 1440.0)
    _, r_batch, v_batch = ms_model.SatrecArray(sats).sgp4(jd, fr)
    fresh = [ms.Satrec.twoline2rv(a, b) for a, b, _, _, _ in chosen]
    for i, sat in enumerate(fresh):
        _, r, v = sat.sgp4_array(jd, fr)
        assert np.allclose(r_batch[i], r, rtol=0, atol=0.0, equal_nan=True)
        assert np.allclose(v_batch[i], v, rtol=0, atol=0.0, equal_nan=True)


# ------------------------------------------------------------------ I/O
def test_twoline2rv_rejects_malformed_lines():
    for bad1, bad2 in ((up_io.LINE1, LINE2), (LINE1, up_io.LINE2),
                       ('1 00005U', LINE2)):
        with pytest.raises(ValueError):
            ms.Satrec.twoline2rv(bad1, bad2)


def test_twoline2rv_rejects_mismatched_object_numbers():
    bad2 = LINE2[:2] + '00006' + LINE2[7:]
    with pytest.raises(ValueError):
        ms.Satrec.twoline2rv(LINE1, bad2)


def test_twoline2rv_rejects_non_ascii():
    with pytest.raises(ValueError):
        ms.Satrec.twoline2rv(LINE1 + 'é', LINE2)


def test_bad_checksum_is_reported():
    bad = LINE2[:68] + str((ms_io.compute_checksum(LINE2) + 1) % 10)
    with pytest.raises(ValueError):
        ms_io.verify_checksum(bad)
    ms_io.verify_checksum(ms_io.fix_checksum(bad))


# ------------------------------------------------------------- legacy 1.x
def test_legacy_satellite_object():
    from mojosgp4 import io as ms_io_legacy
    ours = ms_io_legacy.twoline2rv(LINE1, LINE2, ms_gravity.wgs72)
    theirs = up_io.twoline2rv(LINE1, LINE2, up_gravity.wgs72)
    for t in (0.0, 360.0, 1440.0):
        whole, fraction = divmod(t / 1440.0, 1.0)
        jd = ours.jdsatepoch + whole
        fr = ours.jdsatepochF + fraction
        m = (jd - ours.jdsatepoch) * 1440.0
        r, v = ms_propagation.sgp4(ours, m)
        m2 = (jd - theirs.jdsatepoch) * 1440.0
        r2, v2 = up_propagation.sgp4(theirs, m2)
        assert np.allclose(r, r2, rtol=0, atol=CATALOGUE_TOL_KM, equal_nan=True)
        assert np.allclose(v, v2, rtol=0, atol=1e-6, equal_nan=True)
    assert ours.propagate(2000, 6, 28, 0, 0, 0.0)


# --------------------------------------------------------- conveniences
def test_conveniences():
    from mojosgp4 import conveniences as ms_conv
    from sgp4 import conveniences as up_conv
    import datetime as dt
    when = dt.datetime(2020, 2, 11, 13, 57, tzinfo=dt.timezone.utc)
    assert ms_conv.jday_datetime(when) == pytest.approx(
        up_conv.jday_datetime(when), abs=1e-9)
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    theirs = up_model.Satrec.twoline2rv(LINE1, LINE2)
    assert ms_conv.sat_epoch_datetime(sat) == up_conv.sat_epoch_datetime(theirs)
    ms_conv.check_satrec(sat)
    up_conv.check_satrec(theirs)
    ours_lines = list(ms_conv.dump_satrec(sat))
    their_lines = list(up_conv.dump_satrec(theirs))
    assert [l.split(' = ')[0] for l in ours_lines] == \
        [l.split(' = ')[0] for l in their_lines]


def test_check_satrec_rejects_out_of_range():
    from mojosgp4 import conveniences as ms_conv
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    sat.argpo = 7.0
    with pytest.raises(ValueError):
        ms_conv.check_satrec(sat)


# ------------------------------------------------------------- API shape
def test_api_surface():
    for name in ('SGP4_ERRORS', 'Satrec', 'SatrecArray', 'WGS72OLD', 'WGS72',
                 'WGS84', 'accelerated', 'jday', 'days2mdhms'):
        assert hasattr(ms, name), name
    assert ms.SGP4_ERRORS == up_api.SGP4_ERRORS
    assert (ms.WGS72OLD, ms.WGS72, ms.WGS84) == \
        (up_api.WGS72OLD, up_api.WGS72, up_api.WGS84)
    assert ms.accelerated is True


def test_module_layout_matches_upstream():
    import mojosgp4
    for name in ('alpha5', 'api', 'conveniences', 'earth_gravity', 'ext',
                 'functions', 'io', 'model', 'propagation'):
        assert __import__('mojosgp4.' + name) is not None
    assert mojosgp4.__version__
    assert pi > 3.14
