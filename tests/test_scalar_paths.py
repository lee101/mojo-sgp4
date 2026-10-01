"""The scalar entry points: the shapes and side effects the Python layer owes
its callers, now that it keeps its scratch buffers between calls.

`propagation._sgp4`, `functions`, `ext` and `earth_gravity` all write into
module-level scratch that is no longer zeroed before each crossing, because
every exported kernel writes each slot it returns.  A kernel that ever stopped
writing one would leave the previous call's value behind, so these tests call
each helper twice with different arguments and check that nothing survives
from the first call.  They also pin the container types, because the callers
(`Satrec.sgp4`, `Satrec.sgp4_tsince`, `SatrecArray`) index and unpack them.
"""

from __future__ import annotations

import numpy as np
import pytest

import sgp4.api as up_api
import sgp4.earth_gravity as up_gravity
import sgp4.ext as up_ext
import sgp4.functions as up_functions
import sgp4.model as up_model
import sgp4.propagation as up_propagation
from pkgutil import get_data

import mojosgp4.api as ms
import mojosgp4.earth_gravity as ms_gravity
import mojosgp4.ext as ms_ext
import mojosgp4.functions as ms_functions
import mojosgp4.propagation as ms_propagation
from mojosgp4._lib import NFIELDS

LINE1 = '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
LINE2 = '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'

# satellite 29141 of the verification file, which decays; see the README on
# `mrt < 1.0` and the error code it raises
DECAY_LINE1 = '1 29141U 85108AA  06170.26783845  .00000000  00000-0  7154-2 0  8994'
DECAY_LINE2 = '2 29141  62.0906  77.1978 0308725 325.5477  36.1906 15.05380134  4179'


def _ver_tles():
    lines = get_data('sgp4', 'SGP4-VER.TLE').decode('ascii').splitlines()
    out = []
    it = iter(lines)
    for line1 in it:
        if not line1.startswith('1'):
            continue
        out.append((line1, next(it)))
    return out


VER_TLES = _ver_tles()


# ------------------------------------------------------- scratch reuse
def test_jday_does_not_inherit_the_previous_call():
    first = ms_functions.jday(2020, 2, 11, 13, 57, 0)
    second = ms_functions.jday(1957, 10, 4, 19, 26, 24.0)
    assert first != second
    assert first == pytest.approx(up_functions.jday(2020, 2, 11, 13, 57, 0))
    assert second == pytest.approx(up_functions.jday(1957, 10, 4, 19, 26, 24.0))


def test_days2mdhms_does_not_inherit_the_previous_call():
    a = ms_functions.days2mdhms(2000, 1.0)
    b = ms_functions.days2mdhms(1999, 366.0)
    assert a == up_functions.days2mdhms(2000, 1.0)
    assert b == up_functions.days2mdhms(1999, 366.0)


def test_day_of_year_to_month_day_does_not_inherit_the_previous_call():
    a = ms_functions._day_of_year_to_month_day(1, False)
    b = ms_functions._day_of_year_to_month_day(366, True)
    assert a == (1, 1)
    assert b == (12, 31)


def test_invjday_does_not_inherit_the_previous_call():
    a = ms_ext.invjday(2451545.0)
    b = ms_ext.invjday(2415020.5)
    assert a[0] == 2000 and b[0] == 1900
    assert a == pytest.approx(up_ext.invjday(2451545.0))
    assert b == pytest.approx(up_ext.invjday(2415020.5))


def test_newtonnu_does_not_inherit_the_previous_call():
    a = ms_ext.newtonnu(0.5, 1.0)
    b = ms_ext.newtonnu(0.0, 2.0)
    assert a == pytest.approx(up_ext.newtonnu(0.5, 1.0))
    # a circular orbit leaves e0 and m at nu, not at the previous call's value
    assert b == pytest.approx((2.0, 2.0))


def test_cross_does_not_inherit_the_previous_call():
    out = [0.0, 0.0, 0.0]
    ms_ext.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], out)
    assert out == [0.0, 0.0, 1.0]
    ms_ext.cross([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], out)
    assert out == [-3.0, 6.0, -3.0]


def test_getgravconst_does_not_inherit_the_previous_call():
    assert ms_gravity.getgravconst(0) == up_gravity.wgs72old
    assert ms_gravity.getgravconst(2) == up_gravity.wgs84


def test_gstime_returns_a_float():
    value = ms_propagation.gstime(2451545.0)
    assert type(value) is float
    assert value == pytest.approx(up_propagation.gstime(2451545.0))


# ------------------------------------------------------- sgp4_tsince
def test_sgp4_tsince_returns_error_and_two_tuples():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    error, r, v = sat.sgp4_tsince(0.0)
    assert type(error) is int
    ref = up_api.Satrec.twoline2rv(LINE1, LINE2).sgp4_tsince(0.0)
    assert error == ref[0]
    np.testing.assert_allclose(r, ref[1], atol=1.0e-5)
    np.testing.assert_allclose(v, ref[2], atol=1.0e-9)


def test_sgp4_tsince_agrees_with_the_module_level_sgp4():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    other = ms.Satrec.twoline2rv(LINE1, LINE2)
    for tsince in (0.0, 137.0, 1440.0, 20000.0):
        a = sat.sgp4_tsince(tsince)
        b = ms_propagation.sgp4(other, tsince)
        assert a[0] == other.error
        assert b[0] == pytest.approx(a[1])
        assert b[1] == pytest.approx(a[2])


def test_sgp4_writes_error_into_the_struct():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    error, _, _ = sat.sgp4_tsince(0.0)
    assert sat.error == error
    assert error == 0


def test_error_message_is_set_then_cleared():
    # a satellite whose mean motion is negative trips `nm <= 0`
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    sat.no_unkozai = -1.0
    error, r, v = sat.sgp4_tsince(0.0)
    assert error == 2
    assert 'less than zero' in sat.error_message
    assert all(np.isnan(x) for x in r + v)

    # the next call succeeds, and the message must not survive it
    healthy = ms.Satrec.twoline2rv(LINE1, LINE2)
    healthy._extra['error_message'] = 'stale'
    error, _, _ = ms_propagation._sgp4(healthy, 0.0)
    assert error == 0
    assert healthy.error_message is None


def test_error_message_records_the_out_of_range_value():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    sat.no_unkozai = -1.0
    sat.sgp4_tsince(0.0)
    assert 'mean motion' in sat.error_message
    assert '-1.000000' in sat.error_message


# ------------------------------------------------------- the cached address
def test_cached_address_follows_the_buffer():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    assert sat._address == sat._buf.ctypes.data
    before = sat.sgp4_tsince(0.0)[1]

    replacement = np.zeros(NFIELDS, dtype=np.float64)
    replacement[:] = sat._buf
    sat._buf = replacement
    assert sat._address == replacement.ctypes.data
    assert sat.sgp4_tsince(0.0)[1] == before

    # and the propagator is writing into the new block, not the old one
    old = sat._buf.copy()
    sat.sgp4_tsince(360.0)
    assert sat._buf.tolist() != old.tolist()


def test_int_fields_follow_a_reassigned_buffer():
    # the int64 view is a second window onto the same bytes; if it did not
    # move with the block, every `Int` field would keep writing into the
    # buffer that was replaced, and the kernel would read the old one
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    replacement = np.zeros(NFIELDS, dtype=np.float64)
    replacement[:] = sat._buf
    sat._buf = replacement
    sat.method = 'd'
    assert sat.method == 'd'
    sat.error = 6
    assert sat.error == 6
    assert sat.sgp4_tsince(0.0)[0] == 0
    assert sat.error == 0


@pytest.mark.parametrize('buf', [
    np.zeros(NFIELDS, dtype=np.int64),
    np.zeros(NFIELDS, dtype=np.float32),
    np.zeros(NFIELDS * 2)[::2],
    np.zeros(NFIELDS - 1),
])
def test_satrec_rejects_a_block_the_kernel_cannot_read(buf):
    # a wrongly typed, strided or short block would be read as the wrong
    # layout, or past its end
    with pytest.raises(ValueError):
        ms.Satrec(buf)


# ------------------------------------------------------- batch path
@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 7, 8, 63, 64, 65])
def test_satrec_array_covers_every_time(n):
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    ref = up_api.Satrec.twoline2rv(LINE1, LINE2)
    times = np.arange(float(n)) * 7.0
    jd = 2451723.0 + np.floor(times / 1440.0)
    fr = times / 1440.0 - np.floor(times / 1440.0)
    e, r, v = ms.SatrecArray([sat]).sgp4(jd, fr)
    re, rr, rv = up_api.SatrecArray([ref]).sgp4(jd, fr)
    assert e.shape == (1, n)
    assert np.array_equal(e, re)
    assert r.shape == (1, n, 3) and v.shape == (1, n, 3)
    np.testing.assert_allclose(r, rr, atol=1.0e-5)
    np.testing.assert_allclose(v, rv, atol=1.0e-9)


def test_satrec_array_walks_every_satellite():
    # one element set repeated, so this checks the stride arithmetic of the
    # batch loop and not the per-element-set parity, which test_parity.py
    # covers with the tolerances the README documents
    sats = [ms.Satrec.twoline2rv(*VER_TLES[0]) for _ in range(16)]
    refs = [up_api.Satrec.twoline2rv(*VER_TLES[0]) for _ in range(16)]
    n = 37
    times = np.arange(float(n)) * 11.0
    jd = 2451723.0 + np.floor(times / 1440.0)
    fr = times / 1440.0 - np.floor(times / 1440.0)
    e, r, v = ms.SatrecArray(sats).sgp4(jd, fr)
    re, rr, rv = up_api.SatrecArray(refs).sgp4(jd, fr)
    assert e.shape == (16, n)
    assert np.array_equal(e, re)
    np.testing.assert_allclose(r, rr, atol=1.0e-5)
    np.testing.assert_allclose(v, rv, atol=1.0e-9)
    # every row is the same, and none of them is a copy of its neighbour
    for row in range(1, 16):
        assert np.array_equal(r[row], r[0])
        assert np.array_equal(v[row], v[0])


def test_satrec_array_error_codes_match_upstream_over_the_catalogue():
    sats = [ms.Satrec.twoline2rv(a, b) for a, b in VER_TLES]
    refs = [up_api.Satrec.twoline2rv(a, b) for a, b in VER_TLES]
    n = 9
    times = np.arange(float(n)) * 120.0
    jd = 2451723.0 + np.floor(times / 1440.0)
    fr = times / 1440.0 - np.floor(times / 1440.0)
    e, _, _ = ms.SatrecArray(sats).sgp4(jd, fr)
    re, _, _ = up_api.SatrecArray(refs).sgp4(jd, fr)
    assert np.array_equal(e, re)


def test_satrec_array_of_one_satellite_matches_sgp4_array():
    sat = ms.Satrec.twoline2rv(*VER_TLES[0])
    ref = up_api.Satrec.twoline2rv(*VER_TLES[0])
    n = 101
    times = np.arange(float(n)) * 3.0
    jd = 2451723.0 + np.floor(times / 1440.0)
    fr = times / 1440.0 - np.floor(times / 1440.0)
    e, r, v = sat.sgp4_array(jd, fr)
    re, rr, rv = ref.sgp4_array(jd, fr)
    assert np.array_equal(e, re)
    np.testing.assert_allclose(r, rr, atol=1.0e-5)
    np.testing.assert_allclose(v, rv, atol=1.0e-9)


def test_satrec_array_sets_error_message_like_upstream():
    # the batch path must leave the same `error` and `error_message` behind as
    # the scalar path, or a caller cannot tell a failed batch from a good one
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    ref = up_model.Satrec.twoline2rv(LINE1, LINE2)
    sat.ecco = ref.ecco = 1.5
    jd = np.array([2451723.0, 2451724.0])
    fr = np.array([0.5, 0.5])
    e, _, _ = sat.sgp4_array(jd, fr)
    re, _, _ = ref.sgp4_array(jd, fr)
    assert np.array_equal(e, re) and e.tolist() == [1, 1]
    assert sat.error == ref.error == 1
    assert sat.error_message == ref.error_message
    assert 'eccentricity' in sat.error_message


def test_satrec_array_error_message_is_cleared_by_a_later_good_date():
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    sat.no_unkozai = -1.0
    sat.sgp4_array(np.array([2451723.0]), np.array([0.5]))
    assert sat.error_message is not None
    sat.no_unkozai = sat.no_kozai
    e, _, _ = sat.sgp4_array(np.array([2451723.0]), np.array([0.5]))
    assert e.tolist() == [0]
    assert sat.error_message is None


def test_satrec_array_writes_state_back_into_the_catalogue():
    # upstream's SatrecArray mutates each satrec: the deep space integrator
    # state and `error` survive the call, so a later call continues the
    # integration rather than restarting it.  A few element sets in the
    # catalogue are decaying, and a 1e-12 seed moves those by kilometres, so
    # the numerical comparison is against a near-Earth set; the state
    # comparison covers the whole catalogue.
    ours = [ms.Satrec.twoline2rv(*t) for t in VER_TLES]
    theirs = [up_model.Satrec.twoline2rv(*t) for t in VER_TLES]
    ours[0].ecco = theirs[0].ecco = 1.5
    jd = np.array([2453911.5, 2453912.5])
    fr = np.array([0.0, 0.0])
    _, r1, _ = ms.SatrecArray(ours).sgp4(jd, fr)
    _, r2, _ = up_model.SatrecArray(theirs).sgp4(jd, fr)
    assert ours[0].error == theirs[0].error
    assert ours[0].error_message == theirs[0].error_message
    for i in (1, 2, 3, 5):
        # the integrator state is the same recurrence in both, but Mojo's
        # `pow` is not correctly rounded, so compare to a relative 1e-8
        # rather than bit for bit
        assert ours[i].atime == pytest.approx(theirs[i].atime, rel=1.0e-8)
        assert ours[i].xli == pytest.approx(theirs[i].xli, rel=1.0e-8)
        assert ours[i].xni == pytest.approx(theirs[i].xni, rel=1.0e-8)
        assert ours[i].error == theirs[i].error
        # the dates here are a long extrapolation from epoch, which is where
        # the README's 3.1 km worst case lives; a near-Earth row is far
        # inside that
        np.testing.assert_allclose(r1[i], r2[i], rtol=0, atol=1.0)


@pytest.mark.parametrize('shape', [(4, 2), (2, 4)])
def test_batch_calls_reject_mismatched_date_lengths(shape):
    # the kernel indexes jd and fr with one counter, so a shorter fr would be
    # read past its end; it must raise rather than return numbers from
    # whatever happens to follow the buffer
    sat = ms.Satrec.twoline2rv(LINE1, LINE2)
    jd = np.full(shape, 2451723.5)
    fr = np.full(shape[::-1], 0.25)
    with pytest.raises(ValueError):
        sat.sgp4_array(jd, fr)
    with pytest.raises(ValueError):
        ms.SatrecArray([sat]).sgp4(jd, fr)


def test_jday_and_days2mdhms_round_trip():
    for year, days in ((2000, 1.0), (1999, 200.5), (2020, 366.0), (1957, 41.25)):
        mon, day, hr, minute, sec = ms_functions.days2mdhms(year, days)
        jd, fr = ms_functions.jday(year, mon, day, hr, minute, sec)
        got = up_functions.jday(year, mon, day, hr, minute, sec)
        assert (jd, fr) == pytest.approx(got)
        assert jd + fr == pytest.approx(
            up_functions.jday(year, mon, day, hr, minute, sec)[0]
            + up_functions.jday(year, mon, day, hr, minute, sec)[1])
