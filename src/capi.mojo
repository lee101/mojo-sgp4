"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric.  Callers own all
memory, including scratch, so nothing here allocates and nothing here can leak.

This file is also where the one structural divergence in the port lives:
`ms_sgp4_array` takes the `Satrec` block as a base address plus a stride, so
that `SatrecArray.sgp4`'s satellite-outer, time-inner loop stays the nested
loop upstream writes.
"""

from ported import (
    Satrec,
    _day_of_year_to_month_day,
    days2mdhms,
    ext_jday,
    fmod,
    getgravconst,
    gstime,
    invjday,
    jday,
    mag,
    newtonnu,
    sgp4,
    sgp4_array,
    sgp4init,
    vector_angle,
    vector_cross,
    vector_dot,
)

comptime PtrF = Pointer[Float64, AnyOrigin[mut=True]]
comptime PtrSat = Pointer[Satrec, AnyOrigin[mut=True]]
comptime PtrU8 = Pointer[UInt8, AnyOrigin[mut=True]]

comptime SATREC_FIELD_COUNT = 119


def p(addr: Int) -> PtrSat:
    return PtrSat(unsafe_from_address=addr)


def pf(addr: Int) -> PtrF:
    return PtrF(unsafe_from_address=addr)


# ----------------------------------------------------------------- helpers
@export("ms_fmod")
def ms_fmod(x: Float64, y: Float64) abi("C") -> Float64:
    return fmod(x, y)


@export("ms_gstime")
def ms_gstime(jdut1: Float64) abi("C") -> Float64:
    return gstime(jdut1)


@export("ms_getgravconst")
def ms_getgravconst(whichconst: Int, dst: Int) abi("C") -> Int:
    """Write the eight `EarthGravity` fields; returns the field count."""
    var c = getgravconst(whichconst)
    var q = pf(dst)
    q.unsafe_store(0, c.tumin)
    q.unsafe_store(1, c.mu)
    q.unsafe_store(2, c.radiusearthkm)
    q.unsafe_store(3, c.xke)
    q.unsafe_store(4, c.j2)
    q.unsafe_store(5, c.j3)
    q.unsafe_store(6, c.j4)
    q.unsafe_store(7, c.j3oj2)
    return 8


# ---------------------------------------------------------------- sgp4io
@export("ms_jday")
def ms_jday(
    year: Int, mon_: Int, day_: Int, hr_: Int, minute_: Int, sec_: Float64, dst: Int
) abi("C"):
    var r = jday(year, mon_, day_, hr_, minute_, sec_)
    var q = pf(dst)
    q.unsafe_store(0, r.jd)
    q.unsafe_store(1, r.fr)


@export("ms_days2mdhms")
def ms_days2mdhms(
    year: Int, days: Float64, round_to_microsecond: Int, dst: Int
) abi("C"):
    var r = days2mdhms(year, days, round_to_microsecond)
    var q = pf(dst)
    q.unsafe_store(0, Float64(r.mon))
    q.unsafe_store(1, Float64(r.day))
    q.unsafe_store(2, Float64(r.hr))
    q.unsafe_store(3, Float64(r.minute))
    q.unsafe_store(4, r.sec)


@export("ms_ext_jday")
def ms_ext_jday(
    year: Int, mon_: Int, day_: Int, hr_: Int, minute_: Int, sec_: Float64
) abi("C") -> Float64:
    return ext_jday(year, mon_, day_, hr_, minute_, sec_)


@export("ms_day_of_year_to_month_day")
def ms_day_of_year_to_month_day(
    day_of_year: Int, is_leap: Int, dst: Int
) abi("C"):
    var r = _day_of_year_to_month_day(day_of_year, is_leap)
    var q = pf(dst)
    q.unsafe_store(0, Float64(r.month))
    q.unsafe_store(1, Float64(r.day))


@export("ms_invjday")
def ms_invjday(jd_: Float64, dst: Int) abi("C"):
    var r = invjday(jd_)
    var q = pf(dst)
    q.unsafe_store(0, Float64(r.year))
    q.unsafe_store(1, Float64(r.mon))
    q.unsafe_store(2, Float64(r.day))
    q.unsafe_store(3, Float64(r.hr))
    q.unsafe_store(4, Float64(r.minute))
    q.unsafe_store(5, r.sec)


# ---------------------------------------------------------------- sgp4ext
@export("ms_mag")
def ms_mag(v: Int) abi("C") -> Float64:
    return mag(pf(v))


@export("ms_dot")
def ms_dot(x: Int, y: Int) abi("C") -> Float64:
    return vector_dot(pf(x), pf(y))


@export("ms_cross")
def ms_cross(vec1: Int, vec2: Int, outvec: Int) abi("C"):
    vector_cross(pf(vec1), pf(vec2), pf(outvec))


@export("ms_angle")
def ms_angle(vec1: Int, vec2: Int) abi("C") -> Float64:
    return vector_angle(pf(vec1), pf(vec2))


@export("ms_newtonnu")
def ms_newtonnu(ecc: Float64, nu: Float64, dst: Int) abi("C"):
    var r = newtonnu(ecc, nu)
    var q = pf(dst)
    q.unsafe_store(0, r.e0)
    q.unsafe_store(1, r.m)


# ------------------------------------------------------------ propagation
@export("ms_sgp4init")
def ms_sgp4init(
    whichconst: Int,
    opsmode: Int,
    satn: Int,
    epoch: Float64,
    xbstar: Float64,
    xndot: Float64,
    xnddot: Float64,
    xecco: Float64,
    xargpo: Float64,
    xinclo: Float64,
    xmo: Float64,
    xno_kozai: Float64,
    xnodeo: Float64,
    satrec: Int,
) abi("C") -> Int:
    return 1 if sgp4init(
        whichconst,
        opsmode,
        satn,
        epoch,
        xbstar,
        xndot,
        xnddot,
        xecco,
        xargpo,
        xinclo,
        xmo,
        xno_kozai,
        xnodeo,
        p(satrec),
    ) else 0


@export("ms_sgp4")
def ms_sgp4(satrec: Int, tsince: Float64, r: Int, v: Int, errval: Int) abi("C") -> Int:
    """Returns upstream's `satrec.error`.  `errval` receives the value that went
    out of range, which the Python layer formats into `error_message`."""
    var out = sgp4(p(satrec), tsince)
    var rv = pf(r)
    rv.unsafe_store(0, out.r.x)
    rv.unsafe_store(1, out.r.y)
    rv.unsafe_store(2, out.r.z)
    var vv = pf(v)
    vv.unsafe_store(0, out.v.x)
    vv.unsafe_store(1, out.v.y)
    vv.unsafe_store(2, out.v.z)
    pf(errval).unsafe_store(0, out.errval)
    return p(satrec)[].error


@export("ms_sgp4_array")
def ms_sgp4_array(
    satrecs: Int,
    stride: Int,
    n_sat: Int,
    jd: Int,
    fr: Int,
    n: Int,
    e: Int,
    r: Int,
    v: Int,
    errval: Int,
) abi("C"):
    sgp4_array(
        satrecs, stride, n_sat, pf(jd), pf(fr), n,
        PtrU8(unsafe_from_address=e), pf(r), pf(v), pf(errval),
    )


# ----------------------------------------------------------------- layout
@export("ms_satrec_field_count")
def ms_satrec_field_count() abi("C") -> Int:
    """The `Satrec` struct is a packed row of 8-byte fields, so the Python layer
    derives every offset from this count.  `tests/test_parity.py` checks the
    count against its own field list and then checks every field against
    upstream, so a mismatch cannot pass silently."""
    return SATREC_FIELD_COUNT
