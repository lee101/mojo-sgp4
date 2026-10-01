"""A Mojo port of the SGP4 orbit propagator from the Python `sgp4` package.

Read this top to bottom against `sgp4/propagation.py` of python-sgp4 2.27 and
the two should stay legible side by side: same function names, same argument
order, same branch structure, same order of operations.  Upstream keeps the
C++ of Vallado's `sgp4unit.cpp` almost verbatim; so does this.

Three places where the Mojo dialect forces a divergence, all documented in the
README and none of them touching the arithmetic:

*  Upstream returns a Python tuple of 5-76 values from `_dpper`, `_dscom`,
   `_dsinit`, `_dspace` and `_initl`.  Mojo structs fill the same role, with
   the same field names, so the return sites read the same.
*  Upstream mutates a `satrec` object; here `satrec` is a pointer to the
   equivalent `Satrec` struct and `satrec[].x` is upstream's `satrec.x`.
*  Upstream compares `method`/`init`/`operationmode` against `'n'`/`'d'` and
   `'y'`/`'n'`; these are held as small integers here, because a struct with a
   non-trivial field type would not be a plain 8-byte-per-field block.  The
   `python/mojosgp4` layer converts back to the upstream characters.

`gstime` and `getgravconst` are hoisted above their callers, and `sgp4` is
placed before `sgp4init`, because Mojo resolves names in file order and the
upstream call graph is not a DAG.

"""

from std.math import acos, asinh, atan2, cos, exp, floor, log, nan, pi, pow, round
from std.math import sin, sinh, sqrt, tan

comptime deg2rad = pi / 180.0

comptime twopi = 2.0 * pi

comptime x2o3 = 2.0 / 3.0

comptime minutes_per_day = 1440.0

# `sgp4.model` numbers the gravity models; upstream indexes a tuple with them.

comptime WGS72OLD = 0

comptime WGS72 = 1

comptime WGS84 = 2

# Stand-ins for upstream's one-character `method`, `init` and `opsmode` flags.

comptime METHOD_N = 0

comptime METHOD_D = 1

comptime INIT_Y = 0

comptime INIT_N = 1

comptime OPSMODE_A = 0

comptime OPSMODE_I = 1

def fmod(x: Float64, y: Float64) -> Float64:
    """C `fmod`, which `std.math` does not carry: the quotient is truncated
    toward zero, so the result keeps x's sign.  Mojo's `%` is Python's, which
    floors instead, and `newtonnu` depends on the difference."""
    var q = x / y
    var n = floor(abs(q))
    if q < 0.0:
        n = -n
    return x - y * n

# ----------------------------------------------------------------------------

# The satellite.  Field names, and their order, come from `Satrec.__slots__`

# in `sgp4/model.py`; the handful of entries that hold a string or a datetime

# upstream (`classification`, `elnum`, `ephtype`, `epoch`, `error_message`,

# `intldesg`, `revnum`, `satnum_str`) are not part of the propagator and stay

# on the Python side.  Every remaining field is exactly 8 bytes, so the struct

# is a packed row-major block of doubles at 8 * index.

struct Satrec(ImplicitlyCopyable, Copyable, Movable):
    var Om: Float64
    var a: Float64
    var alta: Float64
    var altp: Float64
    var am: Float64
    var argpdot: Float64
    var argpo: Float64
    var atime: Float64
    var aycof: Float64
    var bstar: Float64
    var cc1: Float64
    var cc4: Float64
    var cc5: Float64
    var con41: Float64
    var d2: Float64
    var d2201: Float64
    var d2211: Float64
    var d3: Float64
    var d3210: Float64
    var d3222: Float64
    var d4: Float64
    var d4410: Float64
    var d4422: Float64
    var d5220: Float64
    var d5232: Float64
    var d5421: Float64
    var d5433: Float64
    var dedt: Float64
    var del1: Float64
    var del2: Float64
    var del3: Float64
    var delmo: Float64
    var didt: Float64
    var dmdt: Float64
    var dnodt: Float64
    var domdt: Float64
    var e3: Float64
    var ecco: Float64
    var ee2: Float64
    var em: Float64
    var epochdays: Float64
    var epochyr: Int
    var error: Int
    var eta: Float64
    var gsto: Float64
    var im: Float64
    var inclo: Float64
    var init: Int
    var irez: Int
    var isimp: Int
    var j2: Float64
    var j3: Float64
    var j3oj2: Float64
    var j4: Float64
    var jdsatepoch: Float64
    var jdsatepochF: Float64
    var mdot: Float64
    var method: Int
    var mm: Float64
    var mo: Float64
    var mu: Float64
    var nddot: Float64
    var ndot: Float64
    var nm: Float64
    var no_kozai: Float64
    var no_unkozai: Float64
    var nodecf: Float64
    var nodedot: Float64
    var nodeo: Float64
    var om: Float64
    var omgcof: Float64
    var operationmode: Int
    var peo: Float64
    var pgho: Float64
    var pho: Float64
    var pinco: Float64
    var plo: Float64
    var radiusearthkm: Float64
    var satnum: Int
    var se2: Float64
    var se3: Float64
    var sgh2: Float64
    var sgh3: Float64
    var sgh4: Float64
    var sh2: Float64
    var sh3: Float64
    var si2: Float64
    var si3: Float64
    var sinmao: Float64
    var sl2: Float64
    var sl3: Float64
    var sl4: Float64
    var t: Float64
    var t2cof: Float64
    var t3cof: Float64
    var t4cof: Float64
    var t5cof: Float64
    var tumin: Float64
    var x1mth2: Float64
    var x7thm1: Float64
    var xfact: Float64
    var xgh2: Float64
    var xgh3: Float64
    var xgh4: Float64
    var xh2: Float64
    var xh3: Float64
    var xi2: Float64
    var xi3: Float64
    var xke: Float64
    var xl2: Float64
    var xl3: Float64
    var xl4: Float64
    var xlamo: Float64
    var xlcof: Float64
    var xli: Float64
    var xmcof: Float64
    var xni: Float64
    var zmol: Float64
    var zmos: Float64

    def __init__(out self, Om: Float64, a: Float64, alta: Float64, altp: Float64, am: Float64, argpdot: Float64, argpo: Float64, atime: Float64, aycof: Float64, bstar: Float64, cc1: Float64, cc4: Float64, cc5: Float64, con41: Float64, d2: Float64, d2201: Float64, d2211: Float64, d3: Float64, d3210: Float64, d3222: Float64, d4: Float64, d4410: Float64, d4422: Float64, d5220: Float64, d5232: Float64, d5421: Float64, d5433: Float64, dedt: Float64, del1: Float64, del2: Float64, del3: Float64, delmo: Float64, didt: Float64, dmdt: Float64, dnodt: Float64, domdt: Float64, e3: Float64, ecco: Float64, ee2: Float64, em: Float64, epochdays: Float64, epochyr: Int, error: Int, eta: Float64, gsto: Float64, im: Float64, inclo: Float64, init: Int, irez: Int, isimp: Int, j2: Float64, j3: Float64, j3oj2: Float64, j4: Float64, jdsatepoch: Float64, jdsatepochF: Float64, mdot: Float64, method: Int, mm: Float64, mo: Float64, mu: Float64, nddot: Float64, ndot: Float64, nm: Float64, no_kozai: Float64, no_unkozai: Float64, nodecf: Float64, nodedot: Float64, nodeo: Float64, om: Float64, omgcof: Float64, operationmode: Int, peo: Float64, pgho: Float64, pho: Float64, pinco: Float64, plo: Float64, radiusearthkm: Float64, satnum: Int, se2: Float64, se3: Float64, sgh2: Float64, sgh3: Float64, sgh4: Float64, sh2: Float64, sh3: Float64, si2: Float64, si3: Float64, sinmao: Float64, sl2: Float64, sl3: Float64, sl4: Float64, t: Float64, t2cof: Float64, t3cof: Float64, t4cof: Float64, t5cof: Float64, tumin: Float64, x1mth2: Float64, x7thm1: Float64, xfact: Float64, xgh2: Float64, xgh3: Float64, xgh4: Float64, xh2: Float64, xh3: Float64, xi2: Float64, xi3: Float64, xke: Float64, xl2: Float64, xl3: Float64, xl4: Float64, xlamo: Float64, xlcof: Float64, xli: Float64, xmcof: Float64, xni: Float64, zmol: Float64, zmos: Float64):
        self.Om = Om
        self.a = a
        self.alta = alta
        self.altp = altp
        self.am = am
        self.argpdot = argpdot
        self.argpo = argpo
        self.atime = atime
        self.aycof = aycof
        self.bstar = bstar
        self.cc1 = cc1
        self.cc4 = cc4
        self.cc5 = cc5
        self.con41 = con41
        self.d2 = d2
        self.d2201 = d2201
        self.d2211 = d2211
        self.d3 = d3
        self.d3210 = d3210
        self.d3222 = d3222
        self.d4 = d4
        self.d4410 = d4410
        self.d4422 = d4422
        self.d5220 = d5220
        self.d5232 = d5232
        self.d5421 = d5421
        self.d5433 = d5433
        self.dedt = dedt
        self.del1 = del1
        self.del2 = del2
        self.del3 = del3
        self.delmo = delmo
        self.didt = didt
        self.dmdt = dmdt
        self.dnodt = dnodt
        self.domdt = domdt
        self.e3 = e3
        self.ecco = ecco
        self.ee2 = ee2
        self.em = em
        self.epochdays = epochdays
        self.epochyr = epochyr
        self.error = error
        self.eta = eta
        self.gsto = gsto
        self.im = im
        self.inclo = inclo
        self.init = init
        self.irez = irez
        self.isimp = isimp
        self.j2 = j2
        self.j3 = j3
        self.j3oj2 = j3oj2
        self.j4 = j4
        self.jdsatepoch = jdsatepoch
        self.jdsatepochF = jdsatepochF
        self.mdot = mdot
        self.method = method
        self.mm = mm
        self.mo = mo
        self.mu = mu
        self.nddot = nddot
        self.ndot = ndot
        self.nm = nm
        self.no_kozai = no_kozai
        self.no_unkozai = no_unkozai
        self.nodecf = nodecf
        self.nodedot = nodedot
        self.nodeo = nodeo
        self.om = om
        self.omgcof = omgcof
        self.operationmode = operationmode
        self.peo = peo
        self.pgho = pgho
        self.pho = pho
        self.pinco = pinco
        self.plo = plo
        self.radiusearthkm = radiusearthkm
        self.satnum = satnum
        self.se2 = se2
        self.se3 = se3
        self.sgh2 = sgh2
        self.sgh3 = sgh3
        self.sgh4 = sgh4
        self.sh2 = sh2
        self.sh3 = sh3
        self.si2 = si2
        self.si3 = si3
        self.sinmao = sinmao
        self.sl2 = sl2
        self.sl3 = sl3
        self.sl4 = sl4
        self.t = t
        self.t2cof = t2cof
        self.t3cof = t3cof
        self.t4cof = t4cof
        self.t5cof = t5cof
        self.tumin = tumin
        self.x1mth2 = x1mth2
        self.x7thm1 = x7thm1
        self.xfact = xfact
        self.xgh2 = xgh2
        self.xgh3 = xgh3
        self.xgh4 = xgh4
        self.xh2 = xh2
        self.xh3 = xh3
        self.xi2 = xi2
        self.xi3 = xi3
        self.xke = xke
        self.xl2 = xl2
        self.xl3 = xl3
        self.xl4 = xl4
        self.xlamo = xlamo
        self.xlcof = xlcof
        self.xli = xli
        self.xmcof = xmcof
        self.xni = xni
        self.zmol = zmol
        self.zmos = zmos

comptime PtrSat = Pointer[Satrec, AnyOrigin[mut=True]]

comptime PtrF = Pointer[Float64, AnyOrigin[mut=True]]

comptime PtrI = Pointer[Int, AnyOrigin[mut=True]]

comptime PtrU8 = Pointer[UInt8, AnyOrigin[mut=True]]

# ----------------------------------------------------------------------------

#  function gstime

#

#  this function finds the greenwich sidereal time.

# ----------------------------------------------------------------------------

def gstime(jdut1: Float64) -> Float64:
    var tut1 = (jdut1 - 2451545.0) / 36525.0
    var temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
           (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841  #  sec
    temp = (temp * deg2rad / 240.0) % twopi  # 360/86400 = 1/240, to deg, to rad

    #  ------------------------ check quadrants ---------------------
    if temp < 0.0:
        temp += twopi

    return temp

# ----------------------------------------------------------------------------

#  function getgravconst

#

#  this function gets constants for the propagator.  the common useage is

#  wgs72, which is what upstream indexes with the `model.WGS72` enum.

# ----------------------------------------------------------------------------

struct EarthGravity(ImplicitlyCopyable, Copyable, Movable):
    var tumin: Float64
    var mu: Float64
    var radiusearthkm: Float64
    var xke: Float64
    var j2: Float64
    var j3: Float64
    var j4: Float64
    var j3oj2: Float64

    def __init__(out self, tumin: Float64, mu: Float64, radiusearthkm: Float64, xke: Float64, j2: Float64, j3: Float64, j4: Float64, j3oj2: Float64):
        self.tumin = tumin
        self.mu = mu
        self.radiusearthkm = radiusearthkm
        self.xke = xke
        self.j2 = j2
        self.j3 = j3
        self.j4 = j4
        self.j3oj2 = j3oj2

def getgravconst(whichconst: Int) -> EarthGravity:
    var mu: Float64
    var radiusearthkm: Float64
    var xke: Float64
    var j2: Float64
    var j3: Float64
    var j4: Float64

    if whichconst == WGS72OLD:
        mu = 398600.79964  #  in km3 / s2
        radiusearthkm = 6378.135  #  km
        xke = 0.0743669161
        j2 = 0.001082616
        j3 = -0.00000253881
        j4 = -0.00000165597
    elif whichconst == WGS72:
        #  ------------ wgs-72 constants ------------
        mu = 398600.8  #  in km3 / s2
        radiusearthkm = 6378.135  #  km
        xke = 60.0 / sqrt(radiusearthkm * radiusearthkm * radiusearthkm / mu)
        j2 = 0.001082616
        j3 = -0.00000253881
        j4 = -0.00000165597
    else:
        #  ------------ wgs-84 constants ------------
        mu = 398600.5  #  in km3 / s2
        radiusearthkm = 6378.137  #  km
        xke = 60.0 / sqrt(radiusearthkm * radiusearthkm * radiusearthkm / mu)
        j2 = 0.00108262998905
        j3 = -0.00000253215306
        j4 = -0.00000161098761

    var tumin = 1.0 / xke
    var j3oj2 = j3 / j2
    return EarthGravity(tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2)

# ----------------------------------------------------------------------------

#  procedure dpper

#

#  this procedure provides deep space long period periodic contributions

#    to the mean elements.  by design, these periodics are zero at epoch.

# ----------------------------------------------------------------------------

struct DpperResult(ImplicitlyCopyable, Copyable, Movable):
    var ep: Float64
    var inclp: Float64
    var nodep: Float64
    var argpp: Float64
    var mp: Float64

    def __init__(out self, ep: Float64, inclp: Float64, nodep: Float64, argpp: Float64, mp: Float64):
        self.ep = ep
        self.inclp = inclp
        self.nodep = nodep
        self.argpp = argpp
        self.mp = mp

def _dpper(
    satrec: PtrSat,
    inclo: Float64,
    init: Int,
    ep_: Float64,
    inclp_: Float64,
    nodep_: Float64,
    argpp_: Float64,
    mp_: Float64,
    opsmode: Int,
) -> DpperResult:

    # Mojo binds parameters immutably and will not let a body rebind
    # one, while upstream treats these values as in/out.  Take mutable
    # locals of the same names on the way in; the call is unchanged.
    var ep = ep_
    var inclp = inclp_
    var nodep = nodep_
    var argpp = argpp_
    var mp = mp_
    # Copy satellite attributes into local variables for convenience
    # and symmetry in writing formulae.
    var e3 = satrec[].e3
    var ee2 = satrec[].ee2
    var peo = satrec[].peo
    var pgho = satrec[].pgho
    var pho = satrec[].pho
    var pinco = satrec[].pinco
    var plo = satrec[].plo
    var se2 = satrec[].se2
    var se3 = satrec[].se3
    var sgh2 = satrec[].sgh2
    var sgh3 = satrec[].sgh3
    var sgh4 = satrec[].sgh4
    var sh2 = satrec[].sh2
    var sh3 = satrec[].sh3
    var si2 = satrec[].si2
    var si3 = satrec[].si3
    var sl2 = satrec[].sl2
    var sl3 = satrec[].sl3
    var sl4 = satrec[].sl4
    var t = satrec[].t
    var xgh2 = satrec[].xgh2
    var xgh3 = satrec[].xgh3
    var xgh4 = satrec[].xgh4
    var xh2 = satrec[].xh2
    var xh3 = satrec[].xh3
    var xi2 = satrec[].xi2
    var xi3 = satrec[].xi3
    var xl2 = satrec[].xl2
    var xl3 = satrec[].xl3
    var xl4 = satrec[].xl4
    var zmol = satrec[].zmol
    var zmos = satrec[].zmos

    #  ---------------------- constants -----------------------------
    var zns = 1.19459e-5
    var zes = 0.01675
    var znl = 1.5835218e-4
    var zel = 0.05490

    #  --------------- calculate time varying periodics -----------
    var zm = zmos + zns * t
    # be sure that the initial call has time set to zero
    if init == INIT_Y:
        zm = zmos
    var zf = zm + 2.0 * zes * sin(zm)
    var sinzf = sin(zf)
    var f2 = 0.5 * sinzf * sinzf - 0.25
    var f3 = -0.5 * sinzf * cos(zf)
    var ses = se2 * f2 + se3 * f3
    var sis = si2 * f2 + si3 * f3
    var sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf
    var sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf
    var shs = sh2 * f2 + sh3 * f3
    zm = zmol + znl * t
    if init == INIT_Y:
        zm = zmol
    zf = zm + 2.0 * zel * sin(zm)
    sinzf = sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * cos(zf)
    var sel = ee2 * f2 + e3 * f3
    var sil = xi2 * f2 + xi3 * f3
    var sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf
    var sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf
    var shll = xh2 * f2 + xh3 * f3
    var pe = ses + sel
    var pinc = sis + sil
    var pl = sls + sll
    var pgh = sghs + sghl
    var ph = shs + shll

    if init == INIT_N:

        pe = pe - peo
        pinc = pinc - pinco
        pl = pl - plo
        pgh = pgh - pgho
        ph = ph - pho
        inclp = inclp + pinc
        ep = ep + pe
        var sinip = sin(inclp)
        var cosip = cos(inclp)

        # --- upstream comment block, kept for navigation ---
        # /* ----------------- apply periodics directly ------------ */
        # //  sgp4fix for lyddane choice
        # //  strn3 used original inclination - this is technically feasible
        # //  gsfc used perturbed inclination - also technically feasible
        # //  probably best to readjust the 0.2 limit value and limit discontinuity
        # //  0.2 rad = 11.45916 deg

        if inclp >= 0.2:

            ph = ph / sinip
            pgh = pgh - cosip * ph
            argpp = argpp + pgh
            nodep = nodep + ph
            mp = mp + pl

        else:

            #  ---- apply periodics with lyddane modification ----
            var sinop = sin(nodep)
            var cosop = cos(nodep)
            var alfdp = sinip * sinop
            var betdp = sinip * cosop
            var dalf = ph * cosop + pinc * cosip * sinop
            var dbet = -ph * sinop + pinc * cosip * cosop
            alfdp = alfdp + dalf
            betdp = betdp + dbet
            nodep = nodep % twopi if nodep >= 0.0 else -(-nodep % twopi)
            #   sgp4fix for afspc written intrinsic functions
            #  nodep used without a trigonometric function ahead
            if nodep < 0.0 and opsmode == OPSMODE_A:
                nodep = nodep + twopi
            var xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep
            var xnoh = nodep
            nodep = atan2(alfdp, betdp)
            #   sgp4fix for afspc written intrinsic functions
            #  nodep used without a trigonometric function ahead
            if nodep < 0.0 and opsmode == OPSMODE_A:
                nodep = nodep + twopi
            if abs(xnoh - nodep) > pi:
                if nodep < xnoh:
                    nodep = nodep + twopi
                else:
                    nodep = nodep - twopi
            mp = mp + pl
            argpp = xls - mp - cosip * nodep

    return DpperResult(ep, inclp, nodep, argpp, mp)

# ----------------------------------------------------------------------------

#  procedure dscom

#

#  this procedure provides deep space common items used by both the secular

#    and periodics subroutines.

# ----------------------------------------------------------------------------

struct DscomResult(ImplicitlyCopyable, Copyable, Movable):
    var snodm: Float64
    var cnodm: Float64
    var sinim: Float64
    var cosim: Float64
    var sinomm: Float64
    var cosomm: Float64
    var day: Float64
    var e3: Float64
    var ee2: Float64
    var em: Float64
    var emsq: Float64
    var gam: Float64
    var peo: Float64
    var pgho: Float64
    var pho: Float64
    var pinco: Float64
    var plo: Float64
    var rtemsq: Float64
    var se2: Float64
    var se3: Float64
    var sgh2: Float64
    var sgh3: Float64
    var sgh4: Float64
    var sh2: Float64
    var sh3: Float64
    var si2: Float64
    var si3: Float64
    var sl2: Float64
    var sl3: Float64
    var sl4: Float64
    var s1: Float64
    var s2: Float64
    var s3: Float64
    var s4: Float64
    var s5: Float64
    var s6: Float64
    var s7: Float64
    var ss1: Float64
    var ss2: Float64
    var ss3: Float64
    var ss4: Float64
    var ss5: Float64
    var ss6: Float64
    var ss7: Float64
    var sz1: Float64
    var sz2: Float64
    var sz3: Float64
    var sz11: Float64
    var sz12: Float64
    var sz13: Float64
    var sz21: Float64
    var sz22: Float64
    var sz23: Float64
    var sz31: Float64
    var sz32: Float64
    var sz33: Float64
    var xgh2: Float64
    var xgh3: Float64
    var xgh4: Float64
    var xh2: Float64
    var xh3: Float64
    var xi2: Float64
    var xi3: Float64
    var xl2: Float64
    var xl3: Float64
    var xl4: Float64
    var nm: Float64
    var z1: Float64
    var z2: Float64
    var z3: Float64
    var z11: Float64
    var z12: Float64
    var z13: Float64
    var z21: Float64
    var z22: Float64
    var z23: Float64
    var z31: Float64
    var z32: Float64
    var z33: Float64
    var zmol: Float64
    var zmos: Float64

    def __init__(out self, snodm: Float64, cnodm: Float64, sinim: Float64, cosim: Float64, sinomm: Float64, cosomm: Float64, day: Float64, e3: Float64, ee2: Float64, em: Float64, emsq: Float64, gam: Float64, peo: Float64, pgho: Float64, pho: Float64, pinco: Float64, plo: Float64, rtemsq: Float64, se2: Float64, se3: Float64, sgh2: Float64, sgh3: Float64, sgh4: Float64, sh2: Float64, sh3: Float64, si2: Float64, si3: Float64, sl2: Float64, sl3: Float64, sl4: Float64, s1: Float64, s2: Float64, s3: Float64, s4: Float64, s5: Float64, s6: Float64, s7: Float64, ss1: Float64, ss2: Float64, ss3: Float64, ss4: Float64, ss5: Float64, ss6: Float64, ss7: Float64, sz1: Float64, sz2: Float64, sz3: Float64, sz11: Float64, sz12: Float64, sz13: Float64, sz21: Float64, sz22: Float64, sz23: Float64, sz31: Float64, sz32: Float64, sz33: Float64, xgh2: Float64, xgh3: Float64, xgh4: Float64, xh2: Float64, xh3: Float64, xi2: Float64, xi3: Float64, xl2: Float64, xl3: Float64, xl4: Float64, nm: Float64, z1: Float64, z2: Float64, z3: Float64, z11: Float64, z12: Float64, z13: Float64, z21: Float64, z22: Float64, z23: Float64, z31: Float64, z32: Float64, z33: Float64, zmol: Float64, zmos: Float64):
        self.snodm = snodm
        self.cnodm = cnodm
        self.sinim = sinim
        self.cosim = cosim
        self.sinomm = sinomm
        self.cosomm = cosomm
        self.day = day
        self.e3 = e3
        self.ee2 = ee2
        self.em = em
        self.emsq = emsq
        self.gam = gam
        self.peo = peo
        self.pgho = pgho
        self.pho = pho
        self.pinco = pinco
        self.plo = plo
        self.rtemsq = rtemsq
        self.se2 = se2
        self.se3 = se3
        self.sgh2 = sgh2
        self.sgh3 = sgh3
        self.sgh4 = sgh4
        self.sh2 = sh2
        self.sh3 = sh3
        self.si2 = si2
        self.si3 = si3
        self.sl2 = sl2
        self.sl3 = sl3
        self.sl4 = sl4
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5 = s5
        self.s6 = s6
        self.s7 = s7
        self.ss1 = ss1
        self.ss2 = ss2
        self.ss3 = ss3
        self.ss4 = ss4
        self.ss5 = ss5
        self.ss6 = ss6
        self.ss7 = ss7
        self.sz1 = sz1
        self.sz2 = sz2
        self.sz3 = sz3
        self.sz11 = sz11
        self.sz12 = sz12
        self.sz13 = sz13
        self.sz21 = sz21
        self.sz22 = sz22
        self.sz23 = sz23
        self.sz31 = sz31
        self.sz32 = sz32
        self.sz33 = sz33
        self.xgh2 = xgh2
        self.xgh3 = xgh3
        self.xgh4 = xgh4
        self.xh2 = xh2
        self.xh3 = xh3
        self.xi2 = xi2
        self.xi3 = xi3
        self.xl2 = xl2
        self.xl3 = xl3
        self.xl4 = xl4
        self.nm = nm
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.z11 = z11
        self.z12 = z12
        self.z13 = z13
        self.z21 = z21
        self.z22 = z22
        self.z23 = z23
        self.z31 = z31
        self.z32 = z32
        self.z33 = z33
        self.zmol = zmol
        self.zmos = zmos

def _dscom(
    epoch: Float64,
    ep: Float64,
    argpp: Float64,
    tc: Float64,
    inclp: Float64,
    nodep: Float64,
    np: Float64,
) -> DscomResult:
    #  -------------------------- constants -------------------------
    var zes = 0.01675
    var zel = 0.05490
    var c1ss = 2.9864797e-6
    var c1l = 4.7968065e-7
    var zsinis = 0.39785416
    var zcosis = 0.91744867
    var zcosgs = 0.1945905
    var zsings = -0.98088458

    #  --------------------- local variables ------------------------
    var nm = np
    var em = ep
    var snodm = sin(nodep)
    var cnodm = cos(nodep)
    var sinomm = sin(argpp)
    var cosomm = cos(argpp)
    var sinim = sin(inclp)
    var cosim = cos(inclp)
    var emsq = em * em
    var betasq = 1.0 - emsq
    var rtemsq = sqrt(betasq)

    #  ----------------- initialize lunar solar terms ---------------
    var peo = 0.0
    var pinco = 0.0
    var plo = 0.0
    var pgho = 0.0
    var pho = 0.0
    var day = epoch + 18261.5 + tc / 1440.0
    var xnodce = (4.5236020 - 9.2422029e-4 * day) % twopi
    var stem = sin(xnodce)
    var ctem = cos(xnodce)
    var zcosil = 0.91375164 - 0.03568096 * ctem
    var zsinil = sqrt(1.0 - zcosil * zcosil)
    var zsinhl = 0.089683511 * stem / zsinil
    var zcoshl = sqrt(1.0 - zsinhl * zsinhl)
    var gam = 5.8351514 + 0.0019443680 * day
    var zx = 0.39785416 * stem / zsinil
    var zy = zcoshl * ctem + 0.91744867 * zsinhl * stem
    zx = atan2(zx, zy)
    zx = gam + zx - xnodce
    var zcosgl = cos(zx)
    var zsingl = sin(zx)

    #  ------------------------- do solar terms ---------------------
    var zcosg = zcosgs
    var zsing = zsings
    var zcosi = zcosis
    var zsini = zsinis
    var zcosh = cnodm
    var zsinh = snodm
    var cc = c1ss
    var xnoi = 1.0 / nm

    # Mojo needs this loop's values to outlive it, because the solar and lunar
    # term assembly that follows reads the state left by the second pass; the
    # loop body below is otherwise upstream's, unchanged.
    var a1: Float64
    var a3: Float64
    var a7: Float64
    var a8: Float64
    var a9: Float64
    var a10: Float64
    var a2: Float64
    var a4: Float64
    var a5: Float64
    var a6: Float64
    var x1: Float64
    var x2: Float64
    var x3: Float64
    var x4: Float64
    var x5: Float64
    var x6: Float64
    var x7: Float64
    var x8: Float64
    var z31: Float64 = 0.0
    var z32: Float64 = 0.0
    var z33: Float64 = 0.0
    var z1: Float64 = 0.0
    var z2: Float64 = 0.0
    var z3: Float64 = 0.0
    var z11: Float64 = 0.0
    var z12: Float64 = 0.0
    var z13: Float64 = 0.0
    var z21: Float64 = 0.0
    var z22: Float64 = 0.0
    var z23: Float64 = 0.0
    var s3: Float64 = 0.0
    var s2: Float64 = 0.0
    var s4: Float64 = 0.0
    var s1: Float64 = 0.0
    var s5: Float64 = 0.0
    var s6: Float64 = 0.0
    var s7: Float64 = 0.0
    var ss1: Float64 = 0.0
    var ss2: Float64 = 0.0
    var ss3: Float64 = 0.0
    var ss4: Float64 = 0.0
    var ss5: Float64 = 0.0
    var ss6: Float64 = 0.0
    var ss7: Float64 = 0.0
    var sz1: Float64 = 0.0
    var sz2: Float64 = 0.0
    var sz3: Float64 = 0.0
    var sz11: Float64 = 0.0
    var sz12: Float64 = 0.0
    var sz13: Float64 = 0.0
    var sz21: Float64 = 0.0
    var sz22: Float64 = 0.0
    var sz23: Float64 = 0.0
    var sz31: Float64 = 0.0
    var sz32: Float64 = 0.0
    var sz33: Float64 = 0.0
    for lsflg in range(1, 3):

        a1 = zcosg * zcosh + zsing * zcosi * zsinh
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh
        a8 = zsing * zsini
        a9 = zsing * zsinh + zcosg * zcosi * zcosh
        a10 = zcosg * zsini
        a2 = cosim * a7 + sinim * a8
        a4 = cosim * a9 + sinim * a10
        a5 = -sinim * a7 + cosim * a8
        a6 = -sinim * a9 + cosim * a10

        x1 = a1 * cosomm + a2 * sinomm
        x2 = a3 * cosomm + a4 * sinomm
        x3 = -a1 * sinomm + a2 * cosomm
        x4 = -a3 * sinomm + a4 * cosomm
        x5 = a5 * sinomm
        x6 = a6 * sinomm
        x7 = a5 * cosomm
        x8 = a6 * cosomm

        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq
        z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5)
        z12 = -6.0 * (a1 * a6 + a3 * a5) + emsq * \
              (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5))
        z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6)
        z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7)
        z22 = 6.0 * (a4 * a5 + a2 * a6) + emsq * \
              (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8))
        z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8)
        z1 = z1 + z1 + betasq * z31
        z2 = z2 + z2 + betasq * z32
        z3 = z3 + z3 + betasq * z33
        s3 = cc * xnoi
        s2 = -0.5 * s3 / rtemsq
        s4 = s3 * rtemsq
        s1 = -15.0 * em * s4
        s5 = x1 * x3 + x2 * x4
        s6 = x2 * x3 + x1 * x4
        s7 = x2 * x4 - x1 * x3

        #  ----------------------- do lunar terms -------------------
        if lsflg == 1:

            ss1 = s1
            ss2 = s2
            ss3 = s3
            ss4 = s4
            ss5 = s5
            ss6 = s6
            ss7 = s7
            sz1 = z1
            sz2 = z2
            sz3 = z3
            sz11 = z11
            sz12 = z12
            sz13 = z13
            sz21 = z21
            sz22 = z22
            sz23 = z23
            sz31 = z31
            sz32 = z32
            sz33 = z33
            zcosg = zcosgl
            zsing = zsingl
            zcosi = zcosil
            zsini = zsinil
            zcosh = zcoshl * cnodm + zsinhl * snodm
            zsinh = snodm * zcoshl - cnodm * zsinhl
            cc = c1l

    var zmol = (4.7199672 + 0.22997150 * day - gam) % twopi
    var zmos = (6.2565837 + 0.017201977 * day) % twopi

    #  ------------------------ do solar terms ----------------------
    var se2 = 2.0 * ss1 * ss6
    var se3 = 2.0 * ss1 * ss7
    var si2 = 2.0 * ss2 * sz12
    var si3 = 2.0 * ss2 * (sz13 - sz11)
    var sl2 = -2.0 * ss3 * sz2
    var sl3 = -2.0 * ss3 * (sz3 - sz1)
    var sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes
    var sgh2 = 2.0 * ss4 * sz32
    var sgh3 = 2.0 * ss4 * (sz33 - sz31)
    var sgh4 = -18.0 * ss4 * zes
    var sh2 = -2.0 * ss2 * sz22
    var sh3 = -2.0 * ss2 * (sz23 - sz21)

    #  ------------------------ do lunar terms ----------------------
    var ee2 = 2.0 * s1 * s6
    var e3 = 2.0 * s1 * s7
    var xi2 = 2.0 * s2 * z12
    var xi3 = 2.0 * s2 * (z13 - z11)
    var xl2 = -2.0 * s3 * z2
    var xl3 = -2.0 * s3 * (z3 - z1)
    var xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel
    var xgh2 = 2.0 * s4 * z32
    var xgh3 = 2.0 * s4 * (z33 - z31)
    var xgh4 = -18.0 * s4 * zel
    var xh2 = -2.0 * s2 * z22
    var xh3 = -2.0 * s2 * (z23 - z21)

    return DscomResult(
        snodm, cnodm, sinim, cosim, sinomm,
        cosomm, day, e3, ee2, em,
        emsq, gam, peo, pgho, pho,
        pinco, plo, rtemsq, se2, se3,
        sgh2, sgh3, sgh4, sh2, sh3,
        si2, si3, sl2, sl3, sl4,
        s1, s2, s3, s4, s5,
        s6, s7, ss1, ss2, ss3,
        ss4, ss5, ss6, ss7, sz1,
        sz2, sz3, sz11, sz12, sz13,
        sz21, sz22, sz23, sz31, sz32,
        sz33, xgh2, xgh3, xgh4, xh2,
        xh3, xi2, xi3, xl2, xl3,
        xl4, nm, z1, z2, z3,
        z11, z12, z13, z21, z22,
        z23, z31, z32, z33, zmol,
        zmos,
    )

# ----------------------------------------------------------------------------

#  procedure dsinit

#

#  this procedure provides deep space contributions to mean motion dot due

#    to geopotential resonance with half day and one day orbits.

# ----------------------------------------------------------------------------

struct DsinitResult(ImplicitlyCopyable, Copyable, Movable):
    var em: Float64
    var argpm: Float64
    var inclm: Float64
    var mm: Float64
    var nm: Float64
    var nodem: Float64
    var irez: Int
    var atime: Float64
    var d2201: Float64
    var d2211: Float64
    var d3210: Float64
    var d3222: Float64
    var d4410: Float64
    var d4422: Float64
    var d5220: Float64
    var d5232: Float64
    var d5421: Float64
    var d5433: Float64
    var dedt: Float64
    var didt: Float64
    var dmdt: Float64
    var dndt: Float64
    var dnodt: Float64
    var domdt: Float64
    var del1: Float64
    var del2: Float64
    var del3: Float64
    var xfact: Float64
    var xlamo: Float64
    var xli: Float64
    var xni: Float64

    def __init__(out self, em: Float64, argpm: Float64, inclm: Float64, mm: Float64, nm: Float64, nodem: Float64, irez: Int, atime: Float64, d2201: Float64, d2211: Float64, d3210: Float64, d3222: Float64, d4410: Float64, d4422: Float64, d5220: Float64, d5232: Float64, d5421: Float64, d5433: Float64, dedt: Float64, didt: Float64, dmdt: Float64, dndt: Float64, dnodt: Float64, domdt: Float64, del1: Float64, del2: Float64, del3: Float64, xfact: Float64, xlamo: Float64, xli: Float64, xni: Float64):
        self.em = em
        self.argpm = argpm
        self.inclm = inclm
        self.mm = mm
        self.nm = nm
        self.nodem = nodem
        self.irez = irez
        self.atime = atime
        self.d2201 = d2201
        self.d2211 = d2211
        self.d3210 = d3210
        self.d3222 = d3222
        self.d4410 = d4410
        self.d4422 = d4422
        self.d5220 = d5220
        self.d5232 = d5232
        self.d5421 = d5421
        self.d5433 = d5433
        self.dedt = dedt
        self.didt = didt
        self.dmdt = dmdt
        self.dndt = dndt
        self.dnodt = dnodt
        self.domdt = domdt
        self.del1 = del1
        self.del2 = del2
        self.del3 = del3
        self.xfact = xfact
        self.xlamo = xlamo
        self.xli = xli
        self.xni = xni

def _dsinit(
    xke: Float64,
    cosim: Float64,
    emsq_: Float64,
    argpo: Float64,
    s1: Float64,
    s2: Float64,
    s3: Float64,
    s4: Float64,
    s5: Float64,
    sinim: Float64,
    ss1: Float64,
    ss2: Float64,
    ss3: Float64,
    ss4: Float64,
    ss5: Float64,
    sz1: Float64,
    sz3: Float64,
    sz11: Float64,
    sz13: Float64,
    sz21: Float64,
    sz23: Float64,
    sz31: Float64,
    sz33: Float64,
    t: Float64,
    tc: Float64,
    gsto: Float64,
    mo: Float64,
    mdot: Float64,
    no: Float64,
    nodeo: Float64,
    nodedot: Float64,
    xpidot: Float64,
    z1: Float64,
    z3: Float64,
    z11: Float64,
    z13: Float64,
    z21: Float64,
    z23: Float64,
    z31: Float64,
    z33: Float64,
    ecco: Float64,
    eccsq: Float64,
    em_: Float64,
    argpm_: Float64,
    inclm_: Float64,
    mm_: Float64,
    nm_: Float64,
    nodem_: Float64,
) -> DsinitResult:

    # Mojo binds parameters immutably and will not let a body rebind
    # one, while upstream treats these values as in/out.  Take mutable
    # locals of the same names on the way in; the call is unchanged.
    var em = em_
    var emsq = emsq_
    var argpm = argpm_
    var inclm = inclm_
    var mm = mm_
    var nm = nm_
    var nodem = nodem_
    var q22 = 1.7891679e-6
    var q31 = 2.1460748e-6
    var q33 = 2.2123015e-7
    var root22 = 1.7891679e-6
    var root44 = 7.3636953e-9
    var root54 = 2.1765803e-9
    var rptim = 4.37526908801129966e-3  # equates to 7.29211514668855e-5 rad/sec
    var root32 = 3.7393792e-7
    var root52 = 1.1428639e-7
    var znl = 1.5835218e-4
    var zns = 1.19459e-5

    # sgp4fix identify constants and allow alternate values
    # just xke is used here so pass it in rather than have multiple calls

    #  -------------------- deep space initialization ------------
    var irez = 0
    if nm > 0.0034906585 and nm < 0.0052359877:
        irez = 1
    if nm >= 8.26e-3 and nm <= 9.24e-3 and em >= 0.5:
        irez = 2

    #  ------------------------ do solar terms -------------------
    var ses = ss1 * zns * ss5
    var sis = ss2 * zns * (sz11 + sz13)
    var sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    var sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    var shs = -zns * ss2 * (sz21 + sz23)
    #  sgp4fix for 180 deg incl
    if inclm < 5.2359877e-2 or inclm > pi - 5.2359877e-2:
        shs = 0.0
    if sinim != 0.0:
        shs = shs / sinim
    var sgs = sghs - cosim * shs

    #  ------------------------- do lunar terms ------------------
    var dedt = ses + s1 * znl * s5
    var didt = sis + s2 * znl * (z11 + z13)
    var dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    var sghl = s4 * znl * (z31 + z33 - 6.0)
    var shll = -znl * s2 * (z21 + z23)
    #  sgp4fix for 180 deg incl
    if inclm < 5.2359877e-2 or inclm > pi - 5.2359877e-2:
        shll = 0.0
    var domdt = sgs + sghl
    var dnodt = shs
    if sinim != 0.0:

        domdt = domdt - cosim / sinim * shll
        dnodt = dnodt + shll / sinim

    #  ----------- calculate deep space resonance effects --------
    var dndt = 0.0
    var theta = (gsto + tc * rptim) % twopi
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t
    # --- upstream comment block, kept for navigation ---
    # //   sgp4fix for negative inclinations
    # //   the following if statement should be commented out
    # //if (inclm < 0.0)
    # //  {
    # //    inclm  = -inclm;
    # //    argpm  = argpm - pi;
    # //    nodem = nodem + pi;
    # //  }

    #  -------------- initialize the resonance terms -------------
    var del1 = 0.0
    var del2 = 0.0
    var del3 = 0.0
    var xfact = 0.0
    var xlamo = 0.0

    # Mojo needs the resonance block's values to outlive it, because the tuple
    # at the end of the routine reports them whether the block ran or not.
    var aonv: Float64
    var cosisq: Float64
    var emo: Float64
    var emsqo: Float64
    var eoc: Float64
    var g201: Float64
    var g211: Float64
    var g310: Float64
    var g322: Float64
    var g410: Float64
    var g422: Float64
    var g520: Float64
    var g533: Float64
    var g521: Float64
    var g532: Float64
    var sini2: Float64
    var f220: Float64
    var f221: Float64
    var f321: Float64
    var f322: Float64
    var f441: Float64
    var f442: Float64
    var f522: Float64
    var f523: Float64
    var f542: Float64
    var f543: Float64
    var xno2: Float64
    var ainv2: Float64
    var temp1: Float64
    var temp: Float64
    var d2201: Float64 = 0.0
    var d2211: Float64 = 0.0
    var d3210: Float64 = 0.0
    var d3222: Float64 = 0.0
    var d4410: Float64 = 0.0
    var d4422: Float64 = 0.0
    var d5220: Float64 = 0.0
    var d5232: Float64 = 0.0
    var d5421: Float64 = 0.0
    var d5433: Float64 = 0.0
    var g200: Float64
    var g300: Float64
    var f311: Float64
    var f330: Float64
    var xli: Float64 = 0.0
    var xni: Float64 = 0.0
    var atime: Float64 = 0.0
    if irez != 0:

        aonv = pow(nm / xke, x2o3)

        #  ---------- geopotential resonance for 12 hour orbits ------
        if irez == 2:

            cosisq = cosim * cosim
            emo = em
            em = ecco
            emsqo = emsq
            emsq = eccsq

    # Mojo wants these declared once, ahead of the eccentricity branches that
    # pick between the two sets of coefficients; the branches themselves are
    # upstream's, unchanged.
            eoc = em * emsq
            g201 = -0.306 - (em - 0.64) * 0.440

            if em <= 0.65:

                g211 = 3.616 - 13.2470 * em + 16.2900 * emsq
                g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc
                g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc
                g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc

            else:

                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc
                g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc
                if em > 0.715:
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc
                else:
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq

            if em < 0.7:

                g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc
                g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc

            else:

                g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc
                g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc

            sini2 = sinim * sinim
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
            f221 = 1.5 * sini2
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            f441 = 35.0 * sini2 * f220
            f442 = 39.3750 * sini2 * sini2
            f522 = 9.84375 * sinim * (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq) +
                    0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq))
            f523 = sinim * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim +
                   10.0 * cosisq) + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq))
            f542 = 29.53125 * sinim * (2.0 - 8.0 * cosim + cosisq *
                   (-12.0 + 8.0 * cosim + 10.0 * cosisq))
            f543 = 29.53125 * sinim * (-2.0 - 8.0 * cosim + cosisq *
                   (12.0 + 8.0 * cosim - 10.0 * cosisq))
            xno2 = nm * nm
            ainv2 = aonv * aonv
            temp1 = 3.0 * xno2 * ainv2
            temp = temp1 * root22
            d2201 = temp * f220 * g201
            d2211 = temp * f221 * g211
            temp1 = temp1 * aonv
            temp = temp1 * root32
            d3210 = temp * f321 * g310
            d3222 = temp * f322 * g322
            temp1 = temp1 * aonv
            temp = 2.0 * temp1 * root44
            d4410 = temp * f441 * g410
            d4422 = temp * f442 * g422
            temp1 = temp1 * aonv
            temp = temp1 * root52
            d5220 = temp * f522 * g520
            d5232 = temp * f523 * g532
            temp = 2.0 * temp1 * root54
            d5421 = temp * f542 * g521
            d5433 = temp * f543 * g533
            xlamo = (mo + nodeo + nodeo - theta - theta) % twopi
            xfact = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no
            em = emo
            emsq = emsqo

        #  ---------------- synchronous resonance terms --------------
        if irez == 1:

            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
            g310 = 1.0 + 2.0 * emsq
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
            f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim)
            f330 = 1.0 + cosim
            f330 = 1.875 * f330 * f330 * f330
            del1 = 3.0 * nm * nm * aonv * aonv
            del2 = 2.0 * del1 * f220 * g200 * q22
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv
            del1 = del1 * f311 * g310 * q31 * aonv
            xlamo = (mo + nodeo + argpo - theta) % twopi
            xfact = mdot + xpidot - rptim + dmdt + domdt + dnodt - no

        #  ------------ for sgp4, initialize the integrator ----------
        xli = xlamo
        xni = no
        atime = 0.0
        nm = no + dndt

    return DsinitResult(
        em, argpm, inclm, mm,
        nm, nodem,
        irez, atime,
        d2201, d2211, d3210, d3222,
        d4410, d4422, d5220, d5232,
        d5421, d5433, dedt, didt,
        dmdt, dndt, dnodt, domdt,
        del1, del2, del3, xfact,
        xlamo, xli, xni,
    )

# ----------------------------------------------------------------------------

#  procedure dspace

#

#  this procedure provides deep space contributions to mean elements for

#    perturbing third body.

# ----------------------------------------------------------------------------

struct DspaceResult(ImplicitlyCopyable, Copyable, Movable):
    var atime: Float64
    var em: Float64
    var argpm: Float64
    var inclm: Float64
    var xli: Float64
    var mm: Float64
    var xni: Float64
    var nodem: Float64
    var dndt: Float64
    var nm: Float64

    def __init__(out self, atime: Float64, em: Float64, argpm: Float64, inclm: Float64, xli: Float64, mm: Float64, xni: Float64, nodem: Float64, dndt: Float64, nm: Float64):
        self.atime = atime
        self.em = em
        self.argpm = argpm
        self.inclm = inclm
        self.xli = xli
        self.mm = mm
        self.xni = xni
        self.nodem = nodem
        self.dndt = dndt
        self.nm = nm

def _dspace(
    irez: Int,
    d2201: Float64,
    d2211: Float64,
    d3210: Float64,
    d3222: Float64,
    d4410: Float64,
    d4422: Float64,
    d5220: Float64,
    d5232: Float64,
    d5421: Float64,
    d5433: Float64,
    dedt: Float64,
    del1: Float64,
    del2: Float64,
    del3: Float64,
    didt: Float64,
    dmdt: Float64,
    dnodt: Float64,
    domdt: Float64,
    argpo: Float64,
    argpdot: Float64,
    t: Float64,
    tc: Float64,
    gsto: Float64,
    xfact: Float64,
    xlamo: Float64,
    no: Float64,
    atime_: Float64,
    em_: Float64,
    argpm_: Float64,
    inclm_: Float64,
    xli_: Float64,
    mm_: Float64,
    xni_: Float64,
    nodem_: Float64,
    nm_: Float64,
) -> DspaceResult:

    # Mojo binds parameters immutably and will not let a body rebind
    # one, while upstream treats these values as in/out.  Take mutable
    # locals of the same names on the way in; the call is unchanged.
    var atime = atime_
    var em = em_
    var argpm = argpm_
    var inclm = inclm_
    var xli = xli_
    var mm = mm_
    var xni = xni_
    var nodem = nodem_
    var nm = nm_
    var fasx2 = 0.13130908
    var fasx4 = 2.8843198
    var fasx6 = 0.37448087
    var g22 = 5.7686396
    var g32 = 0.95240898
    var g44 = 1.8014998
    var g52 = 1.0508330
    var g54 = 4.4108898
    var rptim = 4.37526908801129966e-3  # equates to 7.29211514668855e-5 rad/sec
    var stepp = 720.0
    var stepn = -720.0
    var step2 = 259200.0

    #  ----------- calculate deep space resonance effects -----------
    var dndt: Float64 = 0.0
    var theta = (gsto + tc * rptim) % twopi
    em = em + dedt * t

    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t

    # --- upstream comment block, kept for navigation ---
    # //   sgp4fix for negative inclinations
    # //   the following if statement should be commented out
    # //  if (inclm < 0.0)
    # // {
    # //    inclm = -inclm;
    # //    argpm = argpm - pi;
    # //    nodem = nodem + pi;
    # // }
    #
    # /* - update resonances : numerical (euler-maclaurin) integration - */
    # /* ------------------------- epoch restart ----------------------  */
    # //   sgp4fix for propagator problems
    # //   the following integration works for negative time steps and periods
    # //   the specific changes are unknown because the original code was so convoluted
    #
    # // sgp4fix take out atime = 0.0 and fix for faster operation
    var ft = 0.0

    # Mojo needs the integrator's values to outlive it, because the tuple at the
    # end of the routine reports them whether the block ran or not.
    var delt: Float64
    var iretn: Float64
    var xndt: Float64 = 0.0
    var xldot: Float64 = 0.0
    var xnddt: Float64 = 0.0
    var xomi: Float64
    var x2omi: Float64
    var x2li: Float64
    var xl: Float64
    if irez != 0:

        #  sgp4fix streamline check
        if atime == 0.0 or t * atime <= 0.0 or abs(t) < abs(atime):

            atime = 0.0
            xni = no
            xli = xlamo

        # sgp4fix move check outside loop
        if t > 0.0:
            delt = stepp
        else:
            delt = stepn

        iretn = 381  # added for do loop
        # iret  =   0; # added for loop
        while iretn == 381:

            #  ------------------- dot terms calculated -------------
            #  ----------- near - synchronous resonance terms -------
            if irez != 2:

                xndt = del1 * sin(xli - fasx2) + del2 * sin(2.0 * (xli - fasx4)) + \
                       del3 * sin(3.0 * (xli - fasx6))
                xldot = xni + xfact
                xnddt = del1 * cos(xli - fasx2) + \
                        2.0 * del2 * cos(2.0 * (xli - fasx4)) + \
                        3.0 * del3 * cos(3.0 * (xli - fasx6))
                xnddt = xnddt * xldot

            else:

                # --------- near - half-day resonance terms --------
                xomi = argpo + argpdot * atime
                x2omi = xomi + xomi
                x2li = xli + xli
                xndt = (d2201 * sin(x2omi + xli - g22) + d2211 * sin(xli - g22) +
                      d3210 * sin(xomi + xli - g32) + d3222 * sin(-xomi + xli - g32) +
                      d4410 * sin(x2omi + x2li - g44) + d4422 * sin(x2li - g44) +
                      d5220 * sin(xomi + xli - g52) + d5232 * sin(-xomi + xli - g52) +
                      d5421 * sin(xomi + x2li - g54) + d5433 * sin(-xomi + x2li - g54))
                xldot = xni + xfact
                xnddt = (d2201 * cos(x2omi + xli - g22) + d2211 * cos(xli - g22) +
                      d3210 * cos(xomi + xli - g32) + d3222 * cos(-xomi + xli - g32) +
                      d5220 * cos(xomi + xli - g52) + d5232 * cos(-xomi + xli - g52) +
                      2.0 * (d4410 * cos(x2omi + x2li - g44) +
                      d4422 * cos(x2li - g44) + d5421 * cos(xomi + x2li - g54) +
                      d5433 * cos(-xomi + x2li - g54)))
                xnddt = xnddt * xldot

            #  ----------------------- integrator -------------------
            #  sgp4fix move end checks to end of routine
            if abs(t - atime) >= stepp:
                # iret  = 0;
                iretn = 381

            else:
                ft = t - atime
                iretn = 0

            if iretn == 381:

                xli = xli + xldot * delt + xndt * step2
                xni = xni + xndt * delt + xnddt * step2
                atime = atime + delt

        nm = xni + xndt * ft + xnddt * ft * ft * 0.5
        xl = xli + xldot * ft + xndt * ft * ft * 0.5
        if irez != 1:
            mm = xl - 2.0 * nodem + 2.0 * theta
            dndt = nm - no

        else:
            mm = xl - nodem - argpm + theta
            dndt = nm - no

        nm = no + dndt

    return DspaceResult(atime, em, argpm, inclm, xli, mm, xni, nodem, dndt, nm)

# ----------------------------------------------------------------------------

#  procedure initl

#

#  this procedure initializes the spg4 propagator.

# ----------------------------------------------------------------------------

struct InitlResult(ImplicitlyCopyable, Copyable, Movable):
    var no: Float64
    var method: Int
    var ainv: Float64
    var ao: Float64
    var con41: Float64
    var con42: Float64
    var cosio: Float64
    var cosio2: Float64
    var eccsq: Float64
    var omeosq: Float64
    var posq: Float64
    var rp: Float64
    var rteosq: Float64
    var sinio: Float64
    var gsto: Float64

    def __init__(out self, no: Float64, method: Int, ainv: Float64, ao: Float64, con41: Float64, con42: Float64, cosio: Float64, cosio2: Float64, eccsq: Float64, omeosq: Float64, posq: Float64, rp: Float64, rteosq: Float64, sinio: Float64, gsto: Float64):
        self.no = no
        self.method = method
        self.ainv = ainv
        self.ao = ao
        self.con41 = con41
        self.con42 = con42
        self.cosio = cosio
        self.cosio2 = cosio2
        self.eccsq = eccsq
        self.omeosq = omeosq
        self.posq = posq
        self.rp = rp
        self.rteosq = rteosq
        self.sinio = sinio
        self.gsto = gsto

def _initl(
    xke: Float64,
    j2: Float64,
    ecco: Float64,
    epoch: Float64,
    inclo: Float64,
    no_: Float64,
    method_: Int,
    opsmode: Int,
) -> InitlResult:
    # sgp4fix use old way of finding gst

    #  ----------------------- earth constants ----------------------
    # sgp4fix identify constants and allow alternate values
    # only xke and j2 are used here so pass them in directly
    #
    # Mojo binds parameters immutably and will not let a body rebind one, while
    # upstream returns `no` and `method` after rewriting them.  Take mutable
    # locals of the same names; the call is unchanged.
    var no = no_
    var method: Int

    #  ------------- calculate auxillary epoch quantities ----------
    var eccsq = ecco * ecco
    var omeosq = 1.0 - eccsq
    var rteosq = sqrt(omeosq)
    var cosio = cos(inclo)
    var cosio2 = cosio * cosio

    #  ------------------ un-kozai the mean motion -----------------
    var ak = pow(xke / no, x2o3)
    var d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    var del_ = d1 / (ak * ak)
    var adel = ak * (1.0 - del_ * del_ - del_ *
            (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    no = no / (1.0 + del_)

    var ao = pow(xke / no, x2o3)
    var sinio = sin(inclo)
    var po = ao * omeosq
    var con42 = 1.0 - 5.0 * cosio2
    var con41 = -con42 - cosio2 - cosio2
    var ainv = 1.0 / ao
    var posq = po * po
    var rp = ao * (1.0 - ecco)
    method = METHOD_N

    # Mojo needs gsto declared once, because either branch below may set it.
    var gsto: Float64
    #  sgp4fix modern approach to finding sidereal time
    if opsmode == OPSMODE_A:

        #  sgp4fix use old way of finding gst
        #  count integer number of days from 0 jan 1970
        var ts70 = epoch - 7305.0
        var ds70 = floor((ts70 + 1.0e-8) / 1.0)
        var tfrac = ts70 - ds70
        #  find greenwich location at epoch
        var c1 = 1.72027916940703639e-2
        var thgr70 = 1.7321343856509374
        var fk5r = 5.07551419432269442e-15
        var c1p2p = c1 + twopi
        gsto = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % twopi
        if gsto < 0.0:
            gsto = gsto + twopi

    else:
        gsto = gstime(epoch + 2433281.5)

    return InitlResult(
        no,
        method,
        ainv, ao, con41, con42, cosio,
        cosio2, eccsq, omeosq, posq,
        rp, rteosq, sinio, gsto,
    )

# ----------------------------------------------------------------------------

#  procedure sgp4

#

#  this procedure is the sgp4 prediction model from space command.  this is an

#    updated and combined version of sgp4 and sdp4, which were originally

#    published separately in spacetrack report #3.

#

#  Upstream returns `(r, v)`, or `(False, False)` on error, and stashes the

#  reason in `satrec.error` / `satrec.error_message`.  A Mojo function has one

#  return value, so the pair comes back as `Sgp4Out`: `r`, `v`, and the number

#  that went out of range, which `python/mojosgp4` formats into upstream's

#  `error_message` string.  Upstream's `False` return is conveyed the way

#  upstream's own `Satrec.sgp4()` conveys it, through `satrec.error`, and

#  `r`/`v` hold upstream's module level `false = (_nan, _nan, _nan)`.

# ----------------------------------------------------------------------------

struct Vec3(ImplicitlyCopyable, Copyable, Movable):
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z

struct Sgp4Out(ImplicitlyCopyable, Copyable, Movable):
    var r: Vec3
    var v: Vec3
    var errval: Float64

    def __init__(out self, r: Vec3, v: Vec3, errval: Float64):
        self.r = r
        self.v = v
        self.errval = errval

def _nan_vec3() -> Vec3:
    return Vec3(nan[DType.float64](), nan[DType.float64](), nan[DType.float64]())

def sgp4(satrec: PtrSat, tsince: Float64) -> Sgp4Out:
    var mrt = 0.0
    var r = _nan_vec3()
    var v = _nan_vec3()
    var errval = 0.0

    # --- upstream comment block, kept for navigation ---
    # /* ------------------ set mathematical constants --------------- */
    # // sgp4fix divisor for divide by zero check on inclination
    # // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    # // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    var temp4 = 1.5e-12
    #  sgp4fix identify constants and allow alternate values
    # tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
    var vkmpersec = satrec[].radiusearthkm * satrec[].xke / 60.0

    #  --------------------- clear sgp4 error flag -----------------
    satrec[].t = tsince
    satrec[].error = 0

    #  ------- update for secular gravity and atmospheric drag -----
    var xmdf = satrec[].mo + satrec[].mdot * satrec[].t
    var argpdf = satrec[].argpo + satrec[].argpdot * satrec[].t
    var nodedf = satrec[].nodeo + satrec[].nodedot * satrec[].t
    var argpm = argpdf
    var mm = xmdf
    var t2 = satrec[].t * satrec[].t
    var nodem = nodedf + satrec[].nodecf * t2
    var tempa = 1.0 - satrec[].cc1 * satrec[].t
    var tempe = satrec[].bstar * satrec[].cc4 * satrec[].t
    var templ = satrec[].t2cof * t2

    if satrec[].isimp != 1:

        var delomg = satrec[].omgcof * satrec[].t
        #  sgp4fix use mutliply for speed instead of pow
        var delmtemp = 1.0 + satrec[].eta * cos(xmdf)
        var delm = satrec[].xmcof * \
               (delmtemp * delmtemp * delmtemp -
               satrec[].delmo)
        var temp = delomg + delm
        mm = xmdf + temp
        argpm = argpdf - temp
        var t3 = t2 * satrec[].t
        var t4 = t3 * satrec[].t
        tempa = tempa - satrec[].d2 * t2 - satrec[].d3 * t3 - \
                satrec[].d4 * t4
        tempe = tempe + satrec[].bstar * satrec[].cc5 * (sin(mm) -
                satrec[].sinmao)
        templ = templ + satrec[].t3cof * t3 + t4 * (satrec[].t4cof +
                satrec[].t * satrec[].t5cof)

    var nm = satrec[].no_unkozai
    var em = satrec[].ecco
    var inclm = satrec[].inclo
    if satrec[].method == METHOD_D:

        var tc = satrec[].t
        var ds = _dspace(
            satrec[].irez,
            satrec[].d2201, satrec[].d2211, satrec[].d3210,
            satrec[].d3222, satrec[].d4410, satrec[].d4422,
            satrec[].d5220, satrec[].d5232, satrec[].d5421,
            satrec[].d5433, satrec[].dedt, satrec[].del1,
            satrec[].del2, satrec[].del3, satrec[].didt,
            satrec[].dmdt, satrec[].dnodt, satrec[].domdt,
            satrec[].argpo, satrec[].argpdot, satrec[].t, tc,
            satrec[].gsto, satrec[].xfact, satrec[].xlamo,
            satrec[].no_unkozai, satrec[].atime,
            em, argpm, inclm, satrec[].xli, mm, satrec[].xni,
            nodem, nm,
        )
        # upstream's tuple unpacking names all ten results; `atime`, `xli`,
        # `xni` and `dndt` are carried in the satrec and nothing reads them here
        var atime = ds.atime
        em = ds.em
        argpm = ds.argpm
        inclm = ds.inclm
        var xli = ds.xli
        mm = ds.mm
        var xni = ds.xni
        nodem = ds.nodem
        var dndt = ds.dndt
        nm = ds.nm

    if nm <= 0.0:

        satrec[].error = 2
        return Sgp4Out(_nan_vec3(), _nan_vec3(), nm)

    var am = pow((satrec[].xke / nm), x2o3) * tempa * tempa
    nm = satrec[].xke / pow(am, 1.5)
    em = em - tempe

    #  fix tolerance for error recognition
    #  sgp4fix am is fixed from the previous nm check
    if em >= 1.0 or em < -0.001:  # || (am < 0.95)

        satrec[].error = 1
        return Sgp4Out(_nan_vec3(), _nan_vec3(), em)

    #  sgp4fix fix tolerance to avoid a divide by zero
    if em < 1.0e-6:
        em = 1.0e-6
    mm = mm + satrec[].no_kozai * templ
    var xlm = mm + argpm + nodem
    var emsq = em * em
    var temp = 1.0 - emsq

    nodem = nodem % twopi if nodem >= 0.0 else -(-nodem % twopi)
    argpm = argpm % twopi
    xlm = xlm % twopi
    mm = (xlm - argpm - nodem) % twopi

    # sgp4fix recover singly averaged mean elements
    satrec[].am = am
    satrec[].em = em
    satrec[].im = inclm
    satrec[].Om = nodem
    satrec[].om = argpm
    satrec[].mm = mm
    satrec[].nm = nm

    #  ----------------- compute extra mean quantities -------------
    var sinim = sin(inclm)
    var cosim = cos(inclm)

    #  -------------------- add lunar-solar periodics --------------
    var ep = em
    var xincp = inclm
    var argpp = argpm
    var nodep = nodem
    var mp = mm
    var sinip = sinim
    var cosip = cosim
    if satrec[].method == METHOD_D:

        var dp = _dpper(
            satrec, satrec[].inclo,
            INIT_N, ep, xincp, nodep, argpp, mp, satrec[].operationmode,
        )
        ep = dp.ep
        xincp = dp.inclp
        nodep = dp.nodep
        argpp = dp.argpp
        mp = dp.mp
        if xincp < 0.0:

            xincp = -xincp
            nodep = nodep + pi
            argpp = argpp - pi

        if ep < 0.0 or ep > 1.0:

            satrec[].error = 3
            return Sgp4Out(_nan_vec3(), _nan_vec3(), ep)

    #  -------------------- long period periodics ------------------
    if satrec[].method == METHOD_D:

        sinip = sin(xincp)
        cosip = cos(xincp)
        satrec[].aycof = -0.5 * satrec[].j3oj2 * sinip
        #  sgp4fix for divide by zero for xincp = 180 deg
        if abs(cosip + 1.0) > 1.5e-12:
            satrec[].xlcof = -0.25 * satrec[].j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip)
        else:
            satrec[].xlcof = -0.25 * satrec[].j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4

    var axnl = ep * cos(argpp)
    temp = 1.0 / (am * (1.0 - ep * ep))
    var aynl = ep * sin(argpp) + temp * satrec[].aycof
    var xl = mp + argpp + nodep + temp * satrec[].xlcof * axnl

    #  --------------------- solve kepler's equation ---------------
    var u = (xl - nodep) % twopi
    var eo1 = u
    var tem5 = 9999.9
    var ktr = 1
    #    sgp4fix for kepler iteration
    #    the following iteration needs better limits on corrections

    # Mojo needs the loop's last sineo1/coseo1 to survive the loop, since
    # upstream reads them below; the loop body is otherwise upstream's.
    var sineo1: Float64 = 0.0
    var coseo1: Float64 = 0.0
    while abs(tem5) >= 1.0e-12 and ktr <= 10:

        sineo1 = sin(eo1)
        coseo1 = cos(eo1)
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        if abs(tem5) >= 0.95:
            tem5 = 0.95 if tem5 > 0.0 else -0.95
        eo1 = eo1 + tem5
        ktr = ktr + 1

    #  ------------- short period preliminary quantities -----------
    var ecose = axnl * coseo1 + aynl * sineo1
    var esine = axnl * sineo1 - aynl * coseo1
    var el2 = axnl * axnl + aynl * aynl
    var pl = am * (1.0 - el2)
    if pl < 0.0:

        satrec[].error = 4
        return Sgp4Out(_nan_vec3(), _nan_vec3(), pl)

    else:

        var rl = am * (1.0 - ecose)
        var rdotl = sqrt(am) * esine / rl
        var rvdotl = sqrt(pl) / rl
        var betal = sqrt(1.0 - el2)
        temp = esine / (1.0 + betal)
        var sinu = am / rl * (sineo1 - aynl - axnl * temp)
        var cosu = am / rl * (coseo1 - axnl + aynl * temp)
        var su = atan2(sinu, cosu)
        var sin2u = (cosu + cosu) * sinu
        var cos2u = 1.0 - 2.0 * sinu * sinu
        temp = 1.0 / pl
        var temp1 = 0.5 * satrec[].j2 * temp
        var temp2 = temp1 * temp

        #  -------------- update for short period periodics ------------
        if satrec[].method == METHOD_D:

            var cosisq = cosip * cosip
            satrec[].con41 = 3.0 * cosisq - 1.0
            satrec[].x1mth2 = 1.0 - cosisq
            satrec[].x7thm1 = 7.0 * cosisq - 1.0

        mrt = rl * (1.0 - 1.5 * temp2 * betal * satrec[].con41) + \
              0.5 * temp1 * satrec[].x1mth2 * cos2u
        su = su - 0.25 * temp2 * satrec[].x7thm1 * sin2u
        var xnode = nodep + 1.5 * temp2 * cosip * sin2u
        var xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u
        var mvt = rdotl - nm * temp1 * satrec[].x1mth2 * sin2u / satrec[].xke
        var rvdot = rvdotl + nm * temp1 * (satrec[].x1mth2 * cos2u +
                1.5 * satrec[].con41) / satrec[].xke

        #  --------------------- orientation vectors -------------------
        var sinsu = sin(su)
        var cossu = cos(su)
        var snod = sin(xnode)
        var cnod = cos(xnode)
        var sini = sin(xinc)
        var cosi = cos(xinc)
        var xmx = -snod * cosi
        var xmy = cnod * cosi
        var ux = xmx * sinsu + cnod * cossu
        var uy = xmy * sinsu + snod * cossu
        var uz = sini * sinsu
        var vx = xmx * cossu - cnod * sinsu
        var vy = xmy * cossu - snod * sinsu
        var vz = sini * cossu

        #  --------- position and velocity (in km and km/sec) ----------
        var _mr = mrt * satrec[].radiusearthkm
        r = Vec3(_mr * ux, _mr * uy, _mr * uz)
        v = Vec3((mvt * ux + rvdot * vx) * vkmpersec,
                 (mvt * uy + rvdot * vy) * vkmpersec,
                 (mvt * uz + rvdot * vz) * vkmpersec)

    #  sgp4fix for decaying satellites
    if mrt < 1.0:

        satrec[].error = 6
        errval = mrt

    return Sgp4Out(r, v, errval)

# ----------------------------------------------------------------------------

#  procedure sgp4init

#

#  this procedure initializes variables for sgp4.  Placed after `sgp4` because

#  it finishes by propagating to t = 0, and Mojo binds names in file order.

# ----------------------------------------------------------------------------

def sgp4init(
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
    satrec: PtrSat,
) -> Bool:
    # --- upstream comment block, kept for navigation ---
    # /* ------------------------ initialization --------------------- */
    # // sgp4fix divisor for divide by zero check on inclination
    # // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
    # // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
    var temp4 = 1.5e-12

    #  ----------- set all near earth variables to zero ------------
    satrec[].isimp = 0
    satrec[].method = METHOD_N
    satrec[].aycof = 0.0
    satrec[].con41 = 0.0
    satrec[].cc1 = 0.0
    satrec[].cc4 = 0.0
    satrec[].cc5 = 0.0
    satrec[].d2 = 0.0
    satrec[].d3 = 0.0
    satrec[].d4 = 0.0
    satrec[].delmo = 0.0
    satrec[].eta = 0.0
    satrec[].argpdot = 0.0
    satrec[].omgcof = 0.0
    satrec[].sinmao = 0.0
    satrec[].t = 0.0
    satrec[].t2cof = 0.0
    satrec[].t3cof = 0.0
    satrec[].t4cof = 0.0
    satrec[].t5cof = 0.0
    satrec[].x1mth2 = 0.0
    satrec[].x7thm1 = 0.0
    satrec[].mdot = 0.0
    satrec[].nodedot = 0.0
    satrec[].xlcof = 0.0
    satrec[].xmcof = 0.0
    satrec[].nodecf = 0.0

    #  ----------- set all deep space variables to zero ------------
    satrec[].irez = 0
    satrec[].d2201 = 0.0
    satrec[].d2211 = 0.0
    satrec[].d3210 = 0.0
    satrec[].d3222 = 0.0
    satrec[].d4410 = 0.0
    satrec[].d4422 = 0.0
    satrec[].d5220 = 0.0
    satrec[].d5232 = 0.0
    satrec[].d5421 = 0.0
    satrec[].d5433 = 0.0
    satrec[].dedt = 0.0
    satrec[].del1 = 0.0
    satrec[].del2 = 0.0
    satrec[].del3 = 0.0
    satrec[].didt = 0.0
    satrec[].dmdt = 0.0
    satrec[].dnodt = 0.0
    satrec[].domdt = 0.0
    satrec[].e3 = 0.0
    satrec[].ee2 = 0.0
    satrec[].peo = 0.0
    satrec[].pgho = 0.0
    satrec[].pho = 0.0
    satrec[].pinco = 0.0
    satrec[].plo = 0.0
    satrec[].se2 = 0.0
    satrec[].se3 = 0.0
    satrec[].sgh2 = 0.0
    satrec[].sgh3 = 0.0
    satrec[].sgh4 = 0.0
    satrec[].sh2 = 0.0
    satrec[].sh3 = 0.0
    satrec[].si2 = 0.0
    satrec[].si3 = 0.0
    satrec[].sl2 = 0.0
    satrec[].sl3 = 0.0
    satrec[].sl4 = 0.0
    satrec[].gsto = 0.0
    satrec[].xfact = 0.0
    satrec[].xgh2 = 0.0
    satrec[].xgh3 = 0.0
    satrec[].xgh4 = 0.0
    satrec[].xh2 = 0.0
    satrec[].xh3 = 0.0
    satrec[].xi2 = 0.0
    satrec[].xi3 = 0.0
    satrec[].xl2 = 0.0
    satrec[].xl3 = 0.0
    satrec[].xl4 = 0.0
    satrec[].xlamo = 0.0
    satrec[].zmol = 0.0
    satrec[].zmos = 0.0
    satrec[].atime = 0.0
    satrec[].xli = 0.0
    satrec[].xni = 0.0

    #  ------------------------ earth constants -----------------------
    #  sgp4fix identify constants and allow alternate values
    #  this is now the only call for the constants
    var grav = getgravconst(whichconst)
    satrec[].tumin = grav.tumin
    satrec[].mu = grav.mu
    satrec[].radiusearthkm = grav.radiusearthkm
    satrec[].xke = grav.xke
    satrec[].j2 = grav.j2
    satrec[].j3 = grav.j3
    satrec[].j4 = grav.j4
    satrec[].j3oj2 = grav.j3oj2

    # ------------------------------------------------------------------------

    # The SGP4 library has changed `satn` from an integer to a string; it is
    # only ever written out, never read back by the mathematics.
    satrec[].error = 0
    satrec[].operationmode = opsmode
    satrec[].satnum = satn

    # --- upstream comment block, kept for navigation ---
    # // sgp4fix - note the following variables are also passed directly via satrec.
    # // it is possible to streamline the sgp4init call by deleting the "x"
    # // variables, but the user would need to set the satrec.* values first. we
    # // include the additional assignments in case twoline2rv is not used.
    satrec[].bstar = xbstar
    # sgp4fix allow additional parameters in the struct
    satrec[].ndot = xndot
    satrec[].nddot = xnddot
    satrec[].ecco = xecco
    satrec[].argpo = xargpo
    satrec[].inclo = xinclo
    satrec[].mo = xmo
    # sgp4fix rename variables to clarify which mean motion is intended
    satrec[].no_kozai = xno_kozai
    satrec[].nodeo = xnodeo

    # single averaged mean elements
    satrec[].am = 0.0
    satrec[].em = 0.0
    satrec[].im = 0.0
    satrec[].Om = 0.0
    satrec[].mm = 0.0
    satrec[].nm = 0.0

    # sgp4fix identify constants and allow alternate values no longer needed
    var ss = 78.0 / satrec[].radiusearthkm + 1.0
    #  sgp4fix use multiply for speed instead of pow
    var qzms2ttemp = (120.0 - 78.0) / satrec[].radiusearthkm
    var qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp

    satrec[].init = INIT_Y
    satrec[].t = 0.0

    # sgp4fix remove satn as it is not needed in initl
    var il = _initl(
        satrec[].xke, satrec[].j2, satrec[].ecco, epoch, satrec[].inclo,
        satrec[].no_kozai, satrec[].method, satrec[].operationmode,
    )
    satrec[].no_unkozai = il.no
    var method = il.method
    var ainv = il.ainv
    var ao = il.ao
    satrec[].con41 = il.con41
    var con42 = il.con42
    var cosio = il.cosio
    var cosio2 = il.cosio2
    var eccsq = il.eccsq
    var omeosq = il.omeosq
    var posq = il.posq
    var rp = il.rp
    var rteosq = il.rteosq
    var sinio = il.sinio
    satrec[].gsto = il.gsto
    satrec[].a = pow(satrec[].no_unkozai * satrec[].tumin, (-2.0 / 3.0))
    satrec[].alta = satrec[].a * (1.0 + satrec[].ecco) - 1.0
    satrec[].altp = satrec[].a * (1.0 - satrec[].ecco) - 1.0

    # --- upstream comment block, kept for navigation ---
    # // sgp4fix remove this check as it is unnecessary
    # // the mrt check in sgp4 handles decaying satellite cases even if the starting
    # // condition is below the surface of te earth
    # //     if (rp < 1.0)
    # //       {
    # //         printf("# *** satn%d epoch elts sub-orbital ***\n", satn);
    # //         satrec.error = 5;
    # //       }

    if omeosq >= 0.0 or satrec[].no_unkozai >= 0.0:

        satrec[].isimp = 0
        if rp < 220.0 / satrec[].radiusearthkm + 1.0:
            satrec[].isimp = 1
        var sfour = ss
        var qzms24 = qzms2t
        var perige = (rp - 1.0) * satrec[].radiusearthkm

        #  - for perigees below 156 km, s and qoms2t are altered -
        if perige < 156.0:

            sfour = perige - 78.0
            if perige < 98.0:
                sfour = 20.0
            #  sgp4fix use multiply for speed instead of pow
            var qzms24temp = (120.0 - sfour) / satrec[].radiusearthkm
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp
            sfour = sfour / satrec[].radiusearthkm + 1.0

        var pinvsq = 1.0 / posq

        var tsi = 1.0 / (ao - sfour)
        satrec[].eta = ao * satrec[].ecco * tsi
        var etasq = satrec[].eta * satrec[].eta
        var eeta = satrec[].ecco * satrec[].eta
        var psisq = abs(1.0 - etasq)
        var coef = qzms24 * pow(tsi, 4.0)
        var coef1 = coef / pow(psisq, 3.5)
        var cc2 = coef1 * satrec[].no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
                       (4.0 + etasq)) + 0.375 * satrec[].j2 * tsi / psisq * satrec[].con41 *
                       (8.0 + 3.0 * etasq * (8.0 + etasq)))
        satrec[].cc1 = satrec[].bstar * cc2
        var cc3 = 0.0
        if satrec[].ecco > 1.0e-4:
            cc3 = -2.0 * coef * tsi * satrec[].j3oj2 * satrec[].no_unkozai * sinio / satrec[].ecco
        satrec[].x1mth2 = 1.0 - cosio2
        satrec[].cc4 = 2.0 * satrec[].no_unkozai * coef1 * ao * omeosq * \
                      (satrec[].eta * (2.0 + 0.5 * etasq) + satrec[].ecco *
                      (0.5 + 2.0 * etasq) - satrec[].j2 * tsi / (ao * psisq) *
                      (-3.0 * satrec[].con41 * (1.0 - 2.0 * eeta + etasq *
                      (1.5 - 0.5 * eeta)) + 0.75 * satrec[].x1mth2 *
                      (2.0 * etasq - eeta * (1.0 + etasq)) * cos(2.0 * satrec[].argpo)))
        satrec[].cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
                       (etasq + eeta) + eeta * etasq)
        var cosio4 = cosio2 * cosio2
        var temp1 = 1.5 * satrec[].j2 * pinvsq * satrec[].no_unkozai
        var temp2 = 0.5 * temp1 * satrec[].j2 * pinvsq
        var temp3 = -0.46875 * satrec[].j4 * pinvsq * pinvsq * satrec[].no_unkozai
        satrec[].mdot = satrec[].no_unkozai + 0.5 * temp1 * rteosq * satrec[].con41 + 0.0625 * \
                        temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
        satrec[].argpdot = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                            (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                            temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4))
        var xhdot1 = -temp1 * cosio
        satrec[].nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                             2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
        var xpidot = satrec[].argpdot + satrec[].nodedot
        satrec[].omgcof = satrec[].bstar * cc3 * cos(satrec[].argpo)
        satrec[].xmcof = 0.0
        if satrec[].ecco > 1.0e-4:
            satrec[].xmcof = -x2o3 * coef * satrec[].bstar / eeta
        satrec[].nodecf = 3.5 * omeosq * xhdot1 * satrec[].cc1
        satrec[].t2cof = 1.5 * satrec[].cc1
        #  sgp4fix for divide by zero with xinco = 180 deg
        if abs(cosio + 1.0) > 1.5e-12:
            satrec[].xlcof = -0.25 * satrec[].j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
        else:
            satrec[].xlcof = -0.25 * satrec[].j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4
        satrec[].aycof = -0.5 * satrec[].j3oj2 * sinio
        #  sgp4fix use multiply for speed instead of pow
        var delmotemp = 1.0 + satrec[].eta * cos(satrec[].mo)
        satrec[].delmo = delmotemp * delmotemp * delmotemp
        satrec[].sinmao = sin(satrec[].mo)
        satrec[].x7thm1 = 7.0 * cosio2 - 1.0

        #  --------------- deep space initialization -------------
        if 2 * pi / satrec[].no_unkozai >= 225.0:

            satrec[].method = METHOD_D
            satrec[].isimp = 1
            var tc = 0.0
            var inclm = satrec[].inclo

            var dsc = _dscom(
                epoch, satrec[].ecco, satrec[].argpo, tc, satrec[].inclo,
                satrec[].nodeo, satrec[].no_unkozai,
            )
            satrec[].e3 = dsc.e3
            satrec[].ee2 = dsc.ee2
            satrec[].peo = dsc.peo
            satrec[].pgho = dsc.pgho
            satrec[].pho = dsc.pho
            satrec[].pinco = dsc.pinco
            satrec[].plo = dsc.plo
            satrec[].se2 = dsc.se2
            satrec[].se3 = dsc.se3
            satrec[].sgh2 = dsc.sgh2
            satrec[].sgh3 = dsc.sgh3
            satrec[].sgh4 = dsc.sgh4
            satrec[].sh2 = dsc.sh2
            satrec[].sh3 = dsc.sh3
            satrec[].si2 = dsc.si2
            satrec[].si3 = dsc.si3
            satrec[].sl2 = dsc.sl2
            satrec[].sl3 = dsc.sl3
            satrec[].sl4 = dsc.sl4
            satrec[].xgh2 = dsc.xgh2
            satrec[].xgh3 = dsc.xgh3
            satrec[].xgh4 = dsc.xgh4
            satrec[].xh2 = dsc.xh2
            satrec[].xh3 = dsc.xh3
            satrec[].xi2 = dsc.xi2
            satrec[].xi3 = dsc.xi3
            satrec[].xl2 = dsc.xl2
            satrec[].xl3 = dsc.xl3
            satrec[].xl4 = dsc.xl4
            satrec[].zmol = dsc.zmol
            satrec[].zmos = dsc.zmos
            var em = dsc.em
            var emsq = dsc.emsq
            var cosim = dsc.cosim
            var sinim = dsc.sinim
            var sinomm = dsc.sinomm
            var cosomm = dsc.cosomm
            var nm = dsc.nm

            var dpp = _dpper(
                satrec, inclm, satrec[].init,
                satrec[].ecco, satrec[].inclo, satrec[].nodeo, satrec[].argpo,
                satrec[].mo, satrec[].operationmode,
            )
            satrec[].ecco = dpp.ep
            satrec[].inclo = dpp.inclp
            satrec[].nodeo = dpp.nodep
            satrec[].argpo = dpp.argpp
            satrec[].mo = dpp.mp

            var argpm = 0.0
            var nodem = 0.0
            var mm = 0.0

            var dsi = _dsinit(
                satrec[].xke,
                cosim, emsq, satrec[].argpo,
                dsc.s1, dsc.s2, dsc.s3, dsc.s4, dsc.s5, sinim,
                dsc.ss1, dsc.ss2, dsc.ss3, dsc.ss4, dsc.ss5,
                dsc.sz1, dsc.sz3, dsc.sz11, dsc.sz13, dsc.sz21, dsc.sz23,
                dsc.sz31, dsc.sz33,
                satrec[].t, tc, satrec[].gsto, satrec[].mo, satrec[].mdot,
                satrec[].no_unkozai, satrec[].nodeo, satrec[].nodedot, xpidot,
                dsc.z1, dsc.z3, dsc.z11, dsc.z13, dsc.z21, dsc.z23, dsc.z31,
                dsc.z33,
                satrec[].ecco, eccsq, em, argpm, inclm, mm, nm, nodem,
            )

            em = dsi.em
            argpm = dsi.argpm
            inclm = dsi.inclm
            mm = dsi.mm
            nm = dsi.nm
            nodem = dsi.nodem
            satrec[].irez = dsi.irez
            satrec[].atime = dsi.atime
            satrec[].d2201 = dsi.d2201
            satrec[].d2211 = dsi.d2211
            satrec[].d3210 = dsi.d3210
            satrec[].d3222 = dsi.d3222
            satrec[].d4410 = dsi.d4410
            satrec[].d4422 = dsi.d4422
            satrec[].d5220 = dsi.d5220
            satrec[].d5232 = dsi.d5232
            satrec[].d5421 = dsi.d5421
            satrec[].d5433 = dsi.d5433
            satrec[].dedt = dsi.dedt
            satrec[].didt = dsi.didt
            satrec[].dmdt = dsi.dmdt
            var dndt = dsi.dndt
            satrec[].dnodt = dsi.dnodt
            satrec[].domdt = dsi.domdt
            satrec[].del1 = dsi.del1
            satrec[].del2 = dsi.del2
            satrec[].del3 = dsi.del3
            satrec[].xfact = dsi.xfact
            satrec[].xlamo = dsi.xlamo
            satrec[].xli = dsi.xli
            satrec[].xni = dsi.xni

        #----------- set variables if not deep space -----------
        if satrec[].isimp != 1:

            var cc1sq = satrec[].cc1 * satrec[].cc1
            satrec[].d2 = 4.0 * ao * tsi * cc1sq
            var temp = satrec[].d2 * tsi * satrec[].cc1 / 3.0
            satrec[].d3 = (17.0 * ao + sfour) * temp
            satrec[].d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * \
                         satrec[].cc1
            satrec[].t3cof = satrec[].d2 + 2.0 * cc1sq
            satrec[].t4cof = 0.25 * (3.0 * satrec[].d3 + satrec[].cc1 *
                         (12.0 * satrec[].d2 + 10.0 * cc1sq))
            satrec[].t5cof = 0.2 * (3.0 * satrec[].d4 +
                         12.0 * satrec[].cc1 * satrec[].d3 +
                         6.0 * satrec[].d2 * satrec[].d2 +
                         15.0 * cc1sq * (2.0 * satrec[].d2 + cc1sq))

    # --- upstream comment block, kept for navigation ---
    # /* finally propogate to zero epoch to initialize all others. */
    # // sgp4fix take out check to let satellites process until they are actually below earth surface
    # //       if(satrec.error == 0)
    _ = sgp4(satrec, 0.0)

    satrec[].init = INIT_N

    # sgp4fix return boolean. satrec.error contains any error codes
    return True

# ----------------------------------------------------------------------------

#  sgp4/functions.py

# ----------------------------------------------------------------------------

struct JdayResult(ImplicitlyCopyable, Copyable, Movable):
    var jd: Float64
    var fr: Float64

    def __init__(out self, jd: Float64, fr: Float64):
        self.jd = jd
        self.fr = fr

struct DivmodResult(ImplicitlyCopyable, Copyable, Movable):
    var quot: Float64
    var rem: Float64

    def __init__(out self, quot: Float64, rem: Float64):
        self.quot = quot
        self.rem = rem


struct MonthDayResult(ImplicitlyCopyable, Copyable, Movable):
    var month: Int
    var day: Int

    def __init__(out self, month: Int, day: Int):
        self.month = month
        self.day = day


struct MdhmsResult(ImplicitlyCopyable, Copyable, Movable):
    var mon: Int
    var day: Int
    var hr: Int
    var minute: Int
    var sec: Float64

    def __init__(out self, mon: Int, day: Int, hr: Int, minute: Int, sec: Float64):
        self.mon = mon
        self.day = day
        self.hr = hr
        self.minute = minute
        self.sec = sec

def _divmod(a: Float64, b: Float64) -> DivmodResult:
    """Python's float `divmod`: the quotient is floored, the remainder is
    whatever is left, and the two satisfy `a == q * b + r`."""
    var q = floor(a / b)
    return DivmodResult(q, a - q * b)

def jday(year: Int, mon: Int, day: Int, hr: Int, minute: Int, sec: Float64) -> JdayResult:
    """Return two floats that, when added, produce the specified Julian date."""
    var jd_ = (367.0 * Float64(year)
           - floor(7.0 * (Float64(year) + floor((Float64(mon) + 9.0) / 12.0)) * 0.25)
           + floor(275.0 * Float64(mon) / 9.0)
           + Float64(day)
           + 1721013.5)
    var fr_ = (sec + Float64(minute) * 60.0 + Float64(hr) * 3600.0) / 86400.0
    return JdayResult(jd_, fr_)

# `sgp4.ext` carries a second, older `jday` that returns one combined float
# rather than the two-float pair that `sgp4.functions.jday` returns.  `io.py`
# imports the combined one, and the difference matters: it is what puts the
# epoch's fractional day into `jdsatepoch`.
def ext_jday(year: Int, mon: Int, day: Int, hr: Int, minute: Int, sec: Float64) -> Float64:
    return (367.0 * Float64(year)
            - floor(7.0 * (Float64(year) + floor((Float64(mon) + 9.0) / 12.0)) * 0.25)
            + floor(275.0 * Float64(mon) / 9.0)
            + Float64(day) + 1721013.5
            + ((sec / 60.0 + Float64(minute)) / 60.0 + Float64(hr)) / 24.0)


def _day_of_year_to_month_day(day_of_year: Int, is_leap: Int) -> MonthDayResult:
    """Core logic for turning days into months, for easy testing."""
    var february_bump = (2 - is_leap) * (1 if day_of_year >= 60 + is_leap else 0)
    var august = 1 if day_of_year >= 215 else 0
    var md = _divmod(Float64(2 * (day_of_year - 1 + 30 * august + february_bump)), 61.0)
    var month = Int(md.quot) + 1 - august
    var day = Int(md.rem) // 2 + 1
    return MonthDayResult(month, day)

def days2mdhms(year: Int, days: Float64, round_to_microsecond: Int = 6) -> MdhmsResult:
    """Convert a floating point number of days into the year into date and time."""
    var second = days * 86400.0
    if round_to_microsecond:
        second = round(second, round_to_microsecond)

    var dm = _divmod(second, 60.0)
    var minutes = dm.quot
    second = dm.rem
    if round_to_microsecond:
        second = round(second, round_to_microsecond)

    var minute_ = Int(floor(minutes))
    var hm = _divmod(Float64(minute_), 60.0)
    var hour_ = Int(hm.quot)
    minute_ = Int(hm.rem)
    var dh = _divmod(Float64(hour_), 24.0)
    var day_of_year = Int(dh.quot)
    hour_ = Int(dh.rem)

    var is_leap = 1 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 0
    var md = _day_of_year_to_month_day(day_of_year, is_leap)
    var month = md.month
    var day = md.day
    if month == 13:  # behave like the original in case of overflow
        month = 12
        day = day + 31

    return MdhmsResult(month, day, hour_, minute_, second)

# ----------------------------------------------------------------------------

#  sgp4/ext.py

# ----------------------------------------------------------------------------

def mag(x: PtrF) -> Float64:
    return sqrt(x.unsafe_load(0) * x.unsafe_load(0) + x.unsafe_load(1) * x.unsafe_load(1) + x.unsafe_load(2) * x.unsafe_load(2))

def vector_cross(vec1: PtrF, vec2: PtrF, outvec: PtrF):
    outvec.unsafe_store(0, vec1.unsafe_load(1) * vec2.unsafe_load(2) - vec1.unsafe_load(2) * vec2.unsafe_load(1))
    outvec.unsafe_store(1, vec1.unsafe_load(2) * vec2.unsafe_load(0) - vec1.unsafe_load(0) * vec2.unsafe_load(2))
    outvec.unsafe_store(2, vec1.unsafe_load(0) * vec2.unsafe_load(1) - vec1.unsafe_load(1) * vec2.unsafe_load(0))

def vector_dot(x: PtrF, y: PtrF) -> Float64:
    return x.unsafe_load(0) * y.unsafe_load(0) + x.unsafe_load(1) * y.unsafe_load(1) + x.unsafe_load(2) * y.unsafe_load(2)

def vector_angle(vec1: PtrF, vec2: PtrF) -> Float64:
    var small = 0.00000001
    var undefined = 999999.1
    var magv1 = mag(vec1)
    var magv2 = mag(vec2)
    if magv1 * magv2 > small * small:
        var temp = vector_dot(vec1, vec2) / (magv1 * magv2)
        if abs(temp) > 1.0:
            temp = 1.0 if temp > 0.0 else -1.0
        return acos(temp)
    else:
        return undefined

struct NewtonnuResult(ImplicitlyCopyable, Copyable, Movable):
    var e0: Float64
    var m: Float64

    def __init__(out self, e0: Float64, m: Float64):
        self.e0 = e0
        self.m = m

def newtonnu(ecc: Float64, nu: Float64) -> NewtonnuResult:
    #  ---------------------  implementation   ---------------------
    var e0 = 999999.9
    var m = 999999.9
    var small = 0.00000001
    #  --------------------------- circular ------------------------
    if abs(ecc) < small:
        m = nu
        e0 = nu
    else:
        #  ---------------------- elliptical -----------------------
        if ecc < 1.0 - small:
            var sine = (sqrt(1.0 - ecc * ecc) * sin(nu)) / (1.0 + ecc * cos(nu))
            var cose = (ecc + cos(nu)) / (1.0 + ecc * cos(nu))
            e0 = atan2(sine, cose)
            m = e0 - ecc * sin(e0)
        else:
            #  -------------------- hyperbolic  --------------------
            if ecc > 1.0 + small:
                if ecc > 1.0 and abs(nu) + 0.00001 < pi - acos(1.0 / ecc):
                    var sine = (sqrt(ecc * ecc - 1.0) * sin(nu)) / (1.0 + ecc * cos(nu))
                    e0 = asinh(sine)
                    m = ecc * sinh(e0) - e0
            else:
                #  ----------------- parabolic ---------------------
                if abs(nu) < 168.0 * pi / 180.0:
                    e0 = tan(nu * 0.5)
                    m = e0 + (e0 * e0 * e0) / 3.0
    if ecc < 1.0:
        m = fmod(m, 2.0 * pi)
        if m < 0.0:
            m = m + 2.0 * pi
        e0 = fmod(e0, 2.0 * pi)
    return NewtonnuResult(e0, m)

struct InvjdayResult(ImplicitlyCopyable, Copyable, Movable):
    var year: Int
    var mon: Int
    var day: Int
    var hr: Int
    var minute: Int
    var sec: Float64

    def __init__(out self, year: Int, mon: Int, day: Int, hr: Int, minute: Int, sec: Float64):
        self.year = year
        self.mon = mon
        self.day = day
        self.hr = hr
        self.minute = minute
        self.sec = sec

def invjday(jd_: Float64) -> InvjdayResult:
    #  --------------- find year and days of the year ---------------
    var temp = jd_ - 2415019.5
    var tu = temp / 365.25
    var year = 1900 + Int(floor(tu))
    var leapyrs = Int(floor((Float64(year - 1901) * 0.25)))

    #  optional nudge by 8.64x10-7 sec to get even outputs
    var days = temp - (Float64(year - 1900) * 365.0 + Float64(leapyrs)) + 0.00000000001

    #  ------------ check for case of beginning of a year -----------
    if days < 1.0:
        year = year - 1
        leapyrs = Int(floor((Float64(year - 1901) * 0.25)))
        days = temp - (Float64(year - 1900) * 365.0 + Float64(leapyrs))

    #  ----------------- find remaing data  -------------------------
    var r = days2mdhms(year, days, 0)
    var sec = r.sec - 0.00000086400
    return InvjdayResult(year, r.mon, r.day, r.hr, r.minute, sec)

# ----------------------------------------------------------------------------

#  sgp4/model.py -- SatrecArray.sgp4

#

#  Upstream's pure-Python SatrecArray walks the satellites on the outside and

#  the times on the inside.  The loop order is observable: the deep space

#  integrator carries `atime`, `xli` and `xni` from one call to the next, so a

#  satellite's times must stay consecutive in the same order.  Each satellite

#  advances its own state, which is why one flat block of `Satrec` structs and

#  a fixed stride is all the memory this needs.

# ----------------------------------------------------------------------------

def sgp4_array(
    satrecs: Int,
    stride: Int,
    n_sat: Int,
    jd: PtrF,
    fr: PtrF,
    n: Int,
    e: PtrU8,
    r: PtrF,
    v: PtrF,
    errval: PtrF,
):
    var s = 0
    while s < n_sat:
        var satrec = PtrSat(unsafe_from_address=satrecs + s * stride)
        var i = 0
        while i < n:
            var tsince = ((jd.unsafe_load(i) - satrec[].jdsatepoch) * minutes_per_day +
                          (fr.unsafe_load(i) - satrec[].jdsatepochF) * minutes_per_day)
            var out = sgp4(satrec, tsince)
            e.unsafe_store(s * n + i, UInt8(satrec[].error))
            errval.unsafe_store(s, out.errval)
            var base = (s * n + i) * 3
            r.unsafe_store(base, out.r.x)
            r.unsafe_store(base + 1, out.r.y)
            r.unsafe_store(base + 2, out.r.z)
            v.unsafe_store(base, out.v.x)
            v.unsafe_store(base + 1, out.v.y)
            v.unsafe_store(base + 2, out.v.z)
            i += 1
        s += 1
