"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PKG))
SRC = os.path.join(ROOT, "src")

LIB_NAME = "libmojo-sgp4.so"


def lib_candidates() -> list[str]:
    """Where to look for the compiled kernel, best first.

    An installed wheel has no `src/`, so it carries the shared library beside
    the package; a checkout keeps it in `dist/`.  `MOJOSGP4_LIB` overrides
    both, which is how a caller points at a build of their own.
    """
    override = os.environ.get("MOJOSGP4_LIB")
    if override:
        return [override]

    return [os.path.join(PKG, LIB_NAME), os.path.join(ROOT, "dist", LIB_NAME)]


I = ctypes.c_int64
F = ctypes.c_double
V = ctypes.c_void_p

# The `Satrec` struct in src/ported.mojo, in declaration order.  Every field is
# eight bytes wide, so field `i` lives at byte offset `8 * i` and the struct is
# `8 * len(SATREC_FIELDS)` bytes; `ms_satrec_field_count()` reports the count
# from the compiled side and tests/test_parity.py checks that the two agree and
# that every field matches upstream.
SATREC_FIELDS = (
    'Om', 'a', 'alta', 'altp', 'am', 'argpdot', 'argpo', 'atime', 'aycof',
    'bstar', 'cc1', 'cc4', 'cc5', 'con41', 'd2', 'd2201', 'd2211', 'd3',
    'd3210', 'd3222', 'd4', 'd4410', 'd4422', 'd5220', 'd5232', 'd5421',
    'd5433', 'dedt', 'del1', 'del2', 'del3', 'delmo', 'didt', 'dmdt',
    'dnodt', 'domdt', 'e3', 'ecco', 'ee2', 'em', 'epochdays', 'epochyr',
    'error', 'eta', 'gsto', 'im', 'inclo', 'init', 'irez', 'isimp', 'j2',
    'j3', 'j3oj2', 'j4', 'jdsatepoch', 'jdsatepochF', 'mdot', 'method',
    'mm', 'mo', 'mu', 'nddot', 'ndot', 'nm', 'no_kozai', 'no_unkozai',
    'nodecf', 'nodedot', 'nodeo', 'om', 'omgcof', 'operationmode', 'peo',
    'pgho', 'pho', 'pinco', 'plo', 'radiusearthkm', 'satnum', 'se2', 'se3',
    'sgh2', 'sgh3', 'sgh4', 'sh2', 'sh3', 'si2', 'si3', 'sinmao', 'sl2',
    'sl3', 'sl4', 't', 't2cof', 't3cof', 't4cof', 't5cof', 'tumin',
    'x1mth2', 'x7thm1', 'xfact', 'xgh2', 'xgh3', 'xgh4', 'xh2', 'xh3',
    'xi2', 'xi3', 'xke', 'xl2', 'xl3', 'xl4', 'xlamo', 'xlcof', 'xli',
    'xmcof', 'xni', 'zmol', 'zmos',
)

# Fields the Mojo struct holds as `Int` rather than `Float64`.
SATREC_INT_FIELDS = frozenset({
    'epochyr', 'error', 'init', 'irez', 'isimp', 'method', 'operationmode',
    'satnum',
})

# Upstream stores these as one-character strings; the Mojo struct keeps them as
# small integers, and `model.Satrec` converts in both directions.
# `sgp4.model` numbers the gravity models; upstream indexes a tuple with them.
WGS72OLD, WGS72, WGS84 = 0, 1, 2

METHOD_N, METHOD_D = 0, 1
INIT_Y, INIT_N = 0, 1
OPSMODE_A, OPSMODE_I = 0, 1

_CHAR_FIELDS = {
    'method': {METHOD_N: 'n', METHOD_D: 'd'},
    'init': {INIT_Y: 'y', INIT_N: 'n'},
    'operationmode': {OPSMODE_A: 'a', OPSMODE_I: 'i'},
}
_CHAR_VALUES = {name: {v: k for k, v in table.items()}
                for name, table in _CHAR_FIELDS.items()}

# name -> (argtypes, restype)
_SIGNATURES = {
    "ms_fmod": ([F, F], F),
    "ms_gstime": ([F], F),
    "ms_getgravconst": ([I, V], I),
    "ms_jday": ([I, I, I, I, I, F, V], None),
    "ms_days2mdhms": ([I, F, I, V], None),
    "ms_ext_jday": ([I, I, I, I, I, F], F),
    "ms_day_of_year_to_month_day": ([I, I, V], None),
    "ms_invjday": ([F, V], None),
    "ms_mag": ([V], F),
    "ms_dot": ([V, V], F),
    "ms_cross": ([V, V, V], None),
    "ms_angle": ([V, V], F),
    "ms_newtonnu": ([F, F, V], None),
    "ms_sgp4init": ([I, I, I] + [F] * 10 + [V], I),
    "ms_sgp4": ([V, F, V, V, V], I),
    "ms_sgp4_array": ([V, I, I, V, V, I, V, V, V, V], None),
    "ms_satrec_field_count": ([], I),
}

NFIELDS = len(SATREC_FIELDS)
STRIDE = NFIELDS * 8


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOSGP4_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOSGP4_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Return a usable path to the compiled kernel, compiling it if needed.

    An existing library is used as is when there are no Mojo sources to
    compare it against, which is the case for an installed wheel.  Set
    `MOJOSGP4_LIB` to point at a specific build.
    """
    candidates = lib_candidates()
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force:
        for path in candidates:
            if not os.path.exists(path):
                continue
            if not sources or os.path.getmtime(path) >= max(
                    os.path.getmtime(s) for s in sources):
                return path
    if not sources:
        raise BuildError(
            "no Mojo sources at %s and no compiled library at %s; run "
            "`pixi run build`, or set MOJOSGP4_LIB"
            % (SRC, ' or '.join(candidates)))
    out = candidates[0] if os.environ.get("MOJOSGP4_LIB") \
        else candidates[-1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "--fp-mode", "contract=off",
        "-I", SRC, os.path.join(SRC, "capi.mojo"), "-o", out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(out):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return out


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
        n = int(_lib.ms_satrec_field_count())
        if n != NFIELDS:
            raise BuildError(
                "src/ported.mojo struct Satrec has %d fields but "
                "python/mojosgp4/_lib.py lists %d" % (n, NFIELDS))
    return _lib



def main() -> int:
    """`python -m mojosgp4._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
