"""mojo-sgp4 against the upstream `sgp4` package, on the same inputs.

    pixi run bench

Upstream ships two back ends: a C++ extension (`sgp4.api`, which is what an
ordinary `import sgp4` gives you) and a pure-Python one (`sgp4.model`).  Both
are timed here, because the honest comparison is against the one a caller
would actually get, and because the pure-Python one is the source this port
was written from.

Every number printed is measured; nothing is estimated.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from pkgutil import get_data

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojosgp4.api as ms  # noqa: E402
import sgp4.api as up_fast  # noqa: E402
import sgp4.model as up_slow  # noqa: E402

assert up_fast.accelerated, "upstream C++ extension not available"


def timeit(fn, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def catalogue():
    """Every element set in the official SGP4 verification file."""
    lines = iter(get_data('sgp4', 'SGP4-VER.TLE').decode('ascii').splitlines())
    out = []
    for line1 in lines:
        if not line1.startswith('1'):
            continue
        line2 = next(lines)
        out.append((line1, line2))
    return out


TLEs = catalogue()
# A group of element sets that share an epoch, so a shared set of dates means
# something to all of them.  Exactly one element set in SGP4-VER.TLE has epoch
# day `00`, so a pool of N is that one repeated N times: the catalogues below
# are N independent Satrec objects, which is what the per-element cost and the
# stride arithmetic in the batch kernel turn on.
SAME_EPOCH = [t for t in TLEs if t[0][18:20] == '00'][:1]


def pool(n):
    return (SAME_EPOCH * n)[:n]


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("twoline2rv, 33 element sets (ms)")
def _():
    ours = lambda: [ms.Satrec.twoline2rv(a, b) for a, b in TLEs]
    fast = lambda: [up_fast.Satrec.twoline2rv(a, b) for a, b in TLEs]
    slow = lambda: [up_slow.Satrec.twoline2rv(a, b) for a, b in TLEs]
    return ours, (fast, slow)


@case("sgp4_tsince, near-Earth (1e5 calls)")
def _():
    sat = ms.Satrec.twoline2rv(*TLEs[0])
    ref = up_fast.Satrec.twoline2rv(*TLEs[0])
    ref_slow = up_slow.Satrec.twoline2rv(*TLEs[0])
    ts = [i * 0.1 for i in range(100_000)]

    def ours():
        f = sat.sgp4_tsince
        for t in ts:
            f(t)

    def theirs():
        f = ref.sgp4_tsince
        for t in ts:
            f(t)

    def theirs_slow():
        f = ref_slow.sgp4_tsince
        for t in ts:
            f(t)
    return ours, (theirs, theirs_slow)


@case("sgp4_tsince, deep space (1e5 calls)")
def _():
    sat = ms.Satrec.twoline2rv(*TLEs[4])
    ref = up_fast.Satrec.twoline2rv(*TLEs[4])
    ref_slow = up_slow.Satrec.twoline2rv(*TLEs[4])
    ts = [i * 0.1 for i in range(100_000)]

    def ours():
        f = sat.sgp4_tsince
        for t in ts:
            f(t)

    def theirs():
        f = ref.sgp4_tsince
        for t in ts:
            f(t)

    def theirs_slow():
        f = ref_slow.sgp4_tsince
        for t in ts:
            f(t)
    return ours, (theirs, theirs_slow)


def _jdfr(n):
    times = np.arange(float(n))
    return (2453911.5 + np.floor(times / 1440.0),
            times / 1440.0 - np.floor(times / 1440.0))


@case("Satrec.sgp4_array, 64 satellites x 1440 times")
def _():
    jd, fr = _jdfr(1440)
    sats = [ms.Satrec.twoline2rv(*t) for t in pool(64)]
    refs = [up_fast.Satrec.twoline2rv(*t) for t in pool(64)]

    def ours():
        for s in sats:
            s.sgp4_array(jd, fr)

    def theirs():
        for s in refs:
            s.sgp4_array(jd, fr)
    return ours, (theirs, None)


@case("SatrecArray.sgp4, 64 satellites x 1440 times")
def _():
    jd, fr = _jdfr(1440)
    sats = [ms.Satrec.twoline2rv(*t) for t in pool(64)]
    refs_fast = [up_fast.Satrec.twoline2rv(*t) for t in pool(64)]
    refs_slow = [up_slow.Satrec.twoline2rv(*t) for t in pool(64)]
    ours = lambda: ms.SatrecArray(sats).sgp4(jd, fr)
    fast = lambda: up_fast.SatrecArray(refs_fast).sgp4(jd, fr)
    slow = lambda: up_slow.SatrecArray(refs_slow).sgp4(jd, fr)
    return ours, (fast, slow)


@case("SatrecArray.sgp4, 512 satellites x 2880 times")
def _():
    jd, fr = _jdfr(2880)
    sats = [ms.Satrec.twoline2rv(*t) for t in pool(512)]
    refs_fast = [up_fast.Satrec.twoline2rv(*t) for t in pool(512)]
    ours = lambda: ms.SatrecArray(sats).sgp4(jd, fr)
    fast = lambda: up_fast.SatrecArray(refs_fast).sgp4(jd, fr)
    return ours, (fast, None)


@case("SatrecArray.sgp4, 8 satellites x 10080 times")
def _():
    jd, fr = _jdfr(10080)
    sats = [ms.Satrec.twoline2rv(*t) for t in pool(8)]
    refs_fast = [up_fast.Satrec.twoline2rv(*t) for t in pool(8)]
    ours = lambda: ms.SatrecArray(sats).sgp4(jd, fr)
    fast = lambda: up_fast.SatrecArray(refs_fast).sgp4(jd, fr)
    return ours, (fast, None)


@case("jday + days2mdhms + gstime (1e5 each)")
def _():
    from mojosgp4 import functions as mf, propagation as mp
    from sgp4 import functions as uf, propagation as up
    args = [(2000 + i % 20, 1 + i % 12, 1 + i % 28, i % 24, i % 60, i * 0.01)
            for i in range(100_000)]
    jds = [2451545.0 + i * 0.5 for i in range(100_000)]

    def ours():
        for a in args:
            mf.jday(*a)
            mf.days2mdhms(a[0], a[5] + 3.25)
        for j in jds:
            mp.gstime(j)

    def theirs():
        for a in args:
            uf.jday(*a)
            uf.days2mdhms(a[0], a[5] + 3.25)
        for j in jds:
            up.gstime(j)
    return ours, (theirs, None)


@case("SatrecArray.sgp4, 128 satellites x 1440 times, pure Python")
def _():
    jd, fr = _jdfr(1440)
    ours = [ms.Satrec.twoline2rv(*t) for t in pool(128)]
    slow = [up_slow.Satrec.twoline2rv(*t) for t in pool(128)]
    return (lambda: ms.SatrecArray(ours).sgp4(jd, fr),
            (None, lambda: up_slow.SatrecArray(slow).sgp4(jd, fr)))


def main() -> None:
    machine = os.uname()
    print('mojo-sgp4 benchmark')
    print('machine: %s %s, %d cores' % (machine.sysname, machine.machine,
                                        os.cpu_count()))
    print('upstream sgp4: C++ extension %s' %
          ('present' if up_fast.accelerated else 'absent'))
    print()
    print('| case | mojo-sgp4 | upstream (C++) | upstream (pure Python) '
          '| vs C++ | vs Python |')
    print('| --- | ---: | ---: | ---: | ---: | ---: |')
    for name, build in CASES:
        ours_fn, (fast_fn, slow_fn) = build()
        ours_fn()
        a = timeit(ours_fn)
        b = timeit(fast_fn) if fast_fn is not None else math.nan
        c = timeit(slow_fn, repeat=1) if slow_fn is not None else math.nan
        row = '| %s | %.2f ms | ' % (name, a * 1e3)
        row += ('%.2f ms | ' % (b * 1e3)) if fast_fn is not None else 'n/a | '
        row += ('%.2f ms | ' % (c * 1e3)) if slow_fn is not None else 'n/a | '
        row += ('%.2fx | ' % (b / a)) if fast_fn is not None else 'n/a | '
        row += ('%.2fx |' % (c / a)) if slow_fn is not None else 'n/a |'
        print(row)


if __name__ == '__main__':
    main()
