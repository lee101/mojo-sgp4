# mojo-sgp4

The SGP4 orbit propagator from the Python [`sgp4`](https://pypi.org/project/sgp4/)
package, rewritten in Mojo, with a Python API that mirrors upstream's names and
signatures.

`sgp4` is a straight transcription of David Vallado's `sgp4unit.cpp`; this repo
transcribes that transcription into Mojo, function by function, in upstream's
order, and calls it through `ctypes`. Read `src/ported.mojo` next to
`sgp4/propagation.py` of python-sgp4 2.x and they line up.

```python
import numpy as np
from mojosgp4.api import Satrec, WGS72, SatrecArray

line1 = '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
line2 = '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'

sat = Satrec.twoline2rv(line1, line2, WGS72)
error, position_km, velocity_km_s = sat.sgp4(2451723.28495062, 0.0)
print(error, position_km)
# 0 (7022.4652985858875, -1400.0829516642518, 0.03996296233479315)

# or a whole catalogue at one set of dates, which is where the kernel earns
# its keep: one crossing for every satellite and every time
tles = [(line1, line2)]
sats = [Satrec.twoline2rv(a, b, WGS72) for a, b in tles]
jds = np.array([2451723.28495062, 2451723.53495062])
frs = np.array([0.0, 0.25])
e, r, v = SatrecArray(sats).sgp4(jds, frs)   # shapes (1, 2), (1, 2, 3), (1, 2, 3)
print(e.tolist(), r[0, 0])
# [[0, 0]] [ 7.02246530e+03 -1.40008295e+03  3.99629623e-02]
```

That snippet is `tests/test_readme_example.py`, so it cannot rot.

## Install

From a checkout, with the Mojo toolchain that `pixi.toml` pins:

```
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles `src/capi.mojo` into `dist/libmojo-sgp4.so` and
copies it into the package, so the wheel carries it and an installed
`mojosgp4` imports without a toolchain. In a checkout the package also rebuilds
the library on demand if it is missing or older than `src/`, so
`import mojosgp4` works after `pixi install` alone. `MOJOSGP4_LIB` points it at
a specific build, and `MOJOSGP4_MOJO` at a specific compiler.

## What is covered

Ported, in upstream's source order, in `src/ported.mojo`:

| upstream | what it does |
| --- | --- |
| `sgp4.propagation.gstime` | greenwich sidereal time from a Julian date |
| `sgp4.propagation.getgravconst` | the WGS72OLD / WGS72 / WGS84 constant sets |
| `sgp4.propagation._dpper` | deep space long-period periodics |
| `sgp4.propagation._dscom` | deep space common items, solar and lunar |
| `sgp4.propagation._dsinit` | geopotential resonance terms |
| `sgp4.propagation._dspace` | third-body and resonance mean-element drift |
| `sgp4.propagation._initl` | un-Kozai-ing and epoch quantities |
| `sgp4.propagation.sgp4` | the propagator itself, near-earth and deep space |
| `sgp4.propagation.sgp4init` | the initialiser |
| `sgp4.functions.jday`, `days2mdhms`, `_day_of_year_to_month_day` | date handling |
| `sgp4.ext.jday`, `invjday` | the older combined-float `jday`, and its inverse |
| `sgp4.ext.mag`, `dot`, `cross`, `angle`, `newtonnu` | vector and Kepler helpers |
| `sgp4.model.Satrec`, `SatrecArray`, `Satellite` | the satellite objects |
| `sgp4.io.twoline2rv`, `verify_checksum`, `fix_checksum`, `compute_checksum` | TLE parsing |
| `sgp4.alpha5.to_alpha5`, `from_alpha5` | satellite number encoding |
| `sgp4.earth_gravity` | the three constant sets |
| `sgp4.conveniences` | `jday_datetime`, `sat_epoch_datetime`, `check_satrec`, `dump_satrec` |
| `sgp4.api` | `SGP4_ERRORS`, `Satrec`, `SatrecArray`, `WGS72OLD`/`WGS72`/`WGS84`, `jday`, `days2mdhms` |

The whole of `SGP4-VER.TLE` — all 33 official verification element sets, both
near-earth and deep space, synchronous and half-day resonance, and the ones
that decay — is checked against upstream in `tests/test_parity.py`.

### What is not covered

* **`sgp4.ext.rv2coe`.** State-vector to classical-elements conversion. It is
  not part of propagation, it is not compute-bound, and it is not on any hot
  path; it was left out rather than half-ported. Same for `sgp4.omm`,
  `sgp4.exporter` (CCSDS OMM and TLE import/export), and `sgp4.wulfgar`.
* **`sgp4.wrapper` and the C++ `SatrecArray`.** Upstream's accelerated classes
  are a C extension; here the Mojo kernel is always what runs, so
  `mojosgp4.api.accelerated` is `True` unconditionally and there is one
  implementation rather than two.
* **`sgp4.tests`.** Upstream's own test harness, which drives `tcppver.out`.
  This repo has its own suite instead, described under Tests below.
* **The eight derived columns of the `tcppver.out` long lines** (temperature,
  eccentricity, inclination, RAAN, argument of perigee, mean anomaly, mean
  motion, revolution number). Those come from the verification *formatter*, not
  from the propagator, and they are not produced here.

## Known divergences

Each of these is a property of the Mojo dialect or its standard library, not a
choice about the algorithm. `src/ported.mojo` says the same at each site.

1. **Values that upstream returns as tuples come back as structs.**
   `_dpper` returns 5 values, `_dscom` returns 76, `_dsinit` returns 31. Mojo
   functions have one return value, so each is a struct with upstream's field
   names in upstream's order. `sgp4` returns upstream's `(r, v)` pair as
   `Sgp4Out`, which also carries the out-of-range number that upstream formats
   into `satrec.error_message`; `python/mojosgp4/propagation.py` builds the
   sentence, so the wording is upstream's.

2. **`method`, `init` and `operationmode` are integers in the struct.** Upstream
   compares them against `'n'`, `'d'`, `'y'` and `'a'`. The struct has to stay a
   packed row of 8-byte fields for the Python side to read it as a NumPy block,
   and a `String` field is neither 8 bytes nor trivially copyable. The named
   constants `METHOD_N`, `INIT_Y`, `OPSMODE_A` and friends are at the top of the
   file, and `mojosgp4.model.Satrec` converts in both directions, so
   `satrec.method` still reads `'d'`.

3. **Parameters are immutable and cannot be shadowed.** Upstream's `_dpper`,
   `_dsinit`, `_dspace` and `_initl` take several arguments as in/out and
   rebind them. Here those parameters carry a trailing underscore and the body
   opens with `var name = name_` under a comment saying so. Argument order,
   count and meaning are unchanged.

4. **Loop-carried locals are declared above their loop.** Upstream's `for lsflg
   in 1, 2` in `_dscom` and the `if irez != 0` blocks in `_dsinit`/`_dspace`
   leave values behind that are read after the block; Mojo needs them declared
   first. The loops and branches are upstream's.

5. **`gstime` and `getgravconst` are hoisted above their callers, and `sgp4`
   sits before `sgp4init`.** Mojo resolves names in file order and the upstream
   call graph is not a DAG: `sgp4init` ends by propagating to `t = 0`, and
   `_initl` calls `gstime`.

6. **Satellite numbers are integers at the ABI.** Upstream 2.x changed `satn`
   from an integer to an Alpha-5 string. `sgp4init` takes the integer and the
   Python layer does the `to_alpha5`/`from_alpha5` round trip, so
   `satrec.satnum_str` and `satrec.satnum` behave as upstream's do.

7. **Arithmetic precision.** `build.sh` passes `--fp-mode contract=off`, because
   Mojo otherwise fuses `a * b - c * d` into an FMA and rounds once where
   CPython rounds twice. With it off, every helper — `jday`, `days2mdhms`,
   `invjday`, `gstime`, `mag`, `dot`, `cross`, `angle`, `newtonnu`, the gravity
   constants — reproduces upstream to the last bit or to one ulp.

   `std.math.pow` is the exception. Mojo's is accurate to about 4e-9 relative
   for a fractional exponent where CPython's is correctly rounded:

   | exponent | max relative error over 2e5 random bases |
   | --- | --- |
   | `2/3` | 7.9e-10 |
   | `-2/3` | 7.9e-10 |
   | `3/2` | 1.8e-9 |
   | `7/2` | 4.1e-9 |
   | `4` (integer) | 4.3e-16 |

   Measured by exporting `std.math.pow` from a throwaway Mojo shared library
   and comparing against CPython's over 200,000 random bases per exponent,
   bases uniform in [1e-6, 1e3], seed 7.

   SGP4 takes both the un-Kozai'd mean motion and the drag coefficients through
   `pow`, so this is the one place the port and upstream genuinely part company.
   Measured over all 33 verification element sets:

   * every field of `Satrec` agrees to **2.9e-8** relative or better;
   * positions at epoch (`t = 0`) agree to **1.2e-6 km**, which is 1.2 mm;
   * positions over each satellite's whole verification span agree to
     **3.1 km** worst case. That worst case is one satellite; the other 32 sit
     at 1.9 km or better, and all but four are under 0.3 km.

   The 3.1 km is satellite 29141, at the minute where the propagator decides
   the satellite has hit the atmosphere: `mrt` is 0.996252 against a limit of
   1.0. Near that boundary the trajectory amplifies its input exponentially, so
   a 1e-12 relative seed is metres at one step and kilometres later. Upstream's
   own C++ and pure-Python back ends agree to about 1e-11 km here because they
   share the same `pow`. `tests/test_parity.py` asserts bounds above these
   maxima, so a regression past them fails the suite.

## Benchmarks

`pixi run bench` prints this table. Upstream is timed twice: its C++ extension,
which is what `import sgp4` normally gives you, and its pure-Python
`sgp4.model`, which is the source this port was written from.

Machine: Linux x86_64, 36 cores (72 threads), Intel Xeon E5-2697 v4 @ 2.30 GHz,
Ubuntu 24.04, Python 3.13.14, `mojo ==1.2.0.dev2026092605`, `sgp4 ==2.26`
(C++ extension present). Timings are the best of five, and the `pixi run bench`
task holds a machine-wide `flock` so concurrent jobs cannot distort them. This
is one measured run pasted verbatim; repeat runs move the small cases by a few
percent, and the ratios move with them, so treat the conclusions below rather
than the digits as the result.

The satellite pools are real: each row builds the number of independent
`Satrec` objects its name says. They are all the same element set, because
exactly one of the 33 in `SGP4-VER.TLE` shares an epoch day with the dates
used, and the point of these rows is the per-element cost and the stride
arithmetic, not catalogue diversity.

| case | mojo-sgp4 | upstream (C++) | upstream (pure Python) | vs C++ | vs Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| twoline2rv, 33 element sets (ms) | 3.98 ms | 0.25 ms | 3.45 ms | 0.06x | 0.87x |
| sgp4_tsince, near-Earth (1e5 calls) | 334.97 ms | 105.59 ms | 651.80 ms | 0.32x | 1.95x |
| sgp4_tsince, deep space (1e5 calls) | 594.63 ms | 167.08 ms | 3092.22 ms | 0.28x | 5.20x |
| Satrec.sgp4_array, 64 satellites x 1440 times | 50.86 ms | 65.14 ms | n/a | 1.28x | n/a |
| SatrecArray.sgp4, 64 satellites x 1440 times | 52.56 ms | 67.75 ms | 2281.06 ms | 1.29x | 43.40x |
| SatrecArray.sgp4, 512 satellites x 2880 times | 822.77 ms | 1072.18 ms | n/a | 1.30x | n/a |
| SatrecArray.sgp4, 8 satellites x 10080 times | 44.07 ms | 58.15 ms | n/a | 1.32x | n/a |
| jday + days2mdhms + gstime (1e5 each) | 504.00 ms | 323.27 ms | n/a | 0.64x | n/a |
| SatrecArray.sgp4, 128 satellites x 1440 times, pure Python | 109.54 ms | n/a | 4861.05 ms | n/a | 44.38x |

Read it honestly:

* **The batch kernels win**, by 1.28 to 1.32x against upstream's C++
  `SatrecArray` and by 43 to 44x against upstream's pure-Python one. This is
  the part of SGP4 that is actually compute-bound, and it is the part this
  port exists for.
* **Single propagations still lose**, by about 3x against the C++ extension.
  An SGP4 step on one satellite is a microsecond or two of arithmetic and a
  crossing through `ctypes` is most of the rest; the gap is roughly the size
  of the crossing, which is the floor for any `ctypes` port. Against
  pure-Python upstream the near-Earth case is 2.0x faster and the deep-space
  case 5.2x faster, because there the arithmetic really does dominate the
  crossing.
* **The scalar helpers lose**, 0.64x, for the same reason. `jday` and `gstime`
  are a few dozen floating-point operations each.
* **`twoline2rv` loses**, 0.06x against the C++ extension, and 0.87x against
  the pure-Python one. TLE parsing is string slicing, unchanged from upstream;
  the time is `sgp4init` crossing the FFI once per element set.



## What was optimised, and what was not

The kernels in `src/ported.mojo` are a straight transcription and were not
rewritten for speed. What pays is the Python layer around them, because the
per-call marshalling is comparable to the arithmetic: on this machine
`ndarray.ctypes.data` costs **1.42 microseconds** to evaluate and
`ms_sgp4` costs about 2, so five such lookups per propagation were more than
half the call.

Kept:

| change | why |
| --- | --- |
| scratch addresses taken once, at import | `propagation`, `functions`, `ext` and `earth_gravity` each held NumPy scratch and re-derived `.ctypes.data` on every call |
| the scratch is no longer zeroed | every exported kernel writes every slot it returns, so zeroing first was pure cost; `tests/test_scalar_paths.py` calls each helper twice with different arguments so a kernel that ever stopped writing one would fail the suite |
| results read with `ndarray.tolist()` | 0.16 us against 1.56 us for `tuple(float(v) for v in arr)` |
| `Satrec` caches its own block address | as `_addr`, refreshed when `_buf` is assigned |
| `satrec.error` is not written back from Python | `ms_sgp4` returns the value the kernel has already stored in the struct |
| `Satrec.sgp4` / `sgp4_tsince` call the crossing directly | they went through `propagation.sgp4` and then read `self.error` back through `__getattr__` |
| one dict lookup per numeric attribute | `__getattr__`/`__setattr__` walked the name tables first |

### Tried, measured, reverted

**SIMD over the `tsince` prologue.** `sgp4_array`'s inner loop computes
`(jd[i] - jdsatepoch) * 1440 + (fr[i] - jdsatepochF) * 1440` per element, which
is four operations of pure elementwise arithmetic and the only vectorisable
thing in the loop. Vectorised at `simd_width_of[DType.float64]()` (4 lanes of
`float64`, four `sgp4` calls unrolled, scalar tail for `n % 4`), it measured
two to seven percent *slower* than the scalar loop, consistently, on four
shapes. The four `sgp4` calls that consume the lanes are long and
branch-heavy, so the vector has to stay live across all of them, and the four
scalar operations it replaces are not where the time goes. Reverted;
`src/ported.mojo` has the scalar loop.

The reason SIMD has nothing to grip here is structural and worth stating,
because it is the same reason the batch kernel is the only part that wins:
`sgp4()` is a sequential recurrence. Kepler's equation is an iteration whose
trip count depends on the previous step, the deep-space integrator carries
`atime`, `xli` and `xni` from one call to the next, and every step takes an
error branch. It does not vectorise, and neither does anything that calls it.

**`max.algorithm.parallelize` over the satellite loop.** The outer loop of
`sgp4_array` is the one that genuinely divides: different satellites share
nothing, and only a satellite's own times have to stay consecutive and in
order. It compiles on this toolchain and it runs from a `ctypes` host. It is
not shippable: with the deep-space integrator in the worker body the process
dies with SIGSEGV inside a Mojo worker thread, faulting in
`libKGENCompilerRTShared.so` with no frame from this port at all, and where it
does not fault it is far slower than serial.

So the port ships serial. That is a statement about this toolchain's
`parallelize`, not about the algorithm: the loop really is parallel, and it
would be the right thing to switch on in a runtime where the launch is sound.

**GPU.** `max.gpu.host.DeviceContext`, `ctx.enqueue_create_buffer`,
`ctx.enqueue_function` and `ctx.synchronize` all resolve and type-check on
this toolchain, so the host API is available and was not skipped for want of
it. There is no GPU path here because of what the kernel is, not because of
what the toolchain offers:

1. `sgp4()` is a sequential recurrence per satellite — the reasons above. A
   grid over one satellite's timeline cannot express it. The only parallelism
   is one thread per satellite running a serial inner loop, which is the shape
   that loses to a microsecond-scale CPU call by orders of magnitude, and the
   API this port serves is dominated by exactly that: one propagation per
   call.
2. Parity. The tolerances `tests/test_parity.py` asserts come from scalar
   arithmetic that is bit-identical to CPython's, and the divergences section
   above records that a 1e-12 relative difference in `pow` alone moves a
   decaying satellite by kilometres. GPU `sin`, `cos`, `atan2` and `pow` are
   not bit-identical to the host's, so a GPU path would break bounds the suite
   asserts rather than merely lose a benchmark.

For the same reason the ABI was not widened so a caller could ask for many
steps in one crossing. It would help the single-propagation case, and the
single-propagation case is already within a small factor of the floor a
`ctypes` port can reach.


## How it works

**FFI.** `src/ported.mojo` holds the kernels in upstream's order. `src/capi.mojo`
is the only file with `@export`, and its whole job is to bridge the two: every
buffer crosses as an `Int` address and is rebuilt as a
`Pointer[Float64, AnyOrigin[mut=True]]` inside the wrapper, because `@export`
refuses parametric functions and a pointer with an inferred origin is
parametric. Nothing in the kernel allocates, so nothing can leak; Python owns
every byte, including scratch. `@export` requires an explicit `abi("C")` and an
`abi("C")` function may not be `raises`, so all the error signalling is an
`Int` return plus a code written into the satellite.

Because Python owns the memory, the Python side is what keeps the boundary
honest, and the two places that could go wrong are checked:

* `Satrec` refuses a block that is not exactly `NFIELDS` contiguous, 8-byte
  aligned `float64`. Anything else would be read by the kernel as a different
  layout, or past the end.
* `sgp4_array` refuses `jd` and `fr` of different lengths, and refuses
  anything but 1-D. The kernel walks both with a single index, so a shorter
  `fr` would otherwise be read past its end.

`Satrec` also keeps its `int64` view in step with the block it aliases, so
reassigning `_buf` cannot leave the `Int` fields writing into the buffer that
was replaced.

The Python side of the boundary is where this port spends most of its time on
a single propagation, and it is written to spend as little as possible there.
The module-level scratch in `propagation`, `functions`, `ext` and
`earth_gravity` keeps its address across calls rather than re-deriving it, and
is not zeroed before each crossing, because every exported kernel writes every
slot it returns. Results come back through `ndarray.tolist()`, which is one
pass over the buffer in C, rather than an element-at-a-time generator.

That scratch is shared module state, so the helpers are not reentrant within a
process, which is the same bargain upstream's C extension makes.

**Error bookkeeping.** `ms_sgp4` and `ms_sgp4_array` both return the code the
kernel stored in the satellite, and both also hand back the out-of-range value
that produced it, so the Python layer can build upstream's `error_message`
sentence. `SatrecArray.sgp4` propagates in a concatenated block, so it copies
each row back into the caller's `Satrec`: upstream's `SatrecArray` mutates its
satellites, and the deep-space integrator state has to survive the call for a
later one to continue rather than restart.

**Memory layout.** The `Satrec` struct is a flat, packed row of 8-byte fields,
declared in the order of upstream's `Satrec.__slots__`. Python allocates a
`float64` NumPy array of 119 elements, keeps a `float64` and an `int64` view of
the same bytes, and exposes attribute reads and writes as direct indexing — no
copying, no serialisation, and the propagator writes back into the same array
the caller holds. `ms_satrec_field_count()` is the one place the two sides
could drift, so `tests/test_parity.py` checks the count against Python's own
field list and then checks all 119 fields against upstream for all 33 element
sets.

That layout is also what makes `SatrecArray` cheap: a catalogue is one
contiguous run of those rows, and `ms_sgp4_array` takes a base address and a
stride. It walks satellites on the outside and times on the inside, which is the
order upstream's pure-Python `SatrecArray.sgp4` uses and the order the deep
space integrator needs, because it carries `atime`, `xli` and `xni` from one
call to the next.

**Layout.**

```
src/ported.mojo        the kernels, in upstream's source order
src/capi.mojo          the @export C ABI, and the only place the two sides meet
build/build.sh         mojo build --emit shared-lib -> dist/libmojo-sgp4.so
python/mojosgp4/       ctypes wrapper, mirroring upstream's module layout
tests/test_parity.py   parity against the real sgp4 package
tests/test_scalar_paths.py  the scalar entry points: types, scratch reuse, errors
bench/bench.py         benchmarks against upstream, prints a markdown table
```

## Tests

`pixi run test` runs 226 checks against the real `sgp4` package, installed from
conda-forge as a test dependency. `tests/test_parity.py` asserts numerical
parity, not merely that the code runs: every helper against its upstream twin,
every one of the 119 `satrec` fields against upstream's, the propagated
position and velocity over all 33 official `SGP4-VER.TLE` element sets and
their full verification spans, the sequence of error codes the official suite
expects, `sgp4_array` and `SatrecArray` against upstream's, the legacy 1.x
`Satellite` object, and the TLE parser's error paths.

`tests/test_scalar_paths.py` covers the single-propagation surface and the
marshalling around it: the types `sgp4_tsince` and `propagation.sgp4` hand
back, the error code and `error_message` bookkeeping on both the scalar and
the batch path including the clearing of a stale message, that the cached block
address and the `int64` view both follow a reassigned `_buf`, that a block the
kernel could not read is rejected, that mismatched `jd` and `fr` lengths are
rejected rather than read past the end, that every time in a batch call is
covered including counts that are not a multiple of any vector width, and —
for every helper that writes into shared scratch — that a second call with
different arguments cannot see the first call's values.

## License

MIT, matching upstream `python-sgp4`. The SGP4 algorithm is in the public
domain; see Vallado, Crawford, Hujsak and Kelso, *Revisiting Spacetrack Report
#3*, AIAA 2006-6753.
