"""The usage example in README.md, run verbatim so it cannot rot."""

import numpy as np
import pytest

from mojosgp4.api import Satrec, SatrecArray, WGS72

line1 = '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753'
line2 = '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667'


def test_readme_example():
    sat = Satrec.twoline2rv(line1, line2, WGS72)
    error, position_km, velocity_km_s = sat.sgp4(2451723.28495062, 0.0)
    assert error == 0
    assert position_km == pytest.approx(
        (7022.4652985858875, -1400.0829516642518, 0.03996296233479315), rel=1e-9)
    assert len(velocity_km_s) == 3

    tles = [(line1, line2)]
    sats = [Satrec.twoline2rv(a, b, WGS72) for a, b in tles]
    jds = np.array([2451723.28495062, 2451723.53495062])
    frs = np.array([0.0, 0.25])
    e, r, v = SatrecArray(sats).sgp4(jds, frs)
    assert e.shape == (1, 2)
    assert r.shape == v.shape == (1, 2, 3)
    assert e.tolist() == [[0, 0]]
    assert r[0, 0] == pytest.approx(
        (7022.4652985858875, -1400.0829516642518, 0.03996296233479315), rel=1e-9)
