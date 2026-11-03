"""Exact cases with known answers, at a small truncation so they run in milliseconds"""

import numpy as np
import pytest

from stratodiff.data import spectral

J = 21
LON = np.linspace(0, 359, 37)


def packed(entries: dict[tuple[int, int], complex]) -> np.ndarray:
    """GRIB-ordered coefficients with the given (n, m) entries set and the rest zero."""
    x = []
    for m in range(J + 1):
        for n in range(m, J + 1):
            c = entries.get((n, m), 0)
            x += [np.real(c), np.imag(c)]
    return np.array(x)


def test_unpack_refuses_the_wrong_size():
    with pytest.raises(ValueError):
        spectral.unpack(np.zeros(10), J)


def test_constant():
    c = spectral.unpack(packed({(0, 0): 210.0}), J)
    assert np.allclose(spectral.scalar(c, 37.0, LON), 210.0)


@pytest.mark.parametrize("lat", [-80.0, -12.5, 0.0, 45.0, 84.0])
def test_first_harmonics(lat):
    """P(1,0) = sqrt(3) sin(lat) and P(1,1) = sqrt(3/2) cos(lat) in ECMWF's normalisation."""
    c = spectral.unpack(packed({(1, 0): 2.0, (1, 1): 0.5 - 0.25j}), J)
    phi, lam = np.deg2rad(lat), np.deg2rad(LON)
    expect = 2.0 * np.sqrt(3) * np.sin(phi) + 2 * np.sqrt(1.5) * np.cos(phi) * (
        0.5 * np.cos(lam) + 0.25 * np.sin(lam))
    assert np.allclose(spectral.scalar(c, lat, LON), expect)


@pytest.mark.parametrize("lat", [-60.0, 0.0, 30.0, 75.0])
def test_solid_body_rotation(lat):
    """u = U cos(lat) has vorticity 2 U sin(lat) / a, i.e. only coefficient (1, 0) = 2 U / (a sqrt 3)."""
    U, a = 20.0, spectral.EARTH_RADIUS_M
    vo = spectral.unpack(packed({(1, 0): 2 * U / (a * np.sqrt(3))}), J)
    u, v = spectral.wind(vo, np.zeros_like(vo), lat, LON)
    assert np.allclose(u, U * np.cos(np.deg2rad(lat)))
    assert np.allclose(v, 0, atol=1e-12)
