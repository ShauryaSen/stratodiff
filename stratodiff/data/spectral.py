"""ERA5's spherical-harmonic fields evaluated exactly at any latitude and longitude.

ARCO's `co/model-level-wind.zarr-v2` stores vorticity, divergence and temperature as T639 coefficients:
410,240 floats per field, (real, imaginary) pairs ordered by m, then n from m to 639. The Legendre functions
are ECMWF's, normalised so that half the integral of their square over [-1, 1] is 1.

Checked against ARCO's own 0.25 degree grid for 2020-01-15 00Z at levels 50 and 65 (DECISIONS D4):
u and v RMSE 0.03-0.06 m/s, T 0.02 K. Reproduce with tools/check_arco_spectral.py."""

import numpy as np

EARTH_RADIUS_M = 6_371_229.0  # the IFS radius


def unpack(x: np.ndarray, truncation: int = 639) -> np.ndarray:
    """Packed GRIB coefficients to a complex array c[n, m], zero above the triangle."""
    J = truncation
    c = np.zeros((J + 1, J + 1), complex)
    i = 0
    for m in range(J + 1):
        k = J + 1 - m
        c[m:, m] = x[i:i + 2 * k:2] + 1j * x[i + 1:i + 2 * k:2]
        i += 2 * k
    if i != x.size:
        raise ValueError(f"{x.size} values do not fill a T{J} triangle ({i} expected)")
    return c


def _epsilon(J: int) -> np.ndarray:
    """eps[n, m] = sqrt((n^2 - m^2) / (4 n^2 - 1)), the recursion's coefficient, up to n = J + 2."""
    n = np.arange(J + 3)[:, None].astype(float)
    m = np.arange(J + 2)[None, :].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.sqrt(np.clip(n * n - m * m, 0, None) / (4 * n * n - 1))
    e[0] = 0
    return e


def legendre(mu: float, truncation: int = 639) -> tuple[np.ndarray, np.ndarray]:
    """P[n, m] for n, m <= J, and H[n, m] = (1 - mu^2) dP/dmu, at one sin(latitude).
    Values that underflow near the poles become zero, which is what they are to double precision."""
    J = truncation
    eps = _epsilon(J)
    P = np.zeros((J + 2, J + 2))
    P[0, 0] = 1.0
    cosine = np.sqrt(1 - mu * mu)
    for m in range(1, J + 2):
        P[m, m] = np.sqrt((2 * m + 1) / (2.0 * m)) * cosine * P[m - 1, m - 1]
    for n in range(1, J + 2):
        ms = np.arange(n)
        before = P[n - 2, ms] if n >= 2 else 0.0
        P[n, ms] = (mu * P[n - 1, ms] - eps[n - 1, ms] * before) / eps[n, ms]
    H = np.zeros((J + 1, J + 1))
    for n in range(J + 1):
        ms = np.arange(n + 1)
        below = P[n - 1, ms] if n > 0 else 0.0
        H[n, ms] = -n * eps[n + 1, ms] * P[n + 1, ms] + (n + 1) * eps[n, ms] * below
    return P[:J + 1, :J + 1], H


def _ring(G: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Real field on a ring of longitudes (radians) from its Fourier coefficients G[m]."""
    m = np.arange(G.size)
    terms = G[None, :] * np.exp(1j * np.outer(lon, m))
    return np.real(terms[:, 0]) + 2 * np.real(terms[:, 1:].sum(1))


def scalar(c: np.ndarray, lat_deg: float, lon_deg: np.ndarray) -> np.ndarray:
    """A scalar field (e.g. temperature) at one latitude and any longitudes, from c = unpack(...)."""
    P, _ = legendre(np.sin(np.deg2rad(lat_deg)), c.shape[0] - 1)
    return _ring((c * P).sum(0), np.deg2rad(lon_deg))


def wind(vorticity: np.ndarray, divergence: np.ndarray, lat_deg: float, lon_deg: np.ndarray,
         radius: float = EARTH_RADIUS_M) -> tuple[np.ndarray, np.ndarray]:
    """Eastward and northward wind (m/s) at one latitude, from unpacked vorticity and divergence.
    Inverts them to streamfunction and velocity potential, then differentiates exactly."""
    J = vorticity.shape[0] - 1
    n = np.arange(J + 1, dtype=float)
    scale = np.zeros(J + 1)
    scale[1:] = -radius**2 / (n[1:] * (n[1:] + 1))
    psi, chi = scale[:, None] * vorticity, scale[:, None] * divergence
    mu = np.sin(np.deg2rad(lat_deg))
    P, H = legendre(mu, J)
    im = 1j * np.arange(J + 1)
    cosine = np.sqrt(1 - mu * mu)
    lon = np.deg2rad(lon_deg)
    u = _ring((-(psi * H).sum(0) + im * (chi * P).sum(0)) / (radius * cosine), lon)
    v = _ring((im * (psi * P).sum(0) + (chi * H).sum(0)) / (radius * cosine), lon)
    return u, v
