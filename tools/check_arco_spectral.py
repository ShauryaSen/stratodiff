"""Synthesise u, v, T from ARCO's spectral store and compare them with ARCO's own 0.25 degree grid.

    uv run python tools/check_arco_spectral.py --cache /tmp/arco --date 2020-01-15T00 --levels 50 65

Downloads about 5 MB of coefficients and 3 x ~75 MB gridded chunks per level into --cache, once. This is the
M1 pass check in miniature (SPEC.md); the ingest will run it over every level and hour."""

import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path

import numcodecs
import numpy as np

from stratodiff.data import spectral

BUCKET = "https://storage.googleapis.com/gcp-public-data-arco-era5/"
SPECTRAL = "co/model-level-wind.zarr-v2/"
GRIDDED = "ar/model-level-1h-0p25deg.zarr-v1/"


def fetch(cache: Path, key: str) -> bytes:
    path = cache / key.replace("/", "_")
    if not path.exists():
        urllib.request.urlretrieve(BUCKET + key, path)
    return path.read_bytes()


def array(cache: Path, store: str, var: str, chunk: str) -> np.ndarray:
    meta = json.loads(fetch(cache, store + ".zmetadata"))["metadata"][var + "/.zarray"]
    raw = fetch(cache, f"{store}{var}/{chunk}")
    if meta["compressor"]:
        raw = numcodecs.get_codec(meta["compressor"]).decode(raw)
    return np.frombuffer(raw, dtype=meta["dtype"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--date", default="2020-01-15T00")
    p.add_argument("--levels", type=int, nargs="+", default=[50, 65])
    p.add_argument("--lat-step", type=int, default=12, help="every Nth 0.25 degree latitude, poles excluded")
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    hour = int((datetime.fromisoformat(a.date) - datetime(1900, 1, 1)).total_seconds() // 3600)
    lat = array(a.cache, GRIDDED, "latitude", "0")
    rows = np.arange(24, 697, a.lat_step)  # 84N to 84S
    lon = np.arange(1440) * 0.25
    for level in a.levels:
        k = level - 1
        vo, d, t = (spectral.unpack(array(a.cache, SPECTRAL, v, f"{hour}.{k}.0")) for v in ("vo", "d", "t"))
        ours = {"t": [], "u": [], "v": []}
        for j in rows:
            ours["t"].append(spectral.scalar(t, float(lat[j]), lon))
            u, v = spectral.wind(vo, d, float(lat[j]), lon)
            ours["u"].append(u)
            ours["v"].append(v)
        weights = np.cos(np.deg2rad(lat[rows]))[:, None] * np.ones((1, 1440))
        print(f"{a.date} level {level}")
        for ours_name, arco_name in (("t", "temperature"), ("u", "u_component_of_wind"), ("v", "v_component_of_wind")):
            arco = array(a.cache, GRIDDED, arco_name, f"{hour}.{k // 18}.0.0").reshape(-1, 721, 1440)[k % 18][rows]
            diff = np.array(ours[ours_name]) - arco
            rmse = np.sqrt(np.average(diff**2, weights=weights))
            print(f"  {ours_name}: rmse {rmse:.4f}, max {np.abs(diff).max():.3f}")


if __name__ == "__main__":
    main()
