"""ERA5 for one span of hours into the nside 256 store (milestone M1, SPEC.md).

Per hour and level: read vorticity, divergence and temperature coefficients from ARCO's spectral store,
evaluate u, v and T at the HEALPix pixel centres with data/spectral.py, regrid Z bilinearly from ARCO's
gridded geopotential, check against ARCO's grid, normalise, and write float16 with a channel mask."""

from ..config import ClusterConfig, RunConfig


def ingest(config: RunConfig, cluster: ClusterConfig) -> None:
    raise NotImplementedError("M1: ingest one month to the cluster's data_root (SPEC.md)")
