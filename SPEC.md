# stratodiff: specification

Status 2026-11-04: specified, scaffolded, nothing trained. No result in this file is a measurement of
stratodiff unless it says MEASURED.

## Goal

An open-source, research-grade generator of global stratospheric wind for balloon navigation: random,
reproducible weather on 40 native ERA5 levels (13.7-31.3 km), about 25 km and hourly, queryable at any time
and place. v1 is the generator and a strong data pipeline. Controller transfer and the benchmark are
separate research (out of scope below).

## Core claim (to be tested, not assumed)

Released generators do not cover the balloon band: cBottle has only 200 and 50 hPa near it, and in our
October test its calmer level was under 5 m/s at 19% of points against 70% for ERA5, so no controller could
hold position (MEASURED, `balloon-sim/docs/cbottle-2026-10-21.md`). stratodiff generates the band at ERA5's
native vertical resolution, so that gap closes. Whether it does is checked at M4 and in the deferred benchmark.

## Design (details and basis in DECISIONS.md)

- **Data:** ERA5 1979-2025. Train 1979-2018, validation 2019-2020, evaluation 2021-2025, whole years.
  Wind and temperature from the native T639 spectral coefficients (ARCO `co/model-level-wind.zarr-v2`),
  evaluated exactly at HEALPix pixel centres; Z from ARCO's gridded geopotential, regridded bilinearly.
  Variables u, v, T, Z on L137 levels 29-68: 160 channels.
- **Model:** two-stage cascaded EDM diffusion on a HEALPix U-Net.
  - Stage 1: global, nside 64 (~100 km), 6-hourly, blocks of 12 frames, masked-frame training.
  - Stage 2: residual patches at nside 256 (~25 km), hourly, 13-frame blocks, block means equal Stage 1.
  - Conditioning: day of year and local solar time (8 frequencies each), learned position map, historical
    QBO index. No SST.
  - Unbounded time: InfiniteDiffusion tiling, noise seeded by (time, place).
  - Sampler: 18-step Heun. Network size and the noise maximum set by a small-scale sweep.
- **Storage, at most 1 TB at once:** no full copy of the archive is kept. Two rotating buffers of random
  stretches drawn from all training years, on `dean-compute-02:/scratch/sps252/stratodiff` (node-local,
  not backed up, rebuildable from ARCO): about 0.4 TB of coarse frames and 0.4 TB of fine frames, plus
  0.2 TB headroom. Each downloaded hour feeds both. More storage later means bigger buffers, not new code.
- **Interface:** gridded records with u, v, T, Z and both pressure and height per level; exports the
  balloon-sim record format.

## Milestones (each ends in a check that passes or fails)

| # | Milestone | Pass condition |
|---|---|---|
| M0 | Scaffold | `uv run pytest -q` passes; `stratodiff show-config` resolves every config; a run directory records config, commit and seed |
| M1 | Ingest one month (2020-01) to scratch | u, v, T synthesised from coefficients match ARCO's gridded fields at ARCO grid points: RMSE under 0.1 m/s for u and v and under 0.05 K for T on every level and hour (one hour measured 0.03-0.06 m/s and 0.02 K, see DECISIONS D4). Z agrees with Z rebuilt from T to a tolerance set on the first day and recorded. Float16 storage changes no field by more than 1% of its standard deviation (ASSUMED threshold) |
| M2 | Rotating buffers and the QBO series | Both buffers stay under budget while refilling; every buffer mixes years from across 1979-2018; a restarted refill resumes; the QBO index over 1979-2025 shows the ~28-month cycle |
| M3 | Stage 1 code | A short run trains from the coarse buffer, resumes after a kill, and records config, commit and seed |
| M4 | Stage 1 checks | On 2019-2020: seasonal cycle and QBO phase follow their conditioning; zonal-mean wind and spectra within ERA5's year-to-year range (thresholds written before the run) |
| M5 | Stage 2 from the fine buffer | Block means equal Stage 1 to float tolerance; spectra to 25 km within ERA5's range |
| M6 | Time tiling | Same seed and time give identical output in any query order; no detectable jump at block seams |
| M7 | Export | balloon-sim loads an exported record and flies it; pressure and height per level present |
| MVP | The whole system at demo scale | M1-M7 pass with small models and 1 seed; a generated record flies in balloon-sim; shown to Sarah |
| M8 | Scale up | As storage and GPUs grow: bigger buffers, wider networks, longer training, the M3 sweep with 3 seeds per setting, a model of record with 3 seeds; M4-M5 checks repeated with intervals |

Research milestone after M4: tiling vs autoregression in time, on the same masked-frame weights
(hypothesis and predictions on the model card).

## Compute

Unicorn `dean-compute-02` (4x RTX 6000 Ada, 48 GB), one GPU at a time for now: the node is about 96% busy
(MEASURED, 2026-11-03), and lab members use around two GPUs continuously. CUGeoData later. One command each:
`uv run stratodiff submit --cluster unicorn train configs/stage1.yaml` and the same with `eval`. A cluster
is a YAML profile; no code changes to move clusters.

## Out of scope for v1

- The benchmark against cBottle, ERA5 interpolation, the September model and radiosonde or Loon data.
- Controller transfer, and choosing which realism properties predict it.
- Vertical wind w (left out, not rejected: same ERA5 store, cheap to add).
- A synthetic QBO generator, guidance, channel infilling (the batch format keeps a channel mask for it).
- Finer-than-ERA5 sources (ICON, WOOF) and the variability ERA5 lacks.
- cBottle's expert checkpoints and a distilled few-step sampler.
