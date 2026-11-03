# Roadmap: from scaffold to a research-grade generator

Tick a box when its check passes, not when the code is written. Milestone numbers match `SPEC.md`.
**Decide** items are open choices, each with a recommendation; record the answer in `DECISIONS.md`.
Numbers marked *estimate* are arithmetic from grid sizes, not measurements.

The phases run in order, except that 1 can run alongside 2 and 3, and the storage request in 0 should go
out now because Phase 6 waits on it.

---

## Phase 0. Set up (this week)

- [x] Scaffold pushed: spec, decisions, working rules, config, run log, cluster jobs, spectral reader (M0)
- [ ] Repo housekeeping
  - [ ] Rename the default branch `scaffold` to `main`
  - [ ] Decide: public or private until v1. *Recommend private until Sarah has seen it.*
  - [ ] GitHub Actions running `uv run pytest -q` on every push
- [ ] Unicorn ready
  - [ ] Clone to `~/stratodiff` on Unicorn; `uv sync --extra data` on a `dean-compute-02` job
  - [ ] `df -h /scratch` on `dean-compute-02`; make `/scratch/sps252/stratodiff`
  - [ ] Print the ingest job: `uv run stratodiff submit --cluster unicorn ingest configs/pilot_month.yaml`; check partition and account are accepted
- [ ] Measure ARCO download speed from `dean-compute-02` (one hour of coefficients plus Z chunks, timed)
  - The whole archive is about 81 TB of coefficients plus the Z grids (*estimate*), so this number sets how long full ingest takes
- [ ] Storage request out: email Sarah and CUGeoData asking for about 110 TB of shared storage, readable from every cluster, with the arithmetic

**Done when:** tests pass in CI, a printed Unicorn job looks right, and the download speed is written down.

---

## Phase 1. Ingest one month (M1)

- [ ] **Grid:** HEALPix nside 256 nested pixel centres (latitude per ring, longitudes per ring)
  - Decide: `healpy` or `earth2grid` for pixel geometry. *Recommend earth2grid: the U-Net needs it anyway.*
- [ ] **Fast synthesis.** The reader is exact but slow in plain numpy.
  - [ ] Precompute the Legendre tables once for the 1,023 ring latitudes (about 3.4 GB each for P and H at float64, *estimate*), save to scratch
  - [ ] Synthesis per hour becomes matrix products: about 4×10¹⁰ multiply-adds per hour for 40 levels (*estimate*)
  - [ ] Time one hour on CPU and on one GPU
  - [ ] Decide: CPU or GPU ingest, from the timing. *Recommend GPU if CPU takes over 30 s per hour: 412,000 hours in total.*
  - [ ] Test: fast path equals `spectral.wind` and `spectral.scalar` on random rings
- [ ] **Z:** read ARCO gridded geopotential for levels 29-68, regrid bilinearly to the pixel centres
- [ ] **Normalisation:** per-channel mean and standard deviation from training years only; stored beside the data, versioned
- [ ] **Store:** zarr on scratch, one chunk per hour, float16, with a channel mask; a manifest of hours written
  - [ ] Ingest is resumable: a restarted job skips hours already in the manifest
- [ ] **Checks** (the M1 pass conditions), written to a report file per run
  - [ ] u, v, T against ARCO's grid at ARCO points: RMSE under 0.1 m/s and 0.05 K on every level and hour
  - [ ] Z against Z rebuilt from T (hypsometric): set the tolerance on day 1, then hold it
  - [ ] Float16 changes no field by more than 1% of its standard deviation
- [ ] **Run it:** January 2020, 744 hours, about 190 GB at float16 (*estimate*)

**Done when:** the pilot month is on scratch and every check in the report passes.

---

## Phase 2. Coarse data for all years (M2)

- [ ] Stage 1 data straight from the coefficients: synthesise at nside 256, block-mean to nside 64, keep only the coarse fields
  - This needs no large storage: all 47 years at nside 64, 6-hourly, are about 2.2 TB (*estimate*)
- [ ] Run 1979-2025 as year-sized resumable jobs
- [ ] **Checks**
  - [ ] Every 6-hourly step present and finite
  - [ ] Monthly mean and variance per channel match the nside 256 pilot month where they overlap
- [ ] QBO index: tropical (10S-10N) mean zonal wind at the top level over the days before each block, computed from the store itself (September's definition, `balloon-env-dev/.../hpx/dataset.py`)
  - Decide: lookback length and level. *Recommend keeping September's 30 days, and checking it tracks the published QBO index.*

**Done when:** the full coarse store passes its checks and the QBO index plotted over 1979-2025 shows the ~28-month cycle.

---

## Phase 3. Stage 1 code

Port from `balloon-env-dev/src/eval/windeval/hpx/` (`net.py`, `train_coarse.py`, `dataset.py`, `layout.py`) rather than rewriting; it passed its gates in September.

- [ ] **Data loader:** random 12 x 6 h windows from all training years mixed; calendar and local solar time at 8 frequencies; QBO index; channel mask
- [ ] **Network:** HEALPix U-Net with earth2grid padding; learned position map; attention at the coarsest levels
- [ ] **Diffusion:** EDM preconditioning and loss; log-uniform training noise; 18-step Heun sampler
- [ ] **Masked frames:** some training blocks get a few real frames as conditioning, some none
- [ ] **Training loop:** EMA, checkpoints, resume after preemption, multi-GPU (DDP over the 4 Adas), run log
- [ ] **Tests**
  - [ ] Shapes and padding across HEALPix face edges
  - [ ] Overfits one batch to near-zero loss
  - [ ] Same seed gives the same first 10 steps

**Done when:** a 1,000-step run on one GPU trains, resumes after a kill, and records its config, commit and seed.

---

## Phase 4. Stage 1 sweep and model of record (M3, M4)

- [ ] **Write first:** hypothesis and expected result for the sweep, in the run's notes
- [ ] **Noise maximum:** compute cBottle's rule (A.3.1, the noise must swamp the leading pattern of variation) from the coarse store, then sweep around it
- [ ] **Small-scale sweep:** a few years of data, 3 training seeds per setting
  - Scientific: network size and noise maximum. Nuisance: learning rate. Fixed: everything else, stated.
  - [ ] Report validation loss by noise level, mean and 95% interval over seeds
- [ ] **Model of record:** chosen settings, all training years, 1 seed (3 when compute allows)
- [ ] **Write thresholds first, then check on 2019-2020** (M4)
  - [ ] Seasonal cycle follows day of year
  - [ ] QBO phase follows the QBO index
  - [ ] Zonal-mean wind and spectra inside ERA5's year-to-year range
  - [ ] Calm levels (under 5 m/s) as common as in ERA5 (the cBottle failure)

**Done when:** the Stage 1 model of record passes the M4 checks, with seeds and intervals reported.

---

## Phase 5. Stage 2 on the years scratch can hold (M5)

- [ ] Ingest full-resolution years to scratch: 4 TB holds about 1.8 years at float16 (*estimate*)
  - Decide: which years. *Recommend 2016-2017 and the 2019 validation year: recent, and inside the training split.*
- [ ] **Code:** port the September fine model; residual patches (64 px, 32 px padding), 13 hourly frames, Stage 1 frames lifted as conditioning, exact block-mean projection
- [ ] **Hypothesis written, then train**
- [ ] **Checks**
  - [ ] Block means equal Stage 1 to float tolerance
  - [ ] Spectra down to 25 km inside ERA5's range
  - [ ] No visible patch seams (seam statistic below a threshold set first)

**Done when:** a Stage 2 model passes its checks on the validation year.

---

## Phase 6. Time tiling (M6)

- [ ] InfiniteDiffusion in time: overlapping blocks denoised together, noise seeded by a hash of (time, place, seed)
- [ ] **Tests**
  - [ ] Same seed and time give identical output in any query order
  - [ ] Asking for day 300 does not require generating days 1-299
- [ ] **Checks:** no jump at block boundaries; temporal spectrum and autocorrelation inside ERA5's range up to one block
- [ ] Measure cost: GPU-seconds per simulated day, and cold-query time

**Done when:** a 90-day run is reproducible to the bit and passes the seam and spectrum checks.

---

## Phase 7. Export and interface (M7)

- [ ] Query: (time span, region, seed) to a record with u, v, T, Z and pressure and height per level
- [ ] Export to the balloon-sim record format (`u, v, hours, altitude_km, latitude, longitude`)
- [ ] **Check:** balloon-sim loads an exported record and flies a controller through it

**Done when:** a generated record flies in balloon-sim end to end.

---

## Phase 8. Full scale (M8, needs the shared storage)

- [ ] Ingest full resolution for all training years onto shared storage
- [ ] Retrain Stage 2 on all years with mixed batches
- [ ] Model of record with 3 training seeds for both stages; repeat the M4 and M5 checks with intervals
- [ ] Cluster profile for CUGeoData; run one job there to prove portability

**Done when:** both stages are trained on all training years, with 3 seeds and every check passing.

---

## Phase 9. Research-grade release (v1.0)

- [ ] **Reproducible:** one command rebuilds each check's figure from a checkpoint and config
- [ ] **Data:** manifest of every ingested hour with source and checksums; normalisation stats versioned
- [ ] **Model card** updated with measured results only (the shared doc and `README.md`)
- [ ] **Weights** published with their config and commit (decide where: Hugging Face or Zenodo)
- [ ] `CITATION.cff`, a tagged `v1.0` release, and an install-and-sample example that runs on one GPU
- [ ] Sarah reviews before release

**Done when:** someone outside the lab can install it, generate a balloon-sim record, and reproduce the checks.

---

## After v1 (research, separate from this roadmap)

- Tiling vs autoregression in time (the hypothesis on the model card), on the masked-frame weights
- The benchmark: cBottle, ERA5 interpolation, the September model, radiosondes, Loon
- The fidelity-transfer experiment and its variable shortlist
- Vertical wind w, a synthetic QBO generator, guidance, a distilled sampler, finer-than-ERA5 sources
