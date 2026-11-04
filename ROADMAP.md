# Roadmap: from scaffold to a research-grade generator

Build the whole system first at demo scale (the MVP), then make it better by giving it more: bigger
buffers, wider networks, longer training, more seeds. Scaling is a config change, not new code (D12).

Tick a box when its check passes, not when the code is written. Milestone numbers match `SPEC.md`.
**Decide** items are open choices, each with a recommendation; record the answer in `DECISIONS.md`.
Numbers marked *estimate* are arithmetic, not measurements.

**Budget:** at most 1 TB on disk at once (D10): about 0.4 TB coarse buffer, 0.4 TB fine buffer, 0.2 TB
headroom (D11). One GPU at a time on `dean-compute-02` for now.

---

## Phase 0. Set up

- [x] Scaffold pushed: spec, decisions, working rules, config, run log, cluster jobs, spectral reader (M0)
- [ ] Repo housekeeping
  - [x] Rename the default branch `scaffold` to `main`
  - [ ] Decide: public or private until v1. *Recommend private until Sarah has seen it.*
  - [x] GitHub Actions running `uv run pytest -q` on every push
- [x] Unicorn ready (2026-11-03)
  - [x] Clone to `~/stratodiff` on Unicorn; `uv sync --extra data` on a `dean-compute-02` job; 29 tests pass there
  - [x] Scratch on `dean-compute-02`: 3.0 TB free of 6.8 TB (MEASURED); `/scratch/sps252/stratodiff` made
  - [x] Ingest and train jobs accepted by `sbatch --test-only` on `dean`/`dean`. A GPU job on that partition preempts other labs' jobs
- [x] ARCO download speed from `dean-compute-02` (MEASURED once, 2020-01-15T00): coefficients for one hour 188 MB in 6.3 s (8 parallel); Z grids 104 MB in 1.2 s. About 7.5 s and 290 MB per hour
- [x] Storage plan: 1 TB cap with rotating buffers, no storage request (D10-D12, 2026-11-04)
- [ ] Clear September's 712 GB from `/scratch/sps252` (`era5_hpx`, `era5_hpx_heldout_2023`, `runs`; final checkpoints are kept in `~/data/models`). Shaurya runs the delete

---

## Phase 1. Ingest one month (M1)

- [ ] **Grid:** HEALPix nside 256 nested pixel centres (latitude per ring, longitudes per ring)
  - Decide: `healpy` or `earth2grid` for pixel geometry. *Recommend earth2grid: the U-Net needs it anyway.*
- [ ] **Fast synthesis.** The reader is exact but slow in plain numpy.
  - [ ] Precompute the Legendre tables once for the 1,023 ring latitudes (about 3.4 GB each for P and H at float64, *estimate*; float32 halves it)
  - [ ] Synthesis per hour becomes matrix products: about 4×10¹⁰ multiply-adds per hour for 40 levels (*estimate*)
  - [ ] Time one hour on CPU and on one GPU
  - [ ] Decide: CPU or GPU synthesis, from the timing. *Recommend GPU only if CPU is slower than the 7.5 s download, since download then stops being the limit.*
  - [ ] Test: fast path equals `spectral.wind` and `spectral.scalar` on random rings
- [ ] **Z:** read ARCO gridded geopotential for levels 29-68, regrid bilinearly to the pixel centres
- [ ] **Normalisation:** per-channel mean and standard deviation, versioned. From the pilot month at first; recomputed from a random sample of training years in Phase 2
- [ ] **Frame format:** one hour = 160 channels at nside 256, float16, with a channel mask (about 252 MB, *estimate*); its coarse version is the block mean to nside 64 (about 16 MB)
- [ ] **Checks** (the M1 pass conditions), written to a report file per run
  - [ ] u, v, T against ARCO's grid at ARCO points: RMSE under 0.1 m/s and 0.05 K on every level and hour
  - [ ] Z against Z rebuilt from T (hypsometric): set the tolerance on day 1, then hold it
  - [ ] Float16 changes no field by more than 1% of its standard deviation
  - [ ] Measure lossless compression (e.g. zstd) on the frames: if it helps, the buffers hold more in the same 1 TB
- [ ] **Run it:** January 2020, 744 hours, about 190 GB (*estimate*); it becomes the first fine buffer

**Done when:** the pilot month is on scratch and every check in the report passes.

---

## Phase 2. Rotating buffers and the QBO series (M2)

- [ ] **QBO index, once:** tropical (10S-10N) zonal-mean wind at the top level, daily 1979-2025, from the m = 0 coefficients only (tiny download, a file of a few MB)
  - Decide: lookback and level. *Recommend September's 30-day lookback; check the series against the published QBO index.*
- [ ] **Buffer filler:** a resumable job that picks a random stretch from the training years (1979-2018), downloads each hour once, and writes
  - the fine frames (hourly, nside 256) to the fine buffer
  - the 6-hourly coarse frames (nside 64) to the coarse buffer
- [ ] **Stretch lengths:** coarse 10 days (at least 72 h for a 12-frame block); fine 1 day (at least 13 h for a 13-frame block) (ASSUMED)
- [ ] **Eviction:** when a buffer is full, drop the oldest stretch before writing a new one; never exceed its budget
- [ ] **Manifest:** which hours each buffer holds, with source and checksum, so any training batch can be traced to its ERA5 hours
- [ ] **Validation and evaluation years** never enter the buffers; a fixed small set of 2019-2020 stretches is kept for checks
- [ ] **Checks**
  - [ ] Both buffers stay under budget across a full refill
  - [ ] Each buffer's stretches span the whole 1979-2018 range (histogram by year)
  - [ ] A killed filler resumes without duplicating or losing hours
  - [ ] Throughput measured: hours per day of filling, at 8 and 16 parallel downloads

**Done when:** both buffers fill, rotate and stay under budget, and the QBO series shows its ~28-month cycle.

At this budget (*estimate*): the fine buffer holds about 1,600 hours (~67 days) and refills in about 3.3 h of downloading; the coarse buffer holds about 25,000 coarse frames (~17 years' worth, spread across all 40) and takes about 5 days of downloading (58,440 six-hourly frames x 7.5 s) to cycle through all 40 training years once.

---

## Phase 3. Stage 1 code (M3)

Port from `balloon-env-dev/src/eval/windeval/hpx/` (`net.py`, `train_coarse.py`, `dataset.py`, `layout.py`) rather than rewriting; it passed its gates in September.

- [ ] **Data loader:** random 12 x 6 h windows from the coarse buffer; calendar and local solar time at 8 frequencies; QBO index; channel mask
- [ ] **Network:** HEALPix U-Net with earth2grid padding; learned position map; attention at the coarsest levels
- [ ] **Diffusion:** EDM preconditioning and loss; log-uniform training noise; 18-step Heun sampler
- [ ] **Masked frames:** some training blocks get a few real frames as conditioning, some none
- [ ] **Training loop:** EMA, checkpoints, resume after preemption, run log; multi-GPU (DDP) wired but run on 1 GPU
- [ ] **Tests**
  - [ ] Shapes and padding across HEALPix face edges
  - [ ] Overfits one batch to near-zero loss
  - [ ] Same seed gives the same first 10 steps

**Done when:** a short run trains from the coarse buffer, resumes after a kill, and records its config, commit and seed.

---

## Phase 4. Stage 2 code (M5)

- [ ] Port the September fine model: residual patches (64 px, 32 px padding), 13 hourly frames, Stage 1 frames lifted as conditioning, exact block-mean projection
- [ ] Data loader from the fine buffer, with its coarse conditioning taken from the same hours
- [ ] **Tests:** block means equal the coarse input to float tolerance; overfits one batch; resumes after a kill

**Done when:** a short Stage 2 run trains from the fine buffer and its block means match the coarse input.

---

## Phase 5. Time tiling (M6)

- [ ] InfiniteDiffusion in time: overlapping blocks denoised together, noise seeded by a hash of (time, place, seed)
- [ ] **Tests**
  - [ ] Same seed and time give identical output in any query order
  - [ ] Asking for day 300 does not require generating days 1-299
- [ ] Measure cost: GPU-seconds per simulated day, and cold-query time

**Done when:** a 30-day run is reproducible to the bit.

---

## Phase 6. Export and interface (M7)

- [ ] Query: (time span, region, seed) to a record with u, v, T, Z and pressure and height per level
- [ ] Export to the balloon-sim record format (`u, v, hours, altitude_km, latitude, longitude`)
- [ ] **Check:** balloon-sim loads an exported record and flies a controller through it

**Done when:** a generated record flies in balloon-sim end to end.

---

## Phase 7. MVP: the whole system at demo scale (for Sarah)

- [ ] **Write first:** what the demo is expected to show and fail at, and the thresholds for each check
- [ ] Train a small Stage 1 and a small Stage 2 from the buffers, 1 seed, about a day or two of one GPU each (*estimate*)
- [ ] Run every check (M1-M7) and report pass or fail honestly; a failed realism check is a result, not a blocker
- [ ] Generate a 30-day record, fly it in balloon-sim, and make the side-by-side figures with ERA5
- [ ] Update the model card with demo results, labelled as demo scale

**Done when:** Sarah has seen a generated record fly, and every check has a number beside it.

---

## Phase 8. Scale up (M4, M5 and M8, as resources grow)

Each step is a config change; repeat the M4 and M5 checks after each.

- [ ] **Noise maximum:** compute cBottle's rule (A.3.1) from the coarse buffer, then sweep around it
- [ ] **Small-scale sweep (write the hypothesis first):** network size and noise maximum, 3 training seeds per setting
  - Scientific: network size and noise maximum. Nuisance: learning rate. Fixed: everything else, stated.
  - [ ] Report validation loss by noise level, mean and 95% interval over seeds
- [ ] **Model of record:** chosen settings, trained long enough for the coarse buffer to cycle through all training years several times; 1 seed, then 3
- [ ] **Bigger budget when storage appears:** larger buffers in the config; same code
- [ ] **More GPUs when available:** DDP across GPUs; a CUGeoData profile, with one job run there to prove portability
- [ ] **Checks with intervals over seeds:** M4 (seasonal cycle, QBO phase, zonal-mean wind, spectra, calm-level frequency) and M5 (block means, spectra to 25 km, seams)

**Done when:** both stages pass M4 and M5 with 3 seeds and intervals.

---

## Phase 9. Research-grade release (v1.0)

- [ ] **Reproducible:** one command rebuilds each check's figure from a checkpoint and config
- [ ] **Data:** buffer manifests archived with each model, so every training hour is traceable; normalisation stats versioned
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
