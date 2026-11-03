# Decisions

Each decision has a tag. CITED: a source someone opened and checked (link and section). MEASURED: our own
run, with what was run. ASSUMED: a judgment call, not yet tested. "Ablate" marks an ASSUMED choice likely to
interact with what we are testing, so it should become an ablation.

## Data

| # | Decision | Why | Tag | Ablate |
|---|---|---|---|---|
| D1 | ERA5 as training data | Global, gridded, hourly, 1940 on | ASSUMED | |
| D2 | Split 1979-2018 / 2019-2020 / 2021-2025, whole years | Evaluation years also lie outside cBottle's training (1980-2017) and validation (2018) | CITED: [NGC model card](https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/cbottle/-), dataset partition | |
| D3 | L137 levels 29-68 (40 levels, 9.8-148 hPa) | Widest useful band; coarser models are compared on a subset | ASSUMED | |
| D4 | Wind and T from native T639 coefficients | ERA5 archives them as T639 ([ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation), Table 12). One hour (2020-01-15 00Z, levels 50 and 65) synthesised by us against ARCO's grid: u, v RMSE 0.03-0.06 m/s, T 0.02 K, vorticity 6.7-7.0% of its spread. Gain over bilinear regridding not measured | CITED, MEASURED (one hour) | yes |
| D5 | Add T and Z | cBottle generates U, V, T, Z ([arXiv 2505.06474v2](https://arxiv.org/html/2505.06474v2), Table 2). Z gives true height | CITED | |
| D6 | Z from ARCO gridded geopotential, bilinear | The spectral store has no Z. Bilinear matches cBottle's regridding (NGC card) | CITED | |
| D7 | Final model mixes batches across all years | Sequential training risks favouring recent years | ASSUMED | |
| D8 | Store normalised float16 | Halves storage. Z must be normalised first (float16 max 65,504) | ASSUMED, checked at M1 | |
| D9 | First month on `dean-compute-02` scratch | The training node, fast local disk, keeps the shared drive free; rebuildable from ARCO | ASSUMED | |
| D10 | Ask for ~110 TB shared storage | 503 MB per hour at float32 for 47 years is 207 TB (arithmetic); float16 halves it | ASSUMED | |

## Model

| # | Decision | Why | Tag | Ablate |
|---|---|---|---|---|
| M1 | Two stages, global sketch then fine residual | Our September model passed its gates this way; cBottle uses the same cascade | MEASURED, CITED | |
| M2 | Stage 1 at nside 64 | cBottle's coarse resolution (NGC card) | CITED | yes |
| M3 | Training noise log-uniform to a high maximum | Without it Stage 1 ignored its conditioning (our L6, September); cBottle A.3.1 states the same requirement | MEASURED, CITED | |
| M4 | Noise maximum and network size from a small-scale sweep | Old values were for 36 channels at nside 32 | ASSUMED | |
| M5 | Stage 1 blocks of 12 x 6 h | cBottle-video's block ([arXiv 2608.19008](https://arxiv.org/abs/2608.19008), section 2) | CITED | yes |
| M6 | Masked-frame training | One set of weights runs tiling and autoregression (cBottle-video, section 2) | CITED | |
| M7 | InfiniteDiffusion tiling in time | Random access and reproducible queries ([arXiv 2512.08309](https://arxiv.org/abs/2512.08309), sections 3.4-3.5); tested on terrain only | CITED method, ASSUMED for time | yes |
| M8 | 18-step Heun sampler | September default; distillation tested later | ASSUMED | yes |
| M9 | Conditioning: calendar and local solar time (8 frequencies), learned position map, QBO index, no SST | cBottle A.7 for the first three; cBottle missed the QBO at 50 hPa in our September run | CITED, MEASURED | yes |
| M10 | Seeds: 3 per small-scale ablation; model of record 1 then 3 | The full run is the largest cost | ASSUMED | |

## Repository (decided 2026-11-03, cheap to change)

| # | Decision | Why | Tag |
|---|---|---|---|
| R1 | uv, hatchling, flat package, pytest | Same as balloon-sim | ASSUMED |
| R2 | Frozen dataclasses overridden by strict YAML | balloon-sim's settings style, plus files Slurm jobs can name | ASSUMED |
| R3 | Cluster profiles in YAML; one `submit` command | Moving to CUGeoData is a new profile, not new code | ASSUMED |
| R4 | Run directory records config, commit, dirty diff, seed | Reproducibility rule | ASSUMED |
| R5 | Apache-2.0 | Same licence as cBottle's code, which we may adapt | ASSUMED |
