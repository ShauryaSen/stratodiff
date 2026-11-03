# stratodiff

A diffusion generator of global stratospheric wind for balloon navigation: random, reproducible weather on 40
native ERA5 levels (13.7-31.3 km), about 25 km and hourly, at any time and place.

**Status (2026-11-03):** specified and scaffolded. Nothing is trained, so no performance is claimed.
What it will be: [SPEC.md](SPEC.md). Step by step: [ROADMAP.md](ROADMAP.md). Why each choice:
[DECISIONS.md](DECISIONS.md).

## Layout

```
stratodiff/
  config.py       settings: frozen dataclasses, overridden by strict YAML
  runlog.py       what every run records: config, commit, dirty diff, seed
  cluster.py      a command to a Slurm job for a cluster profile
  cli.py          the one command line
  data/spectral.py  ERA5 spherical harmonics to exact u, v, T at any point (built, tested)
  data/ingest.py    ERA5 to the HEALPix store                       (M1)
  model/train.py    the two diffusion stages                        (M3-M5)
  sample/tiling.py  InfiniteDiffusion in time                       (M6)
  eval/checks.py    each milestone's pass condition                 (M4 on)
  export.py         balloon-sim records with pressure and height    (M7)
configs/          run settings; configs/clusters/ has one profile per cluster
tests/            CPU-only, seconds; network and GPU tests are marked and skipped
tools/            one-off checks, e.g. the spectral check against ARCO's grid
```

## Commands

```bash
uv run pytest -q
uv run stratodiff show-config configs/stage1.yaml
uv run stratodiff submit --cluster unicorn ingest configs/pilot_month.yaml        # print the job
uv run stratodiff submit --cluster unicorn ingest configs/pilot_month.yaml --go   # submit it
```

## Licence

Apache-2.0.
