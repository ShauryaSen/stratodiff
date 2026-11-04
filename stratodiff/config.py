"""One frozen dataclass of settings per piece. A YAML file in configs/ overrides the defaults, and an
unknown key is an error, so a typo can never silently fall back to a default. Basis for each value: DECISIONS.md"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class DataConfig:
    start: str = "2020-01-01T00"  # first hour ingested, UTC
    end: str = "2020-01-31T23"  # last hour ingested, inclusive
    first_level: int = 29  # L137 model level at the top of the band, about 9.8 hPa (D3)
    last_level: int = 68  # L137 model level at the bottom, about 148 hPa
    variables: tuple[str, ...] = ("u", "v", "t", "z")  # 4 x 40 levels = 160 channels (D5)
    nside: int = 256  # HEALPix grid the store is written on, about 25 km
    dtype: Literal["float16", "float32"] = "float16"  # stored after normalising (D8)
    store: str = "era5_hpx256"  # name of the store under the cluster's data_root


@dataclass(frozen=True)
class BufferConfig:
    """The rotating buffers (D10, D11): random stretches of the training years, replaced as training goes,
    so no full copy of the archive is ever on disk. A bigger budget means bigger buffers, nothing else."""

    coarse_tb: float = 0.4  # coarse frames, nside 64, 6-hourly
    fine_tb: float = 0.4  # fine frames, nside 256, hourly
    coarse_stretch_days: int = 10  # length of one random stretch; at least one 12 x 6 h block
    fine_stretch_hours: int = 24  # at least one 13-hour block
    first_year: int = 1979  # training years only; validation and evaluation never enter (D2)
    last_year: int = 2018


@dataclass(frozen=True)
class ModelConfig:
    stage: Literal[1, 2] = 1  # 1: global sketch, 2: fine residual patches
    nside: int = 64  # grid the stage generates on
    frames: int = 12  # frames per block (M5)
    frame_hours: int = 6  # hours between frames
    base_channels: int = 128  # September's value, to be set by the M3 sweep (M4)
    sigma_max: float = 1000.0  # top of the log-uniform training noise, September's value, to be swept (M3, M4)
    sampler_steps: int = 18  # Heun steps from noise to weather (M8)
    calendar_frequencies: int = 8  # sine/cosine pairs for day of year and for local solar time (M9)
    qbo_index: bool = True  # condition on the historical QBO index (M9)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0  # fixes initial weights, batch order and training noise
    batch_size: int = 4
    learning_rate: float = 1e-4
    ema_decay: float = 0.9999
    steps: int = 100_000


@dataclass(frozen=True)
class ClusterConfig:
    name: str = "local"
    data_root: str = "data"  # where stores live
    runs_root: str = "runs"  # where run directories are made
    partition: str = ""  # Slurm partition; empty means run here, no Slurm
    account: str = ""
    nodelist: str = ""  # pin to a node when its local disk holds the data
    gpus: int = 0
    cpus: int = 4
    mem_gb: int = 16
    hours: int = 4  # wall-clock limit
    disk_budget_tb: float = 1.0  # the most this project may hold on the cluster's data disk at once (D10)
    setup: tuple[str, ...] = ()  # shell lines run before the command, e.g. loading modules


@dataclass(frozen=True)
class RunConfig:
    name: str = "unnamed"
    data: DataConfig = field(default_factory=DataConfig)
    buffers: BufferConfig = field(default_factory=BufferConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, values: dict, where: str):
    """Make cls from a dict, recursing into nested settings and refusing keys cls does not have."""
    if not isinstance(values, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(values).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"{where}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}")
    out = {}
    for key, value in values.items():
        kind = hints[key]
        if is_dataclass(kind):
            out[key] = _build(kind, value, f"{where}.{key}")
        elif get_origin(kind) is tuple:
            out[key] = tuple(value)
        elif get_origin(kind) is Literal and value not in get_args(kind):
            raise ValueError(f"{where}.{key}: {value!r} is not one of {get_args(kind)}")
        else:
            out[key] = value
    return replace(cls(), **out)


def load(path: str | Path) -> RunConfig:
    """A run's settings: the defaults above with the YAML file's values over them."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    return _build(RunConfig, values, str(path))


def load_cluster(name: str, root: str | Path = "configs/clusters") -> ClusterConfig:
    """A cluster profile by name, from configs/clusters/<name>.yaml."""
    path = Path(root) / f"{name}.yaml"
    return _build(ClusterConfig, yaml.safe_load(path.read_text()) or {}, str(path))


def to_dict(config) -> dict:
    """Plain nested dict, tuples as lists, ready to write as YAML."""

    def plain(x):
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, tuple):
            return [plain(v) for v in x]
        return x

    return plain(asdict(config))


def frame_bytes(nside: int, data: DataConfig) -> int:
    """Bytes in one stored frame: every channel on a HEALPix grid of this nside."""
    channels = len(data.variables) * (data.last_level - data.first_level + 1)
    return 12 * nside**2 * channels * {"float16": 2, "float32": 4}[data.dtype]


def check_budget(config: RunConfig, cluster: ClusterConfig) -> None:
    """Refuse buffers that would not fit the cluster's disk budget, before anything is downloaded."""
    used = config.buffers.coarse_tb + config.buffers.fine_tb
    if used > cluster.disk_budget_tb:
        raise ValueError(f"buffers need {used} TB but {cluster.name} allows {cluster.disk_budget_tb} TB; "
                         "shrink buffers.coarse_tb or buffers.fine_tb")
