"""Training either stage (M3-M5). Starts by recording the run with runlog.start_run, so every checkpoint
has its config, commit and seed beside it."""

from ..config import ClusterConfig, RunConfig


def train(config: RunConfig, cluster: ClusterConfig, allow_dirty: bool = False) -> None:
    raise NotImplementedError("M3: Stage 1 training, after the M2 dataset exists (SPEC.md)")
