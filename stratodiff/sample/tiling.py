"""InfiniteDiffusion in time (M6): overlapping blocks denoised together, each block's noise seeded by its
own time and place, so any query is reproducible and independent of order (arXiv 2512.08309, 3.4-3.5)."""

from ..config import ClusterConfig, RunConfig


def sample(config: RunConfig, cluster: ClusterConfig) -> None:
    raise NotImplementedError("M6: time tiling, after a Stage 1 model of record (SPEC.md)")
