"""Turn a command into a Slurm job for a cluster profile, so moving clusters is a new YAML file, not new code"""

import shlex
import subprocess
from pathlib import Path

from .config import ClusterConfig


def job_script(cluster: ClusterConfig, job: str, command: list[str], log_dir: str | Path) -> str:
    """The sbatch script that runs command on cluster, logging to log_dir/<job>-<jobid>.out."""
    lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name=stratodiff-{job}"]
    if cluster.partition:
        lines.append(f"#SBATCH --partition={cluster.partition}")
    if cluster.account:
        lines.append(f"#SBATCH --account={cluster.account}")
    if cluster.nodelist:
        lines.append(f"#SBATCH --nodelist={cluster.nodelist}")
    if cluster.gpus:
        lines.append(f"#SBATCH --gres=gpu:{cluster.gpus}")
    lines += [
        f"#SBATCH --cpus-per-task={cluster.cpus}",
        f"#SBATCH --mem={cluster.mem_gb}G",
        f"#SBATCH --time={cluster.hours}:00:00",
        f"#SBATCH --output={Path(log_dir) / f'{job}-%j.out'}",
        "set -euo pipefail",
        *cluster.setup,
        shlex.join(command),
        "",
    ]
    return "\n".join(lines)


def submit(script: str, path: str | Path) -> str:
    """Write the script and hand it to sbatch; returns sbatch's reply (the job id line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    return subprocess.run(["sbatch", str(path)], capture_output=True, text=True, check=True).stdout.strip()
