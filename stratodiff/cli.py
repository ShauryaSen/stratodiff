"""stratodiff's one command line. Each task runs here, or is sent to a cluster with `submit`:

    uv run stratodiff show-config configs/stage1.yaml
    uv run stratodiff submit --cluster unicorn train configs/stage1.yaml          # prints the job
    uv run stratodiff submit --cluster unicorn train configs/stage1.yaml --go     # and submits it"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from . import cluster as slurm
from .config import load, load_cluster, to_dict

TASKS = ("ingest", "train", "sample", "eval")


def run_task(task: str, config_path: str, cluster_name: str, allow_dirty: bool) -> None:
    config = load(config_path)
    where = load_cluster(cluster_name)
    if task == "ingest":
        from .data.ingest import ingest

        ingest(config, where)
    elif task == "train":
        from .model.train import train

        train(config, where, allow_dirty=allow_dirty)
    elif task == "sample":
        from .sample.tiling import sample

        sample(config, where)
    elif task == "eval":
        from .eval.checks import evaluate

        evaluate(config, where)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="stratodiff")
    sub = p.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show-config", help="print a config with every default filled in")
    show.add_argument("config")
    for task in TASKS:
        t = sub.add_parser(task, help=f"run {task} here")
        t.add_argument("config")
        t.add_argument("--cluster", default="local")
        t.add_argument("--allow-dirty", action="store_true", help="run from uncommitted code; the diff is saved")
    s = sub.add_parser("submit", help="send a task to a cluster as a Slurm job")
    s.add_argument("--cluster", required=True)
    s.add_argument("task", choices=TASKS)
    s.add_argument("config")
    s.add_argument("--go", action="store_true", help="submit; without it the job script is only printed")
    s.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args(argv)

    if a.command == "show-config":
        print(yaml.safe_dump(to_dict(load(a.config)), sort_keys=False), end="")
    elif a.command == "submit":
        config, where = load(a.config), load_cluster(a.cluster)
        if a.task == "ingest":
            where = replace(where, gpus=0)  # ingest is CPU and network work; do not hold a GPU for it
        command = ["uv", "run", "stratodiff", a.task, a.config, "--cluster", a.cluster]
        if a.allow_dirty:
            command.append("--allow-dirty")
        logs = Path(where.runs_root) / "slurm"
        script = slurm.job_script(where, f"{a.task}-{config.name}", command, logs)
        if a.go:
            print(slurm.submit(script, logs / f"{a.task}-{config.name}.sbatch"))
        else:
            print(script, end="")
    else:
        run_task(a.command, a.config, a.cluster, a.allow_dirty)


if __name__ == "__main__":
    sys.exit(main())
