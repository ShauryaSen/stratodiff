"""Every run gets a directory holding what is needed to repeat it: the resolved settings, the git commit,
whether the tree was dirty (and the diff if so), and the seed"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import RunConfig, to_dict


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def git_state(repo: Path) -> dict:
    """The commit and whether anything tracked differs from it."""
    commit = _git("rev-parse", "HEAD", cwd=repo).strip()
    diff = _git("diff", "HEAD", cwd=repo)
    return {"commit": commit, "dirty": bool(diff), "diff": diff}


def start_run(config: RunConfig, runs_root: str | Path, repo: str | Path = ".", allow_dirty: bool = False,
              now: datetime | None = None) -> Path:
    """Make the run's directory and record it. Refuses a dirty tree unless allow_dirty, and then keeps the diff."""
    repo = Path(repo)
    state = git_state(repo)
    if state["dirty"] and not allow_dirty:
        raise RuntimeError("uncommitted changes: commit them, or pass allow_dirty to keep the diff beside the run")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    run = Path(runs_root) / f"{stamp}-{config.name}-{state['commit'][:8]}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(yaml.safe_dump(to_dict(config), sort_keys=False))
    meta = {
        "commit": state["commit"],
        "dirty": state["dirty"],
        "seed": config.train.seed,
        "started_utc": stamp,
        "python": sys.version.split()[0],
        "host": platform.node(),
        "argv": sys.argv,
    }
    (run / "run.json").write_text(json.dumps(meta, indent=2))
    if state["dirty"]:
        (run / "uncommitted.diff").write_text(state["diff"])
    return run
