import json
import subprocess
from datetime import datetime, timezone

import pytest
import yaml

from stratodiff.config import RunConfig
from stratodiff.runlog import start_run


@pytest.fixture
def repo(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("a\n")
    git("add", ".")
    git("commit", "-qm", "c")
    return tmp_path


NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def test_records_config_commit_and_seed(repo, tmp_path):
    run = start_run(RunConfig(name="t"), tmp_path / "runs", repo, now=NOW)
    meta = json.loads((run / "run.json").read_text())
    assert len(meta["commit"]) == 40 and meta["dirty"] is False and meta["seed"] == 0
    assert yaml.safe_load((run / "config.yaml").read_text())["name"] == "t"
    assert run.name.startswith("20261103-120000-t-")


def test_dirty_tree_is_refused_then_kept_when_allowed(repo, tmp_path):
    (repo / "f.txt").write_text("b\n")
    with pytest.raises(RuntimeError):
        start_run(RunConfig(), tmp_path / "runs", repo, now=NOW)
    run = start_run(RunConfig(), tmp_path / "runs", repo, allow_dirty=True, now=NOW)
    assert "+b" in (run / "uncommitted.diff").read_text()
