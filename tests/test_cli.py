import os
from pathlib import Path

import pytest

from stratodiff.cli import main

ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def at_repo_root():
    here = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(here)


def test_show_config_fills_defaults(capsys):
    main(["show-config", "configs/stage1.yaml"])
    out = capsys.readouterr().out
    assert "sampler_steps: 18" in out and "name: stage1" in out


def test_submit_prints_a_job_pinned_to_the_data_node(capsys):
    main(["submit", "--cluster", "unicorn", "train", "configs/stage1.yaml"])
    out = capsys.readouterr().out
    assert "#SBATCH --nodelist=dean-compute-02" in out and "#SBATCH --gres=gpu:1" in out
    assert out.strip().endswith("uv run stratodiff train configs/stage1.yaml --cluster unicorn")


def test_ingest_jobs_hold_no_gpu(capsys):
    main(["submit", "--cluster", "unicorn", "ingest", "configs/pilot_month.yaml"])
    assert "--gres" not in capsys.readouterr().out


def test_unbuilt_tasks_say_which_milestone_builds_them():
    with pytest.raises(NotImplementedError, match="M1"):
        main(["ingest", "configs/pilot_month.yaml"])
