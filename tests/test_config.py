from pathlib import Path

import pytest

from stratodiff.config import RunConfig, load, load_cluster, to_dict

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    assert isinstance(load(path), RunConfig)


@pytest.mark.parametrize("name", ["local", "unicorn", "cugeodata"])
def test_every_cluster_profile_loads(name):
    assert load_cluster(name, CONFIGS / "clusters").name == name


def test_yaml_overrides_only_what_it_names(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("name: x\nmodel:\n  frames: 8\n")
    c = load(f)
    assert c.model.frames == 8 and c.model.nside == RunConfig().model.nside and c.name == "x"


@pytest.mark.parametrize("text", ["nmae: x\n", "model:\n  frams: 8\n", "train:\n  seed: 1\n  lr: 1\n"])
def test_unknown_keys_are_refused(tmp_path, text):
    f = tmp_path / "c.yaml"
    f.write_text(text)
    with pytest.raises(KeyError):
        load(f)


def test_literal_values_are_checked(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("data:\n  dtype: float64\n")
    with pytest.raises(ValueError):
        load(f)


def test_round_trip_through_yaml(tmp_path):
    import yaml

    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(to_dict(RunConfig())))
    assert load(f) == RunConfig()
