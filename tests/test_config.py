from __future__ import annotations

import itertools
import shutil
from pathlib import Path

import pytest
import yaml

from mfsi_cxr.utils.config import (
    ALLOWED_PROFILE_KEYS,
    CONFIG_DIR,
    DATA_ROOT_ENV,
    PROFILE_ENV,
    PROFILES,
    ConfigError,
    config_hash,
    differing_keys,
    load_config,
    resolve_profile,
)


@pytest.mark.parametrize("profile", PROFILES)
def test_each_profile_loads(profile: str) -> None:
    cfg = load_config(profile)
    assert cfg["profile"] == profile


def test_profiles_differ_only_in_allowed_keys() -> None:
    cfgs = {p: load_config(p) for p in PROFILES}
    for a, b in itertools.combinations(PROFILES, 2):
        diff = differing_keys(cfgs[a], cfgs[b]) - {"profile"}
        assert diff <= ALLOWED_PROFILE_KEYS, f"{a} vs {b}: {sorted(diff - ALLOWED_PROFILE_KEYS)}"


def test_forbidden_override_is_rejected(tmp_path: Path) -> None:
    cdir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, cdir)
    profile_file = cdir / "profiles" / "smoke.yaml"
    override = yaml.safe_load(profile_file.read_text(encoding="utf-8"))
    override["data"]["max_priors"] = 1  # a method choice, not a size/speed
    profile_file.write_text(yaml.safe_dump(override), encoding="utf-8")
    with pytest.raises(ConfigError, match="data.max_priors"):
        load_config("smoke", config_dir=cdir)


def test_profile_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_profile(None) == "pilot"
    monkeypatch.setenv(PROFILE_ENV, "smoke")
    assert resolve_profile(None) == "smoke"
    assert resolve_profile("full") == "full"
    with pytest.raises(ConfigError):
        resolve_profile("huge")


def test_data_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert load_config("pilot")["data"]["root"] == str(tmp_path)


def test_config_hash_is_stable_and_sensitive() -> None:
    a, b = load_config("smoke"), load_config("smoke")
    assert config_hash(a) == config_hash(b)
    b["eval"]["bootstrap_B"] += 1
    assert config_hash(a) != config_hash(b)


@pytest.mark.parametrize("profile", ["smoke", "pilot", "full"])
def test_no_number_is_read_as_a_string(profile: str) -> None:
    """PyYAML reads 1.0e10 (exponent without sign) as a string; numbers must be written as 1.0e+10."""
    from mfsi_cxr.utils.config import flatten

    for key, value in flatten(load_config(profile)).items():
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                continue
            raise AssertionError(f"{key} = {value!r} is a number stored as a string")
