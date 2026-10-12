from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from masa.common import dynamics_cache
from masa.common.dynamics_cache import (
    cache_enabled,
    cache_key,
    cache_path,
    cached_dynamics,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MASA_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("MASA_DISABLE_DYNAMICS_CACHE", raising=False)
    return tmp_path


def test_cache_lives_in_the_gitignored_checkout_folder_by_default(monkeypatch):
    monkeypatch.delenv("MASA_CACHE_DIR", raising=False)
    import masa

    checkout = Path(masa.__file__).resolve().parents[1]

    assert dynamics_cache.cache_dir() == checkout / ".cache" / "dynamics"
    assert subprocess.run(
        ["git", "check-ignore", "-q", str(checkout / ".cache" / "dynamics" / "x.pkl")],
        cwd=checkout,
    ).returncode == 0


def test_cache_dir_falls_back_outside_a_checkout(monkeypatch):
    monkeypatch.delenv("MASA_CACHE_DIR", raising=False)
    monkeypatch.setattr(dynamics_cache, "_repo_root", lambda: None)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg")

    assert dynamics_cache.cache_dir() == Path("/tmp/xdg/masa/dynamics")


def test_second_call_reads_from_disk_instead_of_recomputing():
    calls = []

    def compute():
        calls.append(1)
        return {"states": [1, 2, 3]}

    first = cached_dynamics("demo", compute)
    second = cached_dynamics("demo", compute)

    assert first == second == {"states": [1, 2, 3]}
    assert len(calls) == 1


def test_cache_round_trips_the_dynamics_shapes_envs_actually_use():
    from collections import defaultdict

    successors = defaultdict(list)
    successors[(0, 1)] = [2, 3]
    probs = defaultdict(list)
    probs[(0, 1)] = [0.4, 0.6]
    payload = (successors, probs, None, 7, {(0, 0, 1): 0}, {0: (0, 0, 1)})

    cached_dynamics("shapes", lambda: payload)
    restored = cached_dynamics("shapes", lambda: pytest.fail("recomputed"))

    assert restored[0][(0, 1)] == [2, 3]
    assert restored[1][(0, 1)] == [0.4, 0.6]
    assert restored[2] is None
    assert restored[3] == 7
    assert restored[4] == {(0, 0, 1): 0}
    assert restored[5] == {0: (0, 0, 1)}


def test_key_tracks_the_inputs_the_dynamics_depend_on():
    layout = np.array([[1, 1], [1, 0]])
    other = np.array([[1, 1], [0, 0]])

    assert cache_key("pacman", layout, 4, 0.6) == cache_key("pacman", layout.copy(), 4, 0.6)
    assert cache_key("pacman", layout, 4, 0.6) != cache_key("pacman", other, 4, 0.6)
    assert cache_key("pacman", layout, 4, 0.6) != cache_key("pacman", layout, 5, 0.6)
    assert cache_key("pacman", layout, 4, 0.6) != cache_key("pacman", layout, 4, 0.5)
    assert cache_key("pacman", layout) != cache_key("mini-pacman", layout)


def test_a_corrupt_cache_file_is_recomputed_rather_than_raising():
    path = cache_path("corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a pickle")

    assert cached_dynamics("corrupt", lambda: "recomputed") == "recomputed"
    # The unreadable file is replaced, so the next process gets a cache hit.
    assert cached_dynamics("corrupt", lambda: pytest.fail("recomputed twice")) == "recomputed"


def test_an_unwritable_cache_dir_still_returns_the_value(monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(dynamics_cache, "_write_atomically", explode)

    assert cached_dynamics("unwritable", lambda: 42) == 42


def test_caching_can_be_disabled_with_an_env_var(monkeypatch):
    monkeypatch.setenv("MASA_DISABLE_DYNAMICS_CACHE", "1")
    calls = []

    def compute():
        calls.append(1)
        return "value"

    assert cache_enabled() is False
    assert cached_dynamics("disabled", compute) == "value"
    assert cached_dynamics("disabled", compute) == "value"
    assert len(calls) == 2
    assert not cache_path("disabled").exists()
