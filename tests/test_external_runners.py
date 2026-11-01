"""Offline runner contracts and explicit, installation-free Julia smoke test."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.validation import run_validation as cli


def test_default_run_never_launches_external_tools(monkeypatch):
    monkeypatch.setattr(cli, "resolve_matlab_bin", lambda *a: pytest.fail("offline resolution"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("external launch"))
    assert cli.main(["--case", "MC-EQ-001", "--dry-run"]) == 0


def test_julia_resolution_precedence(tmp_path, monkeypatch):
    from examples.validation.regenerate.bifurcationkit.runner import resolve_julia

    monkeypatch.setenv("JULIA_BIN", "/environment/julia")
    monkeypatch.setenv("BIFURCATIONKIT_PROJECT", str(tmp_path))
    monkeypatch.setenv("JULIA_DEPOT_PATH", "/environment/depot")
    runtime = resolve_julia("/explicit/julia")
    assert runtime.executable == "/explicit/julia"
    assert runtime.project == tmp_path
    assert runtime.depot == "/environment/depot"
    assert resolve_julia().executable == "/environment/julia"
    monkeypatch.delenv("JULIA_BIN")
    monkeypatch.setattr("shutil.which", lambda name: "/path/julia")
    assert resolve_julia().executable == "/path/julia"


def test_selected_marker_missing_runtime_fails(monkeypatch, tmp_path):
    from examples.validation.regenerate.bifurcationkit.runner import run_julia

    monkeypatch.setenv("JULIA_BIN", "/missing/julia")
    with pytest.raises(RuntimeError, match="Julia is unavailable"):
        run_julia(tmp_path / "smoke.jl", [], 1, log_dir=tmp_path)


def test_timeout_and_nonzero_exit_keep_logs(tmp_path, monkeypatch):
    from examples.validation.regenerate.bifurcationkit.runner import run_julia

    # Any existing executable passes the lookup; subprocess.run is patched below.
    monkeypatch.setenv("JULIA_BIN", sys.executable)
    script = tmp_path / "smoke.jl"
    script.write_text("")
    with pytest.raises(RuntimeError, match="exited 7"):
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 7, "out", "err")
        )
        run_julia(script, [], 1, log_dir=tmp_path)
    assert (tmp_path / "smoke.stdout.log").read_text() == "out"
    assert (tmp_path / "smoke.stderr.log").read_text() == "err"

    def timeout(*a, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL
        raise subprocess.TimeoutExpired(a[0], 1, output=b"partial", stderr=b"timeout")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        run_julia(script, [], 1, log_dir=tmp_path)
    assert (tmp_path / "smoke.stdout.log").read_text() == "partial"


@pytest.mark.julia
def test_julia_environment_smoke(tmp_path):
    from examples.validation.regenerate.bifurcationkit.runner import run_julia, resolve_julia

    project = resolve_julia().project
    script = tmp_path / "environment.jl"
    script.write_text(
        "using BifurcationKit, Accessors, CSV, JSON, SHA, LinearAlgebra\n"
        'print(JSON.json(Dict("manifest_hash" => bytes2hex(sha256(read(joinpath(dirname(Base.active_project()), "Manifest.toml")))))))\n'
    )
    result = run_julia(script, [], 180, log_dir=tmp_path)
    assert (
        json.loads(result.stdout)["manifest_hash"]
        == hashlib.sha256((project / "Manifest.toml").read_bytes()).hexdigest()
    )
