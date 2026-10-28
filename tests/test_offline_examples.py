"""Gate G2: execute each gallery source with external runtimes hidden."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = sorted((ROOT / "examples/tutorials").glob("plot_*.py")) + sorted(
    (ROOT / "examples/validation/gallery").glob("plot_*.py")
)


def offline_environment(tmp_path):
    """Hide runtime configuration and reject direct or shell external launches."""
    guard = tmp_path / "guard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text("""
import os
import shlex
import subprocess
from pathlib import Path
_original_popen = subprocess.Popen

def guarded_popen(command, *args, **kwargs):
    tokens = shlex.split(command) if isinstance(command, str) else list(command)
    expanded = []
    for token in tokens:
        text = os.fsdecode(token)
        expanded.extend(shlex.split(text))
    for token in tokens + expanded:
        name = Path(os.fsdecode(token)).name.lower()
        if name in ('matlab', 'julia', 'juliaup') or name.startswith(('matlab-', 'julia-')):
            raise RuntimeError('offline example attempted external launch: ' + name)
    executable = kwargs.get('executable')
    if executable and Path(executable).name.lower() in ('matlab', 'julia', 'juliaup'):
        raise RuntimeError('offline example attempted external executable')
    return _original_popen(command, *args, **kwargs)
subprocess.Popen = guarded_popen
""")
    env = os.environ.copy()
    for key in (
        "MATLAB_BIN",
        "MATCONT_ROOT",
        "JULIA_BIN",
        "BIFURCATIONKIT_PROJECT",
        "JULIA_PROJECT",
        "JULIA_DEPOT_PATH",
    ):
        env.pop(key, None)
    env["PATH"] = os.pathsep.join(
        p
        for p in env["PATH"].split(os.pathsep)
        if not any((Path(p) / exe).exists() for exe in ("matlab", "julia", "juliaup"))
    )
    env.update(
        JAX_PLATFORMS="cpu",
        MPLBACKEND="Agg",
        JAX_ENABLE_X64=os.environ.get("JAXCONT_TEST_X64", "0"),
        MPLCONFIGDIR=str(tmp_path / "mpl"),
        PYTHONPATH=os.pathsep.join([str(guard), str(ROOT), str(ROOT / "src")]),
    )
    return env


@pytest.mark.slow
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.stem)
def test_examples_tutorials_and_validation_gallery_run_without_matlab_or_julia(tmp_path, entry):
    env = offline_environment(tmp_path)
    output = tmp_path / "generated"
    run = subprocess.run(
        [sys.executable, str(entry), "--output-dir", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert list(output.glob("*.png"))
    assert not (tmp_path / "images").exists()
    if "gallery" in entry.parts:
        import json

        images = list(output.glob("*.png"))
        for image in images:
            record = json.loads(image.with_suffix(".checks.json").read_text())
            assert record["metrics"] and record["report_hash"]
        if entry.stem == "plot_torbpc":
            assert all(
                json.loads(i.with_suffix(".checks.json").read_text())["status"] == "FAIL"
                for i in images
            )


@pytest.mark.parametrize('command', [
    ['/missing/matlab', '-batch', 'quit'],
    ['/missing/julia', '--version'],
    ['/bin/sh', '-c', 'julia --version'],
])
def test_offline_guard_rejects_external_launches(tmp_path, command):
    env = offline_environment(tmp_path)
    script = f'import subprocess; subprocess.run({command!r})'
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0
    assert 'offline example attempted external launch' in result.stderr
