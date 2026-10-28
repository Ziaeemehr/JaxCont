"""Explicit, installation-free Julia subprocess boundary; never imported by docs."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class JuliaRuntime:
    executable: str | None
    project: Path
    depot: str | None


def resolve_julia(cli=None, *, project=None, depot=None):
    """Resolve CLI > environment > PATH (project defaults to this directory)."""
    executable = cli or os.environ.get("JULIA_BIN") or shutil.which("julia")
    return JuliaRuntime(
        str(Path(executable).expanduser().resolve()) if executable else None,
        Path(project or os.environ.get("BIFURCATIONKIT_PROJECT") or Path(__file__).parent)
        .expanduser()
        .resolve(),
        depot if depot is not None else os.environ.get("JULIA_DEPOT_PATH"),
    )


def run_julia(script, args, timeout, *, cli=None, project=None, depot=None, log_dir=None):
    """Run headless Julia, persist both streams, and propagate failures."""
    runtime = resolve_julia(cli, project=project, depot=depot)
    if not runtime.executable or not shutil.which(runtime.executable):
        raise RuntimeError("Julia is unavailable (set JULIA_BIN)")
    if not (runtime.project / "Manifest.toml").is_file():
        raise RuntimeError(f"Julia manifest is missing: {runtime.project}")
    if "--output-dir" in args:
        from ...artifacts import candidate_directory
        output = candidate_directory(args[args.index("--output-dir") + 1])
        (output / "metadata.json").unlink(missing_ok=True)
    script = Path(script).expanduser().resolve()
    logs = Path(log_dir or script.parent).expanduser().resolve()
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if runtime.depot is not None:
        environment["JULIA_DEPOT_PATH"] = runtime.depot
    command = [
        runtime.executable,
        "--startup-file=no",
        f"--project={runtime.project}",
        str(script),
        *map(str, args),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=runtime.project,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        _save_logs(logs, script.stem, exc.stdout, exc.stderr)
        raise RuntimeError(f"Julia {script.name} timed out after {timeout}s; logs: {logs}") from exc
    _save_logs(logs, script.stem, result.stdout, result.stderr)
    if result.returncode:
        raise RuntimeError(f"Julia {script.name} exited {result.returncode}; logs: {logs}")
    return result


def _save_logs(directory, stem, stdout, stderr):
    for stream, value in (("stdout", stdout), ("stderr", stderr)):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        (directory / f"{stem}.{stream}.log").write_text(value or "", encoding="utf-8")
