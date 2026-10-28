# Pinned BifurcationKit environment

The supplied Project.toml and Manifest.toml pin Julia 1.13.3 and
BifurcationKit 0.8.8. Preserve the lockfile; do not resolve it again unless a
required package is missing. Revise and Plots are not dependencies.

Explicit setup (may write to the default ~/.julia depot):

```sh
~/.juliaup/bin/julia --project=examples/validation/regenerate/bifurcationkit examples/validation/regenerate/bifurcationkit/setup.jl </dev/null
```

Setup instantiates, precompiles, and prints a JSON version/hash report.
Nothing installs during pytest, ordinary validation, or documentation builds.
To smoke-test imports without setup:

```sh
JULIA_BIN=~/.juliaup/bin/julia python3 -m pytest tests/test_external_runners.py -m julia -q
```

The runner resolves CLI overrides, then JULIA_BIN, BIFURCATIONKIT_PROJECT,
JULIA_DEPOT_PATH, then PATH (the project defaults to this directory).
Subprocesses close stdin, have a timeout, and retain stdout/stderr on failure.
Only opt-in producers write candidates in an explicit output directory;
reviewed reference data is never an output target. The existing scientific
Julia scripts are historical inputs pending the later producer tasks.
