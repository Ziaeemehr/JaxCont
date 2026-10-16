## Global Constraints

- Pre-1.0 policy: remove, don't deprecate. Every breaking change gets a `CHANGELOG.md` entry under `## [0.5.0] - Unreleased` → `### Breaking`.
- Never call `jax.config.update("jax_enable_x64", ...)` at import time inside `jaxcont`; precision is the caller's choice.
- Tolerances derive from dtype through one helper, `jaxcont.utils.precision.default_tol(dtype)`: `1e-9` for float64, `1e-5` for float32. No new hard-coded `1e-5`/`1e-6` literals in `src/`.
- Every test in the default suite must pass under both `JAXCONT_TEST_X64=0` and `JAXCONT_TEST_X64=1`.
- No Python control flow on traced values in the engine or in anything reachable under `jit`/`vmap`.
- New source comments state invariants, not incident history; history goes in `CHANGELOG.md`.
- Run tests with `python -m pytest -q -n 8`; slow suite with `python -m pytest -q -m slow`.


## Environment
- Repo root (<repo>): the checkout of branch review-remediation-2026-10 (on the Mac it was a worktree; on Ubuntu it is the clone). Work and commit ONLY on this branch.
- Python: <repo>/.venv/bin/python (editable install of this worktree). Run tests as `$PY -m pytest -q -n 8 -p no:cacheprovider`.
- Spec (finding IDs H1-H10, M1-M13): notes/PROJECT_REVIEW_2026-10.md
- Commit messages end with: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
- Never use bare `git stash`. Never push.
