# Phase 3 Release Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase-3 "release integrity" findings from the 2026-08-19 project review: documentation/roadmap/changelog contradicting shipped capabilities and versions, a source version indistinguishable from the already-published `v0.3.1` tag, declared Python/dependency compatibility far broader than what CI actually tests, no release-gating slow-validation CI job, and release-publishing actions pinned to mutable tags instead of commit SHAs. **This plan does not cut or publish a release** — `publish.yml` is `workflow_dispatch`-only and nothing here triggers it; the version bump (Task 4) sets a development version, not a new release tag.

**Architecture:** No new modules or source-code behavior changes. This is entirely documentation, packaging metadata, and CI configuration: reconciling five doc surfaces (root `CHANGELOG.md`, `docs/source/changelog.rst`, `docs/source/roadmap.rst`, `docs/source/index.rst`, `README.md`) against the codebase's actual capabilities and `git tag` history; bumping `src/jaxcont/_version.py` past the published `v0.3.1` tag; widening and hardening `.github/workflows/tests.yml`'s Python/dependency coverage; and pinning `.github/workflows/publish.yml`'s third-party actions to commit SHAs.

**Tech Stack:** Markdown, reStructuredText, MyST (Sphinx's Markdown parser, already enabled via `myst_parser` in `docs/source/conf.py`), TOML (`pyproject.toml`), GitHub Actions YAML, `python -m build` / `twine`.

**Spec:** `notes/PROJECT_REVIEW_2026-08.md` (findings #4, #8, #9, #11 and their "Recommended fix" sections; "Phase 3: Release integrity" in "Recommended implementation order").

## Global Constraints

- **No release/publish action.** Do not run `git tag`, push a tag, or trigger `.github/workflows/publish.yml` (it only runs on manual `workflow_dispatch` — do not dispatch it). Do not remove or weaken that manual-trigger gate.
- Every doc fix must be traceable to the actual current codebase state, not to what an older doc claimed — when in doubt, check `git tag`, `CHANGELOG.md`, and what `src/jaxcont/__init__.py` actually exports, rather than trusting another stale doc as a reference.
- Run tests with:
  ```
  PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
    /home/ziaee/envs/jaxcont/bin/python -m pytest -n auto -m ''
  ```
  From inside a git worktree, `PYTHONPATH` **must** point at that worktree's own `src/` — the venv's editable install otherwise silently resolves `import jaxcont` to the main checkout's code. Most tasks in this plan don't touch `src/`, but always run the full suite once per task regardless, to catch any accidental breakage.
- GitHub Actions `uses:` pins in this plan use commit SHAs fetched live on 2026-08-20 via `gh api repos/<owner>/<repo>/git/refs/tags/<tag>` (dereferencing annotated tags to their commit via `gh api repos/<owner>/<repo>/git/tags/<tag-sha>` where needed). Use the exact SHAs given in Task 6 — do not re-derive or guess them.

---

## Task 1: Reconcile the two changelogs

**Files:**
- Modify: `CHANGELOG.md`
- Create: `docs/source/changelog.md`
- Delete: `docs/source/changelog.rst`
- Modify: any file referencing `changelog.rst` in a Sphinx toctree (search first, see Step 1)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: from this task onward, `docs/source/changelog.md` is a thin MyST `{include}` of the root `CHANGELOG.md` — later tasks that touch `CHANGELOG.md` (none currently planned, but future changes) automatically stay in sync with the Sphinx-rendered changelog with no further doc-sync task ever needed again.

Two problems: (1) root `CHANGELOG.md`'s `## [0.1.0]` entry is labeled `- Unreleased` even though `v0.1.0` was tagged on 2026-07-19 (confirmed via `git log -1 --format=%ai v0.1.0`) and three releases have shipped since. (2) `docs/source/changelog.rst` is a second, hand-maintained changelog that only has entries up to `[0.1.0] - Unreleased` and `[0.0.1]` — it has never been updated for `v0.2.0`, `v0.3.0`, or `v0.3.1`, so the published Sphinx docs' changelog page is two major releases behind the real one.

- [ ] **Step 1: Find what references `changelog.rst`**

Run: `grep -rn "changelog" docs/source/*.rst docs/source/conf.py`

Expected: at least one toctree entry (likely in `docs/source/index.rst`) reading something like `changelog` (Sphinx toctree entries are extension-less, so a rename from `.rst` to `.md` needs no toctree edit — but confirm this by checking the exact line found). Note the file(s) and line(s) for Step 4.

- [ ] **Step 2: Fix the stale `[0.1.0]` date in `CHANGELOG.md`**

Change:
```markdown
## [0.1.0] - Unreleased
```
to:
```markdown
## [0.1.0] - 2026-07-19
```

- [ ] **Step 3: Replace `docs/source/changelog.rst` with a MyST include**

Delete `docs/source/changelog.rst`. Create `docs/source/changelog.md`:

```markdown
# Changelog

```{include} ../../CHANGELOG.md
:start-line: 2
```
```

`:start-line: 2` skips root `CHANGELOG.md`'s own `# Changelog` line (line 1) and the blank line after it (line 2), so the included content starts at "All notable changes..." — avoiding two stacked `# Changelog` headings on the rendered page. Verify this line-skip is still correct against the CURRENT root `CHANGELOG.md` (re-open it and confirm line 1 is `# Changelog` and line 2 is blank before committing to `:start-line: 2` — adjust the number if the file's current head differs).

- [ ] **Step 4: Confirm the toctree still resolves**

If Step 1 found a toctree entry like `changelog` (no extension), no edit is needed — Sphinx's `source_suffix`/MyST config (already present in `docs/source/conf.py`, confirmed via `grep -n "myst_parser" docs/source/conf.py`) resolves `changelog` to `changelog.md` now that `changelog.rst` no longer exists. If the toctree entry explicitly said `changelog.rst`, change it to `changelog.md` (or extension-less `changelog`, matching the style of neighboring toctree entries in the same file).

- [ ] **Step 5: Build the docs and verify the changelog page renders with real content**

Run:
```bash
cd docs
/home/ziaee/envs/jaxcont/bin/python -m sphinx -b html source build/html-phase3-check -W --keep-going 2>&1 | tail -60
```

(`-W` turns warnings into errors, matching the project's `fail_on_warning` CI convention mentioned elsewhere in this codebase; `--keep-going` surfaces every warning instead of stopping at the first.) Expected: build succeeds (exit code 0) with no new warnings attributable to this change. Then check the rendered page: `grep -c "0.3.1" docs/build/html-phase3-check/changelog.html` should be `>= 1` (confirms the real, current changelog content made it into the build, not just the old `0.1.0`/`0.0.1` stub). Remove the `docs/build/html-phase3-check` directory afterward (`rm -rf docs/build/html-phase3-check`) — it's a throwaway verification build, not a committed artifact (confirm `docs/build/` is gitignored: `git check-ignore -q docs/build` should exit 0).

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md docs/source/changelog.md
git rm docs/source/changelog.rst
git add -u  # in case Step 4 touched a toctree file
git commit -m "docs: fix stale v0.1.0 changelog date, make Sphinx changelog page include the real CHANGELOG.md"
```

---

## Task 2: Reconcile README.md and docs/source/index.rst capability/citation claims

**Files:**
- Modify: `README.md`
- Modify: `docs/source/index.rst`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing consumed by other tasks (independent of Tasks 1, 3-6).

Both files' capability summaries still say two-parameter curve continuation is future work, even though `fold_curve_problem`/`hopf_curve_problem` (with their own codimension-two event detection) have been implemented, tested, and exported from `jaxcont` for some time (confirmed via `grep -n "fold_curve_problem\|hopf_curve_problem" src/jaxcont/__init__.py`). Separately, `README.md`'s citation section says "Until the first archive is minted" and gives a bibtex block with `version = {0.3.0}` and no `doi` field — but `CITATION.cff` already has two real, live Zenodo DOIs (`10.5281/zenodo.21812716` concept DOI, `10.5281/zenodo.21812717` for the archived `v0.3.1`), and the README's own DOI badge at the top of the file already links to one of them. The bibtex is stale and the framing is self-contradictory.

- [ ] **Step 1: Fix README.md's capability paragraph**

Change:
```markdown
The current surface supports equilibrium and periodic-orbit continuation,
their principal codimension-one events, direct codimension-two point
solvers, and infinitesimal phase response curves (iPRC, `prc_curve`,
cross-validated against MatCont) plus their parameter-derivative sensitivity
(dPRC, `dprc_curve`, validated independently -- MatCont does not compute
this quantity). Branch switching, general connecting-orbit/BVP workflows,
and two-parameter curve continuation remain future work.
```
to:
```markdown
The current surface supports equilibrium and periodic-orbit continuation,
their principal codimension-one events, direct codimension-two point
solvers, two-parameter continuation (fold/Hopf curves with their own
codimension-two events), and infinitesimal phase response curves (iPRC,
`prc_curve`, cross-validated against MatCont) plus their parameter-derivative
sensitivity (dPRC, `dprc_curve`, validated independently -- MatCont does not
compute this quantity). Branch switching and general connecting-orbit/BVP
workflows remain future work.
```

- [ ] **Step 2: Fix README.md's citation section**

Change:
```markdown
## Citation

If JaxCont supports your research, cite the archived release using its
GitHub/Zenodo DOI. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
Until the first archive is minted:

```bibtex
@software{ziaeemehr_jaxcont_2026,
  author  = {Ziaeemehr, Abolfazl},
  title   = {JaxCont: Differentiable Continuation and Bifurcation Analysis in JAX},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/Ziaeemehr/JaxCont}
}
```
```
to:
```markdown
## Citation

If JaxCont supports your research, please cite the archived release. Full
citation metadata (including both DOIs) is in [`CITATION.cff`](CITATION.cff) --
GitHub renders a "Cite this repository" button from it automatically.

```bibtex
@software{ziaeemehr_jaxcont_2026,
  author  = {Ziaeemehr, Abolfazl},
  title   = {JaxCont: Differentiable Continuation and Bifurcation Analysis in JAX},
  year    = {2026},
  version = {0.3.1},
  doi     = {10.5281/zenodo.21812717},
  url     = {https://github.com/Ziaeemehr/JaxCont}
}
```
```

Before committing to `version = {0.3.1}` / `doi = {10.5281/zenodo.21812717}`, re-check `CITATION.cff`'s current `version:` and the versioned-DOI `identifiers` entry (`cat CITATION.cff`) — use whatever it currently says verbatim, in case it has been updated since this plan was written.

- [ ] **Step 3: Fix docs/source/index.rst's capability sentence**

Change:
```rst
JaxCont is a continuation and bifurcation-analysis library whose default
pseudo-arclength engine runs the whole continuation loop as a compiled JAX
computation. It supports equilibrium and periodic-orbit continuation,
principal codimension-one events, direct codimension-two point solvers, and
phase response curves: infinitesimal PRC (``prc_curve``, cross-validated
against MatCont) and its parameter-derivative sensitivity (``dprc_curve``,
validated independently of MatCont, which does not compute this quantity).
Branch switching, continuation of two-parameter curves, general
boundary-value problems, and connecting orbits remain unsupported.
```
to:
```rst
JaxCont is a continuation and bifurcation-analysis library whose default
pseudo-arclength engine runs the whole continuation loop as a compiled JAX
computation. It supports equilibrium and periodic-orbit continuation,
principal codimension-one events, direct codimension-two point solvers,
two-parameter continuation (fold/Hopf curves with their own
codimension-two events), and phase response curves: infinitesimal PRC
(``prc_curve``, cross-validated against MatCont) and its
parameter-derivative sensitivity (``dprc_curve``, validated independently
of MatCont, which does not compute this quantity). Branch switching,
general boundary-value problems, and connecting orbits remain unsupported.
```

- [ ] **Step 4: Verify no other stale capability claim was missed**

Run: `grep -n "two-parameter" README.md docs/source/index.rst`

Expected: every remaining match describes two-parameter continuation as supported (not as future/unsupported work). If any match still says otherwise, fix it the same way as Steps 1/3.

- [ ] **Step 5: Build the docs**

Run the same build command as Task 1 Step 5 (a fresh throwaway directory, e.g. `docs/build/html-phase3-check-task2`, removed afterward): confirm it succeeds with `-W --keep-going` and that `docs/source/index.rst`'s edit didn't introduce an RST syntax error (a broken `::` or indentation would show up as a build warning/error here).

- [ ] **Step 6: Commit**

```bash
git add README.md docs/source/index.rst
git commit -m "docs: fix stale two-parameter-continuation and citation claims in README and index"
```

---

## Task 3: Rewrite the stale roadmap

**Files:**
- Modify: `docs/source/roadmap.rst`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing consumed by other tasks.

`docs/source/roadmap.rst` says "Current Version: 0.1.0" and marks Hopf normal-form classification, all five codim-2 solvers, periodic-orbit continuation, Floquet multipliers, period-doubling/Neimark-Sacker detection, and two-parameter continuation as still "🚧 In Progress" or "⬜" planned, even though every one of them shipped in `v0.2.0`/`v0.3.0`. "Version History" only lists `v0.1.0`.

- [ ] **Step 1: Replace the version header**

Change:
```rst
Current Version: 0.1.0
----------------------

**Status**: Alpha - Core framework implemented, under active development
```
to:
```rst
Current Version: 0.3.1
-----------------------

**Status**: Alpha/Beta -- broad functional coverage (equilibrium and
periodic-orbit continuation, codimension-one and codimension-two events,
two-parameter continuation, phase response curves) with extensive
analytic, MatCont, and BifurcationKit cross-validation. Under active
hardening before a stable 1.0 release.
```

- [ ] **Step 2: Replace "Phase 1" through "Phase 4" with one "Completed" section**

Change everything from `Phase 1: Core Functionality (v0.1.x)` (currently starting right after the header) through the end of `Phase 4: Two-Parameter Continuation (v0.4.0)` (i.e. delete all of "Phase 1", "Phase 2", "Phase 3", and "Phase 4" and their subsections/checklists) and replace that whole block with:

```rst
Completed (through v0.3.1)
---------------------------

- Natural and pseudo-arclength continuation, both as fully JIT-compiled,
  ``vmap``-safe whole-loop engines
- Newton solver with JAX autodiff; adaptive step-size control
- Equilibrium and periodic-orbit (limit-cycle) problem definitions, the
  latter via Gauss-Legendre orthogonal collocation
- Codimension-one event detection and refinement: fold, Hopf,
  period-doubling, Neimark-Sacker
- Floquet multiplier computation via the collocation monodromy matrix
- Hopf normal-form classification (first Lyapunov coefficient,
  criticality) and five direct codimension-two point solvers (cusp,
  Bogdanov-Takens, generalized Hopf, zero-Hopf, double Hopf), all
  differentiable via the implicit function theorem
- Two-parameter continuation: fold-curve and Hopf-curve problems with
  their own codimension-two event detection
- Infinitesimal phase response curves (iPRC) and their
  parameter-derivative sensitivity (dPRC)
- 2D phase-plane visualization and stability-aware bifurcation-diagram
  plotting
- Differentiable bifurcation locations (``jax.grad``/``jax.jacfwd``
  through fold/Hopf solvers) for inverse design
- Batched continuation sweeps with ``jax.vmap``
- Analytic, MatCont 7.6, and BifurcationKit cross-validation suites
```

Everything from `Phase 5: Advanced Features (v0.5.0)` onward (through the end of the file) is unaffected by this step -- do not touch it here.

- [ ] **Step 3: Retitle the remaining forward-looking phases, dropping invented target dates**

The existing `Phase 5: Advanced Features (v0.5.0)` section (branch switching, homoclinic/heteroclinic orbits, invariant tori, etc.) and `Phase 6: Performance & Polish (v0.6.0)` section (GPU optimization, sparse matrices, etc.) both currently carry a `**Target**: Q_ 20XX` line. These dates were never accurate (the whole file is being rewritten because its dates/status drifted from reality) and re-inventing new ones would repeat the same mistake. In each of the two sections:

- Change the heading `Phase 5: Advanced Features (v0.5.0)` to `Next: Advanced Features` (drop the version number -- these features aren't scoped to a specific next version yet).
- Change the heading `Phase 6: Performance & Polish (v0.6.0)` to `Later: Performance & Polish`.
- Delete each section's `**Target**: Q_ 20XX` line entirely (do not replace it with a new date).
- Leave every bullet under both headings unchanged -- these items (branch switching, homoclinic orbits, GPU optimization, etc.) are still genuinely unimplemented, matching README.md's and index.rst's "remain future work" language from Task 2.

Leave the `Version 1.0: Production Ready` section's heading and bullets unchanged, but delete its `**Target**: Q2 2027` line the same way, for the same reason.

- [ ] **Step 4: Update "Version History"**

Change:
```rst
Version History
---------------

**v0.1.0** (November 2025)
   - Initial release
   - Core continuation framework
   - Basic bifurcation detection
   - Example gallery
```
to:
```rst
Version History
---------------

**v0.3.1** (August 2026)
   - Floquet near-unit-circle detection margin widened (bugfix)
   - Read the Docs build fix

**v0.3.0** (August 2026)
   - Hopf normal-form classification and five direct codimension-two solvers
   - 2D phase-plane visualization

**v0.2.0** (July 2026)
   - First PyPI release
   - Periodic-orbit continuation, Floquet multipliers, period-doubling/
     Neimark-Sacker detection

**v0.1.0** (July 2026)
   - Initial release
   - Core continuation framework
   - Basic bifurcation detection
   - Example gallery
```

Before committing to the month labels above, cross-check them against `git log -1 --format=%ai v0.1.0`, `v0.2.0`, `v0.3.0`, `v0.3.1` (already confirmed for v0.1.0: 2026-07-19; v0.2.0: 2026-07-24 -- confirm v0.3.0 and v0.3.1 the same way) and adjust the month names if any differ from what's written above.

- [ ] **Step 5: Build the docs**

Same as Task 1 Step 5 / Task 2 Step 5 (throwaway build directory, `-W --keep-going`, remove afterward). RST section-underline length mismatches (e.g. `-----------------------` under `Current Version: 0.3.1` must be at least as long as the heading text) are a common error here -- watch for `WARNING: Title underline too short` in the build output and fix any that appear.

- [ ] **Step 6: Commit**

```bash
git add docs/source/roadmap.rst
git commit -m "docs: rewrite roadmap to reflect actual v0.3.1 capabilities instead of stale v0.1.0 status"
```

---

## Task 4: Bump the development version and validate packaging metadata

**Files:**
- Modify: `src/jaxcont/_version.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing consumed by other tasks. `CITATION.cff` and `README.md`'s bibtex (Task 2) intentionally keep citing `0.3.1` -- that's the last *archived* (Zenodo-DOI'd) release, and stays the citable version until a new one is actually cut and archived. This task's version bump is a development-build marker, not a new citable release; do not change `CITATION.cff` or the README bibtex version to match it.

`src/jaxcont/_version.py` still reads `"0.3.1"`, byte-identical to the already-published `v0.3.1` PyPI release. A wheel built from current `main` is indistinguishable by version from the archived release, and PyPI would reject re-uploading `0.3.1` even if it were otherwise ready.

- [ ] **Step 1: Bump the version**

Change:
```python
__version__ = "0.3.1"
```
to:
```python
__version__ = "0.4.0.dev0"
```

(`0.4.0.dev0` is a valid PEP 440 development release identifier -- installs and orders correctly as "before 0.4.0" via `pip`/`packaging`, and reads unambiguously as "not yet released.")

- [ ] **Step 2: Build the distribution**

Run:
```bash
cd <repo-root>
python -m pip install --upgrade build twine
python -m build
```

Expected: `dist/jaxcont-0.4.0.dev0-py3-none-any.whl` (or similar wheel filename stamped `0.4.0.dev0`) and `dist/jaxcont-0.4.0.dev0.tar.gz` are created. If the build fails, do not proceed -- report BLOCKED with the build's error output.

- [ ] **Step 3: Validate the distribution**

Run: `python -m twine check dist/*`

Expected: `PASSED` for both the wheel and the sdist.

- [ ] **Step 4: Confirm the version is actually stamped correctly**

Run:
```bash
wheel_file=$(ls dist/*.whl)
unzip -p "$wheel_file" '*/METADATA' | grep -i "^Version:"
```

Do not hardcode the wheel's exact filename (the version-tag/build-tag portion can vary) -- always discover it via the `dist/*.whl` glob as shown. Expected output: `Version: 0.4.0.dev0`. If it reads anything else, do not proceed -- report BLOCKED.

- [ ] **Step 5: Clean up build artifacts**

First confirm each path is actually gitignored, checked individually (a combined multi-path `git check-ignore` call can exit 0 if only *some* of the paths are ignored, which isn't a strong enough guarantee here):
```bash
for p in dist build src/jaxcont.egg-info; do
  git check-ignore -q "$p" && echo "$p: ignored" || echo "$p: NOT ignored"
done
```
If any path prints "NOT ignored", stop and report BLOCKED rather than deleting it. If all three print "ignored", remove them: `rm -rf dist/ build/ src/jaxcont.egg-info/`

- [ ] **Step 6: Run the full test suite**

Run the full-suite command from Global Constraints. A version-string-only change should not affect any test outcome; confirm the pass/skip/xfail counts match the pre-task baseline exactly.

- [ ] **Step 7: Commit**

```bash
git add src/jaxcont/_version.py
git commit -m "chore: bump development version past the published v0.3.1 release"
```

---

## Task 5: Widen the tested Python/dependency matrix

**Files:**
- Modify: `.github/workflows/tests.yml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing consumed by other tasks.

`pyproject.toml` declares `requires-python = ">=3.9"` and classifiers for Python 3.9-3.12, and dependency floors `jax>=0.3.0`/`jaxlib>=0.3.0`/`numpy>=1.21.0`, but `.github/workflows/tests.yml` only runs Python 3.11 with whatever dependency versions an unconstrained `pip install` happens to resolve today -- so none of the declared floors, and only one of the four declared Python versions, are actually verified by CI. The code imports `jax.Array` directly (`src/jaxcont/api.py`, `src/jaxcont/core/scan_continuation.py`), a type that did not exist as public API in JAX 0.3.0, making the declared JAX floor doubtful without ever having been tested.

- [ ] **Step 1: Widen the CI Python matrix**

In `.github/workflows/tests.yml`, change:
```yaml
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11"]
```
to:
```yaml
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
```

This doesn't require local multi-Python verification -- `actions/setup-python` fetches each interpreter on GitHub's runners, so pushing this change is itself the verification: the next CI run on this branch will report which of 3.9/3.10/3.11/3.12 actually pass. Note in your task report which versions you were able to observe pass/fail (check the Actions run for this branch/PR after pushing, if your access allows it in this environment; if not, note that verification depends on the CI run this push triggers, and say so explicitly -- do not claim a Python version "passes" without having seen it actually run).

- [ ] **Step 2: Empirically determine a real JAX/NumPy dependency floor**

The declared `jax>=0.3.0`/`jaxlib>=0.3.0`/`numpy>=1.21.0` floors predate `jax.Array` becoming public API and have never been installed, let alone tested, in this project. Determine a real floor with a scratch venv:

```bash
python3.12 -m venv /tmp/jaxcont-floor-check
/tmp/jaxcont-floor-check/bin/pip install --upgrade pip
# Try a candidate floor -- jax.Array was stabilized as public API by jax 0.4.x;
# start there and work outward based on what actually installs and imports.
/tmp/jaxcont-floor-check/bin/pip install "jax==0.4.13" "jaxlib==0.4.13" "numpy>=1.22,<2" "scipy" "matplotlib" "equinox>=0.11.0"
/tmp/jaxcont-floor-check/bin/pip install -e . --no-deps
/tmp/jaxcont-floor-check/bin/python -c "import jaxcont; from jax import Array; print(jaxcont.__version__, 'import OK')"
```

If that import succeeds, run a fast smoke subset of the real test suite against this environment (not the full suite -- this scratch venv likely lacks `pytest-xdist`/dev extras; install what's needed):

```bash
/tmp/jaxcont-floor-check/bin/pip install pytest
PYTHONPATH=<repo-root>/src JAX_PLATFORMS=cpu /tmp/jaxcont-floor-check/bin/python -m pytest tests/test_functional_api.py tests/test_bifurcations.py -q -m ''
```

If this passes, that JAX/NumPy pair is a *candidate* floor -- optionally try one version older (e.g. `jax==0.4.7`) to see if the floor can go lower, but stop as soon as either the import or this smoke subset fails, and use the last version that worked as the floor. Delete the scratch venv when done (`rm -rf /tmp/jaxcont-floor-check`).

- [ ] **Step 3: Update pyproject.toml with the empirically-determined floor**

Update the `dependencies` list's `jax>=...`/`jaxlib>=...`/`numpy>=...` entries to the versions Step 2 actually confirmed work (not a guess). If Step 2's first candidate (`0.4.13`) failed outright (didn't install or `jax.Array` import failed), try progressively newer versions (`0.4.20`, `0.4.30`, ...) until one succeeds, and use that as the floor instead -- report in your task report exactly which version was tried and why it was chosen. Leave `scipy>=1.7.0`, `matplotlib>=3.5.0`, and `equinox>=0.11.0` unchanged unless Step 2's install step itself failed to resolve with those floors (in which case, note it and use whatever minimal bump was needed to make the install resolve).

- [ ] **Step 4: Run the full suite in the main venv**

Run the full-suite command from Global Constraints against the normal project venv (not the scratch floor-check venv) to confirm this task's `pyproject.toml`/CI edits didn't break anything in the environment actually used for development. Pass/skip/xfail counts should match the pre-task baseline exactly (this task doesn't change any source behavior).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/tests.yml pyproject.toml
git commit -m "ci: test the full declared Python matrix, set an empirically-verified dependency floor"
```

---

## Task 6: Release-gating slow validation + pin publish actions to commit SHAs

**Files:**
- Modify: `.github/workflows/tests.yml`
- Modify: `.github/workflows/publish.yml`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing consumed by other tasks.

`pyproject.toml`'s `addopts = "-q -m 'not slow and not gpu'"` means every default `pytest` invocation -- including `tests.yml`'s CI job -- silently excludes `@pytest.mark.slow` tests. There is currently no CI job anywhere that ever runs the slow suite, so a regression in it (e.g. the strict MatCont validation cases) could ship undetected. Separately, `.github/workflows/publish.yml` invokes `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`, `actions/download-artifact@v4`, and `pypa/gh-action-pypi-publish@release/v1` by mutable tag/branch name, while that job holds `id-token: write` (OIDC) permission to publish to PyPI -- a compromised or re-pointed tag on any of those actions would run with that privilege.

- [ ] **Step 1: Add a scheduled + release-gating slow-validation job**

In `.github/workflows/tests.yml`, add a second job alongside the existing `test` job (same file, top-level `jobs:` key gains a sibling). Also widen the workflow's `on:` triggers to include a weekly schedule and tag pushes, since the new job should run on a release-tag push (release-gating) in addition to its own schedule. Change:

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

jobs:
  test:
```

to:

```yaml
on:
  push:
    branches: [ main, develop ]
    tags: [ "v*" ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # Weekly, Monday 06:00 UTC -- catches slow-suite regressions (including
    # the MatCont validation cases) that the fast default CI job excludes.
    - cron: "0 6 * * 1"
  workflow_dispatch:

jobs:
  test:
```

Then add this new job after the existing `test` job's final step (append at the end of the file, same indentation level as `test:` under `jobs:`):

```yaml

  slow-validation:
    # Runs the full suite including @pytest.mark.slow tests (e.g. the
    # strict MatCont validation cases) that the default `pytest` invocation
    # above excludes via pyproject.toml's addopts. Triggered weekly and on
    # every version-tag push, so a slow-suite regression can't ship silently
    # between the rare occasions someone runs `make test-all` locally.
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' || startsWith(github.ref, 'refs/tags/v')
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run full suite including slow tests
      run: |
        pytest tests/ -v -m '' --cov=jaxcont --cov-report=xml --cov-report=term-missing
```

- [ ] **Step 2: Validate the YAML**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))" && echo "tests.yml: valid YAML"`

Expected: `tests.yml: valid YAML` with no exception. (This validates syntax, not GitHub Actions semantics -- there is no local way to trigger a real scheduled/tag-push run from this environment; note in your task report that semantic verification depends on this actually running in GitHub Actions once pushed.)

- [ ] **Step 3: Pin publish.yml's actions to commit SHAs**

In `.github/workflows/publish.yml`, change each of these five lines (matching by the `uses:` value, wherever they appear in the file) from a mutable tag to the exact commit SHA below, with the original tag preserved as a trailing comment (standard GitHub-recommended pinning style, so the intended version stays human-readable):

| Current | New |
|---|---|
| `uses: actions/checkout@v4` | `uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2` |
| `uses: actions/setup-python@v5` | `uses: actions/setup-python@0b93645e9fea7318ecaed2b359559ac225c90a2b # v5.3.0` |
| `uses: actions/upload-artifact@v4` | `uses: actions/upload-artifact@b4b15b8c7c6ac21ea08fcf65892d2ee8f75cf882 # v4.4.3` |
| `uses: actions/download-artifact@v4` | `uses: actions/download-artifact@fa0a91b85d4f404e444e00e005971372dc801d16 # v4.1.8` |
| `uses: pypa/gh-action-pypi-publish@release/v1` (appears twice) | `uses: pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33 # v1.14.2` |

Use these exact SHAs verbatim (fetched live via `gh api` on 2026-08-20 and cross-checked: `dc37677b...` is both the tip of `pypa/gh-action-pypi-publish`'s `release/v1` branch and the commit its `v1.14.2` annotated tag dereferences to). Do not re-derive or substitute different SHAs.

`pypa/gh-action-pypi-publish@release/v1` appears twice in this file (once for the TestPyPI step, once for the PyPI step) -- pin both occurrences.

- [ ] **Step 4: Validate the YAML**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yml'))" && echo "publish.yml: valid YAML"`

Expected: `publish.yml: valid YAML`.

- [ ] **Step 5: Confirm publish.yml's manual-trigger gate is unchanged**

Run: `grep -n "workflow_dispatch" .github/workflows/publish.yml`

Expected: still present, and `on:` still has no `push`/`pull_request`/`schedule` trigger for this file -- this task must not add any new way for `publish.yml` to run automatically. If your edits accidentally changed the `on:` block, revert that part.

- [ ] **Step 6: Run the full suite**

Run the full-suite command from Global Constraints. A CI-YAML-only change should not affect any test outcome; confirm the pass/skip/xfail counts match the pre-task baseline exactly.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/tests.yml .github/workflows/publish.yml
git commit -m "ci: add release-gating slow-validation job, pin publish.yml actions to commit SHAs"
```
