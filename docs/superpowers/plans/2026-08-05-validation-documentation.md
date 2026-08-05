# Validation Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a short, honest summary of the MatCont validation results in the hosted documentation.

**Architecture:** Add one static MyST page to the existing Sphinx site and link it from the root toctree. Keep numerical execution outside the documentation build; the page reports the reviewed snapshot and links to the complete suite.

**Tech Stack:** Sphinx, MyST Markdown, reStructuredText, JaxCont validation CLI

## Global Constraints

- Keep the public page concise: conclusion, seven-case table, known mismatch, provenance, and reproduction command.
- Report `MC-LC-002` as a failure; do not alter validation code or tolerances.
- Do not execute the numerical suite as part of the Sphinx build.
- Update the Sphinx landing page to describe the current feature surface.

---

### Task 1: Add the validation results page

**Files:**
- Create: `docs/source/validation.md`
- Modify: `docs/source/index.rst:7-42`

**Interfaces:**
- Consumes: statuses from `python3 -m examples.MatCont.run_validation` and methodology from `examples/MatCont/README.md`
- Produces: a `validation` Sphinx document reachable from the root toctree

- [x] **Step 1: Create the concise MyST page**

Write `docs/source/validation.md` with:

- A `# Validation against MatCont` title.
- A dated conclusion that six of seven supported cases pass.
- A table with `Case`, `What is checked`, and `Result` columns for all seven supported cases.
- One short `MC-LC-002` limitation paragraph reporting missing LPC/PD labels and a maximum critical-multiplier error of approximately `1.10e-2`.
- One methodology/provenance paragraph naming MATLAB R2020a, MatCont 7.6, normalized committed references, declared tolerances, and the comparison policy.
- The CPU reproduction command and a link to the complete suite README.

- [x] **Step 2: Link the page and refresh the landing-page feature summary**

Add `validation` after `auto_examples/index` in the "Using JaxCont" toctree in `docs/source/index.rst`. Replace the old v0.1 description with the current supported surface—equilibrium and periodic-orbit continuation, principal codimension-one events, and direct codimension-two point solvers—and retain a short statement of unsupported families.

- [x] **Step 3: Verify the documented numerical status**

Run:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl-jaxcont-validation python3 -m examples.MatCont.run_validation
```

Expected: exit status `1`; six `PASS` lines and one `FAIL MC-LC-002` line whose diagnostics include missing `LPC`/`PD` event types and `jaxcont_max_multiplier_error` approximately `0.0109874`.

- [x] **Step 4: Build documentation with warnings as errors**

Run:

```bash
make -C docs html SPHINXOPTS="-W --keep-going"
```

Expected: exit status `0`, no Sphinx warnings, and `docs/build/html/validation.html` exists.

- [x] **Step 5: Check formatting and commit**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the planned documentation files are changed.

Commit:

```bash
git add docs/source/validation.md docs/source/index.rst docs/superpowers/plans/2026-08-05-validation-documentation.md
git commit -m "docs: publish MatCont validation results"
```
