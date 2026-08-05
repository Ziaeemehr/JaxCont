# Validation Documentation Design

## Context

JaxCont has a reproducible validation suite in `examples/MatCont`, but the
hosted Sphinx documentation does not expose its results. The repository README
links to the suite, while the Sphinx landing page still describes the older
v0.1 feature surface and says periodic-orbit functionality is excluded.

The validation CLI currently reports six passing supported cases and one known
mismatch (`MC-LC-002`). The documentation must present that limitation plainly
rather than implying complete agreement with MatCont.

## Considered Approaches

1. **Dedicated static validation page (selected).** Publish a concise,
   human-readable snapshot with per-case status, provenance, methodology,
   limitations, and reproduction instructions. This is easy to find, honest
   about the failing case, and does not make documentation builds depend on a
   long numerical run.
2. **Embed raw CLI output.** This is exact but difficult to scan, exposes
   implementation-shaped diagnostics, and becomes stale without conveying the
   meaning of each case.
3. **Execute validation during every Sphinx build.** This keeps results fresh,
   but the suite is slow and intentionally returns a nonzero status for the
   known mismatch. It would make hosted documentation builds costly and
   fragile.

## Selected Design

Add `docs/source/validation.md` as a normal MyST documentation page, separate
from Sphinx-Gallery. Add it to the main "Using JaxCont" toctree so readers can
find validation alongside the quickstart and examples.

The page will contain:

- A direct conclusion: six of seven supported cases pass their declared
  tolerances in the current reviewed snapshot.
- A compact table listing every supported case, what it validates, and its
  status.
- A visible known-mismatch section for `MC-LC-002`, including the relevant
  current diagnostics: event-location errors remain small, but LPC/PD labels
  are missing and the critical Floquet-multiplier tolerance is exceeded.
- Reference provenance: MATLAB R2020a, MatCont 7.6, committed normalized
  CSV/JSON oracles, and the fact that comparisons use declared tolerances.
- A short methodology summary covering branch interpolation, unique event and
  spectrum matching, and treatment of the trivial Floquet multiplier.
- Commands and links needed to reproduce the validation or inspect the full
  suite documentation.
- A brief unsupported-capabilities summary that points to the complete matrix
  in `examples/MatCont/README.md` rather than duplicating the full table.

Update `docs/source/index.rst` so its feature summary matches the current
library surface: equilibrium and periodic-orbit continuation, principal
codimension-one events, and direct codimension-two point solvers. It will also
state the principal unsupported families without presenting roadmap promises
as implemented features.

## Data and Maintenance Policy

The public page is a reviewed static snapshot, not generated during Sphinx
builds. Results must be traceable to the committed `examples/MatCont/reference`
artifacts and the documented validation command. A future CI-generated report
may replace the hand-maintained status table, but report-generation machinery
is outside this change.

Known failures stay marked as failures. Documentation updates must not weaken
tolerances, reinterpret a failure as a pass, or edit validation code.

## Verification

Verification will include:

1. Re-run the validation CLI on CPU and confirm the documented status summary.
2. Build the Sphinx documentation with warnings treated as errors.
3. Check that the new page is reachable from the root toctree and that all
   repository links resolve in the generated documentation.
