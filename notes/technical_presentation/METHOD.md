# Method used to maintain the technical presentation

This document defines how claims, status labels, examples, figures, and
validation language enter the JaxCont v0.4.0 deck. It is a maintenance policy,
not an additional copy of the slide narrative.

## Evidence hierarchy

Use evidence in this order:

```text
current public API and implementation
→ tests and runnable examples
→ validation artifacts and independent references
→ current docs/changelog/roadmap
→ approved specs/plans for rationale only
```

The levels have different jobs:

1. The current public API and implementation establish what exists, its
   signature, returned data, transformation boundaries, and numerical scope.
2. Tests and runnable examples establish intended behavior and provide exact
   commands, diagnostics, and outputs that can be reproduced.
3. Reviewed validation artifacts and independent analytic, shooting, MatCont,
   or BifurcationKit references support accuracy or agreement claims within
   their stated conventions and tolerances.
4. Current user documentation, `CHANGELOG.md`, and `notes/ROADMAP.md` supply
   public context and release history. Historical status snapshots inside a
   long-lived document do not override current source or release metadata.
5. Approved specifications, plans, and Git history may explain why a design
   choice was made. They are not evidence that a planned capability exists.

When levels disagree, investigate the higher-priority evidence and state the
discrepancy. Do not silently combine behavior from different revisions.

## Version and feature-status rule

“JaxCont v0.4.0” is the deck label and identifies the release capability surface.

Before finalizing the deck, check `src/jaxcont/_version.py` and `CHANGELOG.md`,
then classify each feature with exactly one convention:

- **Available in v0.4.0** — present in the v0.4.0 release;
- **Future work — not implemented** — a boundary or direction, not an
  available workflow.

An available API may additionally carry an **experimental in v0.4.0** maturity
qualifier when evidence is incomplete. This is not a third implementation
status: the capability exists, but its validation boundary must remain visible.
LPC/PD/NS detection currently has this qualifier because `MC-LC-002` fails.

At the current revision, PRC/dPRC (`prc_curve`, `branch_prc`, `dprc_curve`,
`plot_prc`, and Examples 13–14), fold/Hopf two-parameter continuation, Hopf
classification, direct CP/BT/GH/ZH/HH point refinement, phase planes,
periodic-orbit continuation, and the validation suite are v0.4.0 capabilities.
If a release changes, update the global label, badges,
documentation, and speaker notes together; do not promote status from a plan
or anticipated release date.

Always preserve the explicit future boundary: no branch switching, automatic
cycle discovery from a Hopf point, general BVP continuation,
connecting-/homoclinic-/heteroclinic-orbit continuation, or periodic-orbit
codimension-two curve continuation is claimed.

## Source ownership

The source is intentionally modular:

- `jaxcont_technical_presentation.tex` owns document metadata, title, chapter
  order, and the final document boundary.
- `presentation_setup.tex` owns packages, Beamer/TikZ/listing conventions,
  shared colors, status badges, and chapter-divider components.
- `chapters/01_orientation.tex` through
  `chapters/13_scope_and_appendix.tex` own section declarations and frames.
- `assets/` owns reviewed deck-specific snapshots; `examples/images/` may be
  referenced directly for a reusable gallery figure.
- `README.md`, this file, and `SPEAKER_NOTES.md` own build/navigation,
  maintenance policy, and delivery guidance respectively.

The final public artifact remains
`notes/technical_presentation/jaxcont_technical_presentation.pdf`.

## Chapter construction pattern

Feature chapters use this sequence:

```text
scientific question
→ representative visual
→ mathematical or numerical method
→ minimum useful public API
→ concrete output
→ interpretation
→ limitation or common misuse
```

Not every chapter needs a separate frame for every arrow, but the order should
remain visible. Begin with what the audience is trying to learn and a picture
of the object or change. Introduce equations after the geometry or scientific
motivation. Show only the API needed to perform the task. End by teaching how
to read the result and what it does not establish.

Each normal frame has one primary teaching message. Put dense derivations,
carry fields, traced/eager reassembly, and similar maintainer detail in the
Chapter 13 appendix. Use descriptive frame/chapter names for cross-references;
page numbers are revision-specific navigation aids.

## Terminology and interpretation conventions

Keep these distinctions explicit throughout the deck:

- continuation follows connected roots; simulation follows a trajectory in
  physical time; a root solve finds one isolated solution;
- pseudo-arclength is the numerical method, predictor–corrector is the repeated
  pattern, and the “scan engine” is JaxCont’s bounded whole-branch functional
  implementation;
- `pseudo_arclength_scan` and `natural_scan` use `jax.lax.while_loop`; “scan”
  does not claim that the outer sweep calls `jax.lax.scan`;
- equilibrium stability is determined by eigenvalue real parts, whereas cycle
  stability is determined by nontrivial Floquet-multiplier magnitudes after
  removing exactly one multiplier nearest +1;
- detecting an event candidate is not the same as a converged local
  refinement;
- Hopf `l1` is a local, scale- and tolerance-aware normal-form quantity, not a
  global periodic-branch prediction;
- direct codimension-two solvers refine one supplied two-parameter guess; they
  do not themselves continue a curve; the separate seeded fold/Hopf curve
  factories continue connected codimension-one event curves;
- JaxCont’s dPRC is `d(PRC)/dp` after reconverging the periodic orbit, not the
  time derivative exported under the dPRC name by MatCont;
- software tests support implementation behavior, not model truth, causality,
  or the scientific interpretation of a computed branch.

## Figure and demo provenance

Use TikZ for editable teaching geometry and repository outputs for application
claims. Deck-specific snapshots under `assets/` must be regenerated from the
exact command in `assets/README.md`, visually reviewed, copied without hand
editing, and recorded with source path, source revision/date, intended chapter,
and SHA-256.

A directly reused gallery image is a separate provenance class. It must stay
tracked, carry an explicit origin revision and hash in the presentation README,
be visually inspected, and have its source example rerun. When the source only
displays figures, that run verifies current behavior but must not be described
as byte-for-byte regeneration. Example 12 is the current direct-gallery
exception. Replacing it requires either a deterministic save/export workflow
with recorded command/revision/hash or a new reviewed snapshot under `assets/`.

When an example changes, rerun it before editing the slide. Check its printed
diagnostics and any generated image; a current filename does not prove current
content. For a display-only direct-gallery source, retain and hash-check the
reviewed input until a reproducible export exists. If a slide prints a
numerical value, copy it from a fresh run or a reviewed validation artifact and
identify whether it is a diagnostic, an analytic answer, or a result judged
against a declared tolerance.

For a live demo, keep a reviewed image and captured expected output available.
Commands that open interactive windows, simulate a long transient, compile a
new JAX shape, or perform independent shooting are better demonstrated from
pre-generated output unless they have been rehearsed on the presentation
machine.

## Validation standard

Present validation as a ladder:

1. inspect convergence, re-evaluated residuals, finite values, validity masks,
   endpoint coverage, and configured stop conditions;
2. repeat with changed mesh, tolerance, step bounds, and direction;
3. compare with an analytic oracle where one exists;
4. compare with an independent algorithm;
5. compare with an independent package or reviewed reference artifact.

Higher rungs supplement rather than replace lower ones. Before numerical
comparison, align state ordering, parameter units, branch orientation,
interpolation grid, phase origin, phase units, eigenvalue/multiplier matching,
and trivial-multiplier removal. Document the alignment rule before applying it.

Do not turn a visual overlap into a pass claim. A pass/fail statement requires
a declared metric and tolerance. Preserve partial results and known failures;
do not relax a tolerance to make a comparison pass. For dPRC, use analytic or
reconverged finite-difference/shooting evidence rather than a misleading
MatCont time-derivative comparison.

## Maintenance verification

For every substantive change:

1. check displayed imports, names, signatures, return fields, and status
   against current source;
2. run every example whose output or numerical claim changed;
3. update snapshot provenance and hashes when an image changed;
4. build the complete deck and scan the log for errors, missing files,
   undefined references, and overfull frames;
5. extract PDF text to confirm chapter/status wording;
6. render and inspect the affected pages at presentation scale;
7. run `make verify` before final handoff;
8. synchronize `README.md`, this method, and `SPEAKER_NOTES.md` whenever
   structure, status, routes, commands, or pagination changes.

Record commit, precision, numerical settings, dependency versions, and target
hardware when producing evidence for external publication.
