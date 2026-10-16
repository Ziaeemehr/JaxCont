# Trace-safe Event Scanning — Design Spec

**Status:** Proposed; documentation only, against `61e0a6c`.
**Scope:** Phase 6.1 of [review remediation](../plans/2026-10-16-review-remediation.md), review H1–H3/H8/M9. Detection under `jit`/`vmap`; refinement stays eager.

## Problem

`src/jaxcont/api.py:852` rejects every traced call with events. The existing
`scan(points)` hook returns variable-length Python brackets
(`src/jaxcont/bifurcations/events.py:63`, `:514`); tracking uses SciPy assignment
and NumPy (`src/jaxcont/bifurcations/modes.py:19,35–54`). Scalar fallback calls
`float()` (`events.py:406`); codim-2 decoding and spectral scans also use host
loops (`src/jaxcont/bifurcations/codim2_events.py:100`). Merely removing the API
exception would neither detect events nor preserve mode identity.

## Chosen design and alternatives

Add a pure array detection path shared by eager built-ins and traced calls.
An `EventScanInput` pytree contains `params (B,)`, `states (B,d)`, optional
tangents/spectra/trivial indices, static strategy keys and an optional tuple
of per-effective-strategy `FloquetSpectrum` buffers, and `valid (B,)`, plus optional static `curve_layout`; `B` is the engine buffer
length, not a runtime count. A static tuple of events produces a tuple of
`EventCandidates`, one entry per event. Each entry contains `values (B,K)`,
`eligible (B,K)`, `bracket_mask (B-1,K)`, `spectra (B,K)` or `None`, and
`permutation (B,K)` or `None`. Scalar tests have `K=1`. Array positions encode
bracket and mode; no compaction, capacity limit, or dropped candidates.

Port the existing predictive Hungarian assignment to JAX with bounded
augmenting-path loops, deterministic lowest-column tie breaks, and the same
nonfinite cost replacement. `track_modes_array(spectra, valid)` returns the
tracked spectra and raw-column permutations. A `lax.scan` carries the previous
two valid spectra; padding does not update prediction. Preserve linear
extrapolation, rather than silently reverting to nearest-neighbor matching.
Map `trivial_index` through the permutation by integer identity; equal
multipliers must not cause a second nearest-value match.

Built-ins implement `scan_arrays(data, args)`; scalar test functions return
JAX scalars. Codim-2 scans evaluate the original RHS spectrum, not the curve
residual Jacobian, and mask the pinned modes. CP/GH use array-pure coefficient
evaluation with branchwise mode orientation and unit Hopf normalization as
specified in 6.2. Extended layout metadata/decoding (Task 6.2a) must land before
these array scans; port `_drop_nearest_pinned_pair` to distinct-column array
selection too, without traced Python branches. For PD/NS bind each event's
effective strategy to its matching cached spectrum, trivial index and status,
as specified in 6.3; never reuse branch spectra across different strategies. Existing `scan(points)` remains an eager compatibility hook for
custom events. A custom event without `scan_arrays` may use an array-pure
`test_function`; a custom Python-only scanner raises a descriptive `TypeError`
at trace construction. No callback or silent empty result.

Rejected: host callbacks (device synchronization and poor batching), greedy
matching (changes mode identities), and padded lists of `EventHit` (hits mean
validated refinements, which detection cannot supply).

## API and invariants

Add `ContinuationResult.event_candidates: tuple[EventCandidates, ...] = ()`
and include it in flatten/unflatten. Under tracing, `events` and
`rejected_events` remain empty lists; candidates are explicitly unvalidated.
Eager built-ins scan with the same kernels and feed candidates into the existing
locality, validation, sorting, and adjacent-root dedup logic. Export
`refine_event_candidates(result, problem, events, *, tolerance=None,
max_iterations=50, solvers=Solvers()) -> ContinuationResult`; it accepts one unbatched result
outside a trace, reconstructs `TrackedMode` only for spectral mode candidates
and dispatches scalar candidates directly to scalar refinement, and returns a result with refined
hits. Pass the same `solvers` bundle used for detection so periodic refinement
uses the same Floquet strategy. Batched users slice a single result pytree first.

Only consecutive valid endpoints may open a bracket. Opposite nonzero signs
or exactly one zero qualify; an identically zero mode never does. A sampled
zero belongs to the preceding eligible bracket, or the following bracket if
there is no preceding owner. Nonfinite values never qualify. Preserve separate
crossing modes in one bracket. `step_update(args,left.u,left.p)` supplies bracket
refinement args, exactly as `events.py:508` currently does; `PeriodicArgs` and
`HopfCurveArgs` retain their structure. Existing reconstructible hooks overwrite
anchors from the left point; the BT fixed anchor is retained unchanged from
problem args rather than overwritten. Stateful cumulative hooks would require stored args
history and are outside this change. Traced `bothside=True` remains unsupported.

Detection masks and assignment permutations have stopped gradients. They do
not promise useful derivatives at a crossing or degeneracy. Smooth branch
outputs retain the current contract: forward mode for `loop='while'`, reverse
mode only with `loop='scan'`. No reverse-mode claim for refinement orchestration.

## Verification

New `tests/test_event_scan.py`:
- `test_crossing_masks_zeros_padding_nonfinite`: values `[-1,0,1,0,NaN]`,
  valid `[1,1,1,0,0]` give mask `[1,0,0,0]`; all zeros give no brackets;
  values `[0,1]` give `[1]`.
- `test_array_assignment_matches_host_predictor`: permuted spectra with modes
  `p,-p,0.2+0.1*p+2j,-0.2+0.1*p+3j`, `p=linspace(-1,1,21)`, agree with
  host tracking away from ties; a separate tied-cost matrix of all zeros
  chooses the identity permutation.
- `test_jit_vmap_detects_shifted_hopf`: shifted equilibrium `(0.5,-0.3)`,
  linear block `[[p-a,-2],[2,p-a]]`, `a=[-0.25,0.25]`, fixed natural
  parameter grid `[-0.5,0.5]` in steps `0.125`: one positive-imaginary
  crossing per batch, at each exact `a`; eager refinement has `abs(p-a)`
  below `20*default_tol(dtype)`.
- `test_scan_preserves_raw_trivial_identity`: duplicate values `1,1,1.1`,
  raw trivial index `1`, permutation `[2,0,1]` excludes tracked column `2` only.
- `test_candidates_refine_with_left_anchor`, `test_custom_python_scan_rejected_under_jit`,
  `test_loop_scan_event_detection_keeps_smooth_branch_grad`: `u=a*p` on
  fixed natural samples `p=[0,.25,.5,.75,1]` (`ds=.25`,
  `adaptive=False`, `max_steps=4`) has derivative of summed states
  `2.5` with respect to `a`, regardless of stopped candidate masks.

Run default and slow regressions in both precision modes; retain existing
multi-mode, failed-refinement, sheet-dedup, and exact-zero tests.

## Eager bothside assembly

Assemble the state branch first: reverse valid backward points into increasing
branch order, keep one shared seed, then append forward points without its
seed. Recompute built-in candidates with the shared array kernels on this
assembled branch. This reverses endpoint ownership, remaps bracket rows to
assembled indices and reruns predictive tracking/orientation and zero ownership
consistently; do not concatenate or reverse old candidate masks/permutations.
Padding remains invalid and creates no bracket. Custom eager scanners receive
the assembled points. Traced bothside remains unsupported.

`test_bothside_candidate_assembly` compares assembled scanning to a direct
single ordered buffer, with a crossing on each arm and exactly at the shared
seed. Check bracket indices/modes, one seed-zero owner, reversed tracking
permutations and eager refined-root dedup, including unequal arm lengths and
padding. `test_scalar_candidate_refinement_dispatch` ensures CP/BT scalar
candidates do not require a fictitious `TrackedMode`.
