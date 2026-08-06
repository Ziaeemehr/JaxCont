# Speaker notes

These notes make the 112-page JaxCont v0.3.1+ deck teachable without assuming
that the audience already knows continuation, bifurcation terminology, or JAX
transformations. Explain the scientific question and picture before the
equation. Use frame titles as navigation anchors; page references below match
the current clean build and must be refreshed if pagination changes.

## Status language to use aloud

“JaxCont v0.3.1+” means the published v0.3.1 release plus selected current-main
features intended for v0.4. It does not mean that v0.4 is released.

- Say **Available in v0.3.1** for released features.
- Say **Current main — planned for v0.4** for PRC/dPRC, `plot_prc`, and
  Examples 13–14.
- Say **Future work — not implemented** for explicit capability boundaries.

Never imply support for branch switching, two-parameter bifurcation-curve
continuation, automatic cycle discovery from a Hopf point, general BVP
continuation, or connecting-/homoclinic-/heteroclinic-orbit continuation.

## Presentation routes and timing

The times are planning ranges, not promises. Add live-demo time separately and
leave margin for compilation, questions, and discussion.

### Overview route — about 35–45 minutes

Use the title; all of Chapters 1 and 2; the Chapter 3 divider plus “A
fixed-parameter slice fails at the fold”, “Pseudo-arclength changes the
coordinate”, “Predict along the tangent, correct across it”, and its recap;
selected Chapter 5 concept/event/result frames; all of Chapters 8 and 11; and
Chapter 12 through “Main takeaways”.

This route answers: what continuation is, why folds need a new coordinate,
what representative equilibrium/cycle/visual outputs mean, how to run a
study, and where the boundaries are. Treat the Chapter 12 appendix as questions
only.

### Methods route — about 65–85 minutes

Show the title and the Chapter 1 status legend, then use Chapters 2–7 and 10.
If time is tight, shorten Chapter 4 to the public call, result tree, engine
dispatch, `vmap`, and the final guarantees table. In Chapter 5, use one of the
two periodic-event figures rather than both. The appendix pages on bordered
continuation and the fixed-buffer engine are optional.

This route emphasizes continuation geometry, collocation, equilibrium and
cycle stability, Hopf refinement, local normal forms, direct codimension-two
systems, and validation.

### Complete route — about 105–125 minutes

Use Chapters 1–12 through “Main takeaways” (pages 1–105). Plan this as a long
seminar or two-part workshop. Pages 106–112 add roughly 10–20 minutes and are
optional technical appendix material. A complete route with two or more live
demos is better split into two sessions.

## Live-demo reliability

Rehearse on the presentation machine and keep the reviewed figure plus expected
stdout available. JAX compilation, an interactive Matplotlib backend, or a new
array shape can turn a short script into a poor live moment.

| Demo | Delivery recommendation | Cue and fallback |
|---|---|---|
| README saddle-node quick start | Best true live demo after one warm-up run | Paste the root README “Quick start” block. Point to the fold event and annotated branch. If the GUI fails, use the Chapter 2–3 cubic drawings and report the expected fold at `p=0`. |
| `example_03_van_der_pol.py` | Reliable live after imports/JIT are warm | Run from `examples/` with `MPLBACKEND=Agg`; show the saved `van_der_pol.png` and stdout. Emphasize that `l1=0` makes this a degenerate center crossing. |
| `example_10_van_der_pol_limit_cycle.py` | Prefer pre-generated output | The script first integrates a transient, locates a cycle, compiles collocation, and continues it. Use the saved gallery figure; show the command and the period trend. |
| `example_12_fitzhugh_nagumo_phase_plane.py` | Prefer the tracked figure | The script opens interactive figures and does not save them. Use the Chapter 8 snapshot unless the GUI has been rehearsed. |
| `example_13_phase_response_curve.py` | Short live option on the current-main checkout | Run with `MPLBACKEND=Agg`; show the saved two-panel image. State the current-main status before the command. |
| `example_14_prc_shooting_validation.py` | Pre-generate; do not depend on it live | Independent shooting, reconverged finite differences, and plotting make it the slowest route. Use the reviewed four-panel asset and captured exact diagnostics. |

The periodic-event Examples 08–09 are also good pre-generated evidence. Their
reviewed images and analytic crossings are already on pages 47–48.

For headless runs:

```bash
cd examples
MPLBACKEND=Agg python example_03_van_der_pol.py
```

Substitute the selected example filename. Example 12 is the exception: because
it does not save its figures, headless execution is a command check rather than
a useful display.

## Chapter 1 — Orientation and application map

**Pages 2–6; source:** `chapters/01_orientation.tex`.

The chapter’s job is to establish the status vocabulary and let the audience
choose a scientific question before they see algorithms.

1. On the divider, say that one model can be viewed as steady states, cycles,
   stability changes, local organizing points, state-space geometry, or phase
   sensitivity.
2. On “How to read v0.3.1+”, read the plus sign literally. Pause long enough
   for the audience to see that PRC/dPRC is orange/current-main and that future
   workflows are red/not implemented.
3. On the application map, begin at the left-hand questions. The method is a
   response to the scientific object being sought, not a menu chosen for speed.
4. Use the routes frame to tell the audience what you will skip. This makes a
   later chapter jump intentional rather than confusing.
5. Treat the outcomes as a contract: the talk is about interpretation and
   trustworthy use, not only API recall.

**Interpretation caution:** “Available” means implemented and released. It
does not mean every research model, seed, or parameter range will converge.

**Transition:** “We now need the one idea shared by every later chapter: a
branch is a connected family of roots.”

## Chapter 2 — Continuation fundamentals

**Pages 7–13; source:** `chapters/02_continuation_fundamentals.tex`.

Use the scalar cubic `F(u,p)=p+u-u^3/3` throughout so that only the numerical
question changes.

1. Contrast three questions: simulation follows time, a root solve fixes `p`,
   and continuation follows connected `(u,p)` pairs.
2. On the branch drawing, point out that a smooth geometric curve need not be
   a single-valued function `u(p)`.
3. Explain prediction as reuse of locality: the previous accepted root is a
   strong seed for the next correction.
4. Read the Jacobian formula as sensitivity, not as an algebra exercise.
5. At the fold, trace the smooth blue curve with your hand while noting that
   the horizontal parameter coordinate reverses.

**Interpretation caution:** A branch diagram is a solution set of the supplied
mathematical model. It is not a trajectory, a basin plot, model validation, or
a causal claim.

**Likely question — “Did the branch end at the fold?”** No. The physical
parameter stopped being a valid local coordinate; the geometric branch can
remain smooth.

**Transition:** “If the branch survives but the coordinate fails, change the
coordinate.”

## Chapter 3 — Natural and pseudo-arclength continuation

**Pages 14–25; source:** `chapters/03_pseudo_arclength.tex`.

The primary teaching move is visual: natural continuation takes a fixed-`p`
slice; pseudo-arclength lets state and parameter move together.

1. Show why the fixed-parameter slice becomes tangent at the fold.
2. On “Pseudo-arclength changes the coordinate”, make `s` an artificial local
   progress coordinate, not physical time.
3. On the predictor–corrector picture, name A (accepted), B (tangent
   prediction), and C (corrected root). The dashed hyperplane is transverse to
   the tangent.
4. Translate the augmented equations: `F=0` says “be a solution”; `g=0` says
   “be one local step ahead”.
5. Explain the two bordered systems only after the picture. The extra parameter
   column and geometric row can keep the full system regular at an ordinary
   fold.
6. Use the state machine and step-size table to show that rejection keeps the
   last accepted point and reduces the next attempt’s step.
7. The method-selection table is the decision slide: natural continuation is
   useful on monotone branches; pseudo-arclength is the exploratory default
   when folds matter.

For the overview route, skip the bordered tangent/Newton derivations and the
detailed step-size table. Keep the geometry and recap.

**Likely question — “Is pseudo-arclength exact arclength?”** No. Its extra
constraint is a local linear approximation built from the preceding tangent.

**Likely question — “Why not always use it?”** It solves a larger system and
needs tangent information. Natural continuation remains simpler for a known
monotone branch.

**Interpretation caution:** Passing an ordinary fold does not discover a
disconnected branch or authorize branch switching at a bifurcation.

**Transition:** “Once continuation is expressed as a finite residual, one
public execution model can serve equilibrium and periodic problems.”

## Chapter 4 — Public API and JAX execution model

**Pages 26–37; source:** `chapters/04_api_and_jax.tex`.

Keep public usage separate from implementation mechanics.

1. Start with the functional front door: problem, algorithm, span, settings,
   events, and one `ContinuationResult`. Stress that the state must solve the
   model at `p_span[0]`.
2. Read the result tree once. `branch.states`, `params`, stability data,
   validity, and event diagnostics serve different interpretation tasks.
3. Explain eager trimming versus traced fixed buffers. Under `jit`/`vmap`, use
   the validity mask or `stats["n_valid"]`; padding is not scientific data.
4. The dispatch diagram is the architectural seam: one public call selects the
   natural or pseudo-arclength engine and returns the same result contract.
5. Clarify “scan”: it is the historical engine name; the bounded outer loop is
   `jax.lax.while_loop`, not `jax.lax.scan`.
6. On JIT, separate cold compilation from warm execution. On `vmap`, each
   branch remains sequential while independent branches are batched.
7. On autodiff and the custom VJP, say that derivatives follow the implemented
   equations and inherit the quality of the converged root.
8. Close with the “enables / does not guarantee” table.

**Demo cue:** The README saddle-node quick start fits immediately after the
front-door frame. Show the result tree only if the audience asks how to inspect
the returned branch.

**Likely question — “Does JIT make the answer more accurate?”** No. It changes
orchestration and compilation, not conditioning, convergence, or the numerical
method.

**Likely question — “Can events run inside `vmap`?”** The current transformed
path requires `events=()`; eager event orchestration is a distinct boundary.

**Interpretation caution:** Fixed shapes enable transformation. They do not
make all event logic batchable or parallelize steps within one branch.

## Chapter 5 — Periodic orbits and Floquet stability

**Pages 38–50; source:** `chapters/05_periodic_orbits.tex`.

The conceptual bridge is: collocation converts one complete cycle and its
period into one finite nonlinear root.

1. Contrast one equilibrium state with an entire sampled orbit plus period.
2. Introduce the three residual blocks—defect, continuity/wrap-around, and
   phase—before the API.
3. Use the shifted-circle picture to explain why the phase condition removes a
   neutral representation freedom.
4. Make the responsibility boundary explicit: the user simulates elsewhere,
   extracts one cycle, rebases its time array to zero, and supplies a reasonable
   period. JaxCont resamples/refines that guess and then continues it.
5. On the packed-state frame, warn that one stored component at one mesh point
   is not automatically cycle amplitude.
6. For stability, remove exactly one trivial multiplier nearest `+1`. Every
   other multiplier must lie inside the unit circle for a stable cycle.
7. Use the PD/NS geometry before the two real figures. A crossing is an event
   condition; the detector does not construct the doubled branch or torus.
8. End with fixed-mesh, dense-solve, float32-tolerance, guess-quality, and event
   boundaries.

**Demo cue:** Prefer the saved Example 10 output. Pages 47–48 already contain
reviewed Examples 08–09 with analytic crossings, so they are safer than live
periodic compilation.

**Likely question — “Why not start a cycle automatically at Hopf?”** That
requires cycle construction/branch switching. JaxCont v0.3.1 refines and
continues a supplied coarse cycle; it does not implement that workflow.

**Likely question — “Why not use negative real parts for cycle stability?”** A
Floquet multiplier measures growth over a full period, so the relevant test is
its magnitude. Negative real parts are the continuous-time equilibrium rule.

## Chapter 6 — Hopf refinement and criticality

**Pages 51–60; source:** `chapters/06_hopf_classification.tex`.

The chapter answers two local questions: where exactly is the Hopf point, and
what local normal form does it have?

1. Separate detection from refinement. A sign-change bracket identifies a
   candidate neighborhood; a converged extended-system solve satisfies the
   equilibrium and imaginary-eigenpair equations together.
2. Name the returned equilibrium, parameter, critical plane, and angular
   frequency. Explain the normalization/phase equations as removal of
   eigenvector scale and rotation ambiguity.
3. Introduce the radial normal form only after asking whether the nearby small
   cycle is locally attracting or repelling.
4. Read `l1 < 0` as supercritical and `l1 > 0` as subcritical on the appropriate
   unfolding side. Treat `l1 ≈ 0` as a tolerance-aware degenerate/GH candidate.
5. On the API frame, state that `lyapunov_coefficient` requires a right-hand
   side complex-analytic in the state near the point.
6. On `Hopf.refine()`, explain the reported `omega0`, `l1`, `criticality`, and
   method, then repeat that finite diagnostics alone are not a convergence
   certificate. Verify the extended-system residual.
7. End with what the solve does not do: no automatic cycle, branch switch,
   Hopf-curve continuation, or global prediction.

**Likely question — “Does the sign of `l1` tell me the whole cycle branch?”**
No. It classifies the local Hopf normal form. Distant folds, global stability,
and far-from-Hopf behavior require a supplied cycle and continuation.

**Likely question — “How small is ‘near zero’?”** There is no universal
number. The default tolerance is a starting point; state/parameter scaling and
the extended-system residual must inform the judgment.

**Likely question — “Why can a finite criticality label still be unsafe?”**
The refinement interface can return a final finite iterate without a separate
convergence flag. Check the defining residual before trusting the label.

**Transition:** “A near-zero `l1` adds a second local condition, so the natural
next object is a direct two-parameter point refinement.”

## Chapter 7 — Direct codimension-two point solvers

**Pages 61–67; source:** `chapters/07_codim2_solvers.tex`.

Keep “one point from one supplied guess” visible on every frame.

1. Use the taxonomy to connect the familiar fold/Hopf conditions to CP, BT,
   GH, ZH, and HH.
2. The comparison table is a reference, not a memorization test. Highlight the
   selected type, minimum state dimension, returned witness vectors/frequencies,
   and seed needs.
3. On the direct-solver contract, point from approximate state plus parameter
   pair through the extended Newton solve to a refined point and `converged`
   flag.
4. Use BT as the concrete example: `J v0 = 0` and `J v1 = v0` witness the
   double-zero Jordan chain.
5. Explain parameter-only wrappers only after checking the corresponding point
   solver. They omit the convergence flag so they compose with gradients.
6. For HH, mention `n >= 4`, required keyword-only `seed_b`, and the
   model-scaled frequency-separation guard. ZH requires `n >= 3`.

**Likely question — “Do these solvers draw a two-parameter bifurcation
diagram?”** No. They refine one nearby root of an extended point-defining
system. They do not continue fold or Hopf curves.

**Likely question — “Why return vectors and frequencies?”** They witness the
defining degeneracy. A parameter pair without a converged spectral/normal-form
witness is not enough to identify the point.

**Likely question — “Can I differentiate the parameter-only wrapper without
checking convergence?”** It is technically composable, but scientifically
unsafe. Establish the local root with `*_point(...)` and its flag first.

**Interpretation caution:** A codimension-two label is local information and
depends on a good seed, convergence, dimension/seed contracts, scaling, and
nondegeneracy assumptions.

## Chapter 8 — Visualization: parameter space and state space

**Pages 68–73; source:** `chapters/08_visualization.tex`.

Begin with the question-to-view table. A plot is selected by what it can answer,
not by which helper is most visually attractive.

1. A branch diagram varies the parameter; an eigenvalue/Floquet view tracks
   local stability; a branch-state projection relates stored solutions; a 2D
   phase plane freezes one parameter and shows local flow geometry.
2. On the composable API surface, point out that `plot_prc` is current-main;
   the other named visualization groups are released.
3. Use the data-flow diagram to separate continuation results from a frozen
   vector field and optional trajectory simulation.
4. On FitzHugh–Nagumo, read the left panel first: locate the Hopf transition
   along `I`. Then move to the right panel at `I=0.5`: identify the hollow
   unstable equilibrium, nullclines, arrows, and trajectory.

**Demo cue:** Use the tracked two-panel image. The source script opens two
interactive figures and does not save them, so do not rely on a headless live
run for the visual.

**Likely question — “Is a branch-state projection a phase portrait?”** No. It
plots stored solution coordinates, not the vector field or time-domain flow at
one frozen parameter.

**Interpretation caution:** A 2D phase plane does not establish a basin,
higher-dimensional geometry, automatic branch discovery, or a continued
periodic orbit. Its trajectory layer uses an explicit simulation.

## Chapter 9 — Phase-response curves and parameter sensitivity

**Pages 74–85; source:** `chapters/09_prc_dprc.tex`.

State the current-main/planned-for-v0.4 status before teaching the method.

1. Use the three-kick picture: the same small state impulse produces different
   first-order phase shifts at different phases.
2. Introduce isochrons and `Z = grad(phi)` visually. The iPRC maps a small kick
   to `Delta phi ≈ Z^T delta x`; it is not a finite-reset map.
3. In the adjoint diagram, state perturbations propagate forward while phase
   sensitivities propagate backward through transposed interval maps and close
   periodically.
4. The normalization `Z(0)·f(x0,p)=2π/T` chooses radian phase scale.
5. Compare `prc_curve`, `branch_prc`, and `dprc_curve`: one orbit, compatible
   batched branch states, and a parameter derivative after reconverging the
   periodic orbit.
6. On the real Example 13 figure, read sign, state component, phase, and
   magnitude. A horizontal shift may be only a phase-origin difference.
7. End with the stable-cycle, fixed-mesh, simple-unit-multiplier, periodic-
   closure, and mesh-point-output boundaries.

**Demo cue:** Example 13 is a short current-main demo when the checkout is
known. Otherwise use the reviewed page-82 figure.

**Likely question — “What exactly is JaxCont’s dPRC?”** It is `d(PRC)/dp` for
the full map that reconverges the periodic orbit, updates its phase anchor, and
then recomputes the PRC. It is not a frozen-orbit partial derivative.

**Likely question — “Is that MatCont’s dPRC?”** No. MatCont exports a time
derivative under that name. Comparing it directly with JaxCont’s parameter
derivative would compare different quantities.

**Likely question — “Can I use the iPRC for a large kick?”** Not by this
first-order interpretation. A large reset can leave the local isochron regime
or cross a basin boundary.

**Interpretation caution:** Compare PRCs only after aligning phase origin,
phase units, normalization, state ordering, and the perturbed component.

## Chapter 10 — Validation: from residuals to independent evidence

**Pages 86–94; source:** `chapters/10_validation.tex`.

This chapter teaches evidence vocabulary, not a single package-wide pass/fail
claim.

1. Climb the ladder from run health to repeats, analytic oracles, independent
   algorithms, and independent packages/reference artifacts. Higher rungs do
   not excuse failures on lower ones.
2. Use the evidence matrix to match the validation method to the capability.
   Avoid a universal “validated” label.
3. The sheared-circle figure has closed-form, collocation, and shooting
   evidence. The Van der Pol figure is an independent numerical comparison but
   has no closed-form PRC here.
4. Treat printed maximum errors as diagnostics unless the source declares a
   metric and tolerance.
5. On the diagnostics table, preserve partial failure. The reviewed
   `MC-LC-002` mismatch is evidence to report, not a reason to weaken a
   threshold.
6. On convention alignment, describe the rule before applying it. Do not shift,
   reorder, or rescale after seeing the answer merely to minimize error.
7. Separate released validation evidence from current-main PRC evidence.

**Likely question — “Why do two PRCs agree only after a horizontal shift?”** A
periodic orbit has an arbitrary phase origin. Apply a reproducible circular
alignment rule and report it; do not interpret the shift as solver error by
default.

**Likely question — “Why remove one multiplier near +1?”** Autonomous cycles
have one trivial time-shift multiplier. Remove exactly one nearest +1 from
both sides before matching the nontrivial spectra.

**Likely question — “If the plots overlap, did validation pass?”** Not yet.
Choose a declared metric, tolerance, interpolation/alignment rule, and scope.

**Likely question — “Do repository tests validate my model?”** No. They
support implementation behavior. Model choice, seed, parameter domain,
resolution, and scientific interpretation need their own evidence.

## Chapter 11 — Guided application workflows

**Pages 95–100; source:** `chapters/11_guided_workflows.tex`.

Use this as the practical handoff from concepts to a repeatable study.

1. On the six-stage loop, spend most time on the trustworthy seed and
   inspect/visualize stages. Begin with a short branch and earn complexity.
2. Demo cards 1–2 cover the README saddle-node and Van der Pol equilibrium
   crossing. The latter has `l1=0`; do not narrate it as a generic birth of the
   familiar Van der Pol cycle.
3. Demo cards 3–4 separate supplied-cycle continuation from frozen-parameter
   phase-plane simulation.
4. Demo cards 5–6 are current-main. `dprc_curve` reconverges the orbit, and the
   shooting example checks the PRC computation on JaxCont-reconverged orbits;
   it is not independent orbit discovery.
5. On “How to read a result”, explain what the public result does and does not
   store. Re-evaluate the model residual on valid states and compare the last
   valid parameter with the requested endpoint and configured stop conditions.

**Demo cue:** Choose at most one true live demo for an overview talk. For a
workshop, run the README quick start first, then Example 13 or a pre-rehearsed
Example 03. Use the other cards as reproducible homework routes.

**Interpretation caution:** Missing points do not prove that a physical branch
ended. The public result does not store residual norms, rejection history,
accepted step sizes, or a termination-reason field.

**Transition:** “The final chapter states the claims we can make, the claims we
cannot make, and the vocabulary needed to revisit the technical details.”

## Chapter 12 — Scope, glossary, and technical appendix

**Pages 101–112; source:** `chapters/12_scope_and_appendix.tex`.

Pages 101–105 close every route:

1. Read the capability matrix by source status. Repeat that v0.4 is not
   released.
2. Use the two glossary frames as a question-driven reference rather than
   reading every definition aloud.
3. End on “Main takeaways”. The final sentence is the interpretation boundary:
   a computed diagram is evidence about the supplied mathematical model, not
   automatic validation of the model or a causal claim.

Pages 106–112 are optional:

- “natural versus pseudo-arclength” and the scalar cubic derivation answer
  numerical-method questions;
- accepted-point pseudocode, fixed carry, branch-free selection, and
  eager/traced reassembly answer implementation/JAX questions;
- the final sources frame records where claims and reproducible evidence live.

Do not show the fixed-buffer internals to an overview audience unless asked.
For a methods audience, the accepted-point pseudocode is usually more useful
than listing every carry field.

**Final audience check:** Ask participants to state one available capability,
one current-main capability, one unsupported workflow, and one validation step
they would add before making a research claim.

## Interpretation cautions to repeat

- Convergence and small residuals support a numerical solution; they do not
  prove that it is the intended branch or that the model is scientifically
  valid.
- Event detection is a candidate; a local extended-system refinement must be
  checked before its diagnostics are trusted.
- Hopf `l1` and codimension-two labels are local and tolerance/scaling aware.
- A supplied periodic orbit is not automatic cycle discovery or branch
  switching.
- A phase-plane trajectory is simulation; a periodic branch is continuation.
- PRC/dPRC requires aligned phase and derivative conventions.
- Independent agreement is meaningful only after lower-level run health and a
  reproducible comparison rule are established.
