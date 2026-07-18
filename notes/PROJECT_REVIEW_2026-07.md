# JaxCont — Project Review & Path Forward

**Date:** 2026-07-17
**Reviewer:** Claude (code + strategy review)
**Verdict:** Worth continuing — *but* only if you narrow the scope hard and ship one thing that works well. Right now it's ~60% of a broad library and 100% of a documentation sprawl.

---

## 1. TL;DR

- **Is it worth making?** Yes, conditionally. There is a real, unfilled niche: *a pure-Python, autodiff-native, GPU-capable continuation library*. Nobody occupies it well. Your instinct about Julia being a blocker is exactly the market gap.
- **Is it in the right direction?** The *code* direction is fine. The *project management* direction is not — you have 13 status/planning `.md` files in `notes/` that contradict each other and no single source of truth. That's why you can't tell where to start.
- **What to do:** Cut scope to **equilibrium continuation + fold/Hopf detection + stability**, make it genuinely fast (JIT) and genuinely documented, ship **v0.1.0** to PyPI, then decide based on whether anyone uses it. Do **not** wait for a big release.

---

## 2. Does the community need this?

**The gap is real.** The current landscape:

| Tool | Language | Status | Weakness |
|------|----------|--------|----------|
| AUTO-07p | Fortran | Gold standard | Ancient, painful to install/use |
| MATCONT | MATLAB | Feature-complete | MATLAB license, GUI-bound |
| BifurcationKit.jl | Julia | Excellent, active | **Julia** (your blocker, and many others') |
| PyDSTool | Python | Mature but aging | Old codebase, no autodiff, slow, semi-maintained |
| PyCoBi / PyRates | Python | Active | Just a wrapper around AUTO-07p Fortran |

So every Python option today is either a wrapper around old Fortran or an aging pure-Python package **with no automatic differentiation and no GPU**. A JAX-native library that gets exact Jacobians for free and can run on GPU is a genuinely differentiated position — *if* it actually works and is easy to use.

**The honest caveat:** this is a niche field. Users are computational scientists (dynamical systems, comp neuro, systems biology, engineering). The total addressable audience is maybe thousands, not millions. That's fine for a research tool / paper-backed package, but don't expect it to go viral. The realistic win is: *your own research uses it, a handful of labs adopt it, you get a JOSS/Zenodo paper and citations.* That is a perfectly good outcome and worth the effort — but scope the work to match that payoff. Don't build MATCONT.

**Sources:**
- [PyCoBi (AUTO-07p wrapper)](https://github.com/pyrates-neuroscience/PyCoBi)
- [bifurcation-analysis GitHub topic](https://github.com/topics/bifurcation-analysis)
- [Tutorial on numerical continuation (arXiv 2008.05226)](https://arxiv.org/pdf/2008.05226)

---

## 3. State of the code (what actually works)

I ran the suite in your `jaxcont` env: **49 passed, 4 skipped, ~60% coverage.** That's a healthy sign — the core is not vaporware.

**Solid / working:**
- Pseudo-arclength continuation (passes fold points) — the hard part, and it's done.
- Natural-parameter continuation.
- Newton solver with autodiff Jacobians (100% test coverage on `newton.py`).
- Fold & Hopf detection with bisection refinement.
- Stability analysis (eigenvalues along branch).
- Basic plotting.

**Partial / stub (per your own IMPLEMENTATION_STATUS.md):**
- Periodic orbits — structure only, untested.
- Floquet multipliers — 22% coverage, mostly stub.
- BVP collocation/shooting — `NotImplementedError`.
- Normal forms / Lyapunov coefficient — return placeholder zeros.

**Poorly covered:** `plotting.py` (9%), `utils/config.py` (27%), `floquet.py` (22%). These are either untested or dead weight.

### Concrete code issues found

1. **Broken error handling (real bug).** In [pseudo_arclength.py:129-139](../src/jaxcont/core/pseudo_arclength.py#L129-L139) and [:199-206](../src/jaxcont/core/pseudo_arclength.py#L199-L206), `try/except` wraps `jnp.linalg.solve`. **JAX does not raise on singular matrices** — it returns `inf`/`nan`. So the "singular Jacobian" fallbacks never trigger; you silently propagate NaNs instead. This will bite exactly at bifurcation points, which is where it matters most. Fix: check conditioning or residual explicitly, or use `jnp.linalg.lstsq`.

2. **`df/dp` by finite difference while everything else is autodiff.** [:117-121](../src/jaxcont/core/pseudo_arclength.py#L117-L121) and [:189-192](../src/jaxcont/core/pseudo_arclength.py#L189-L192) use a manual `eps=1e-7` finite difference for the parameter derivative. You're in JAX — this should be `jacfwd`/`grad` w.r.t. the parameter for an exact derivative. Right now the library's headline feature (autodiff) isn't fully used on its own hot path.

3. **No JIT on the hot loop (your own PROFILING_ANALYSIS.md confirms).** The correction loop, tangent computation, and eigenvalue calls are plain Python calling small JAX ops. Result: ~50 ms/point, dominated by dispatch overhead. For a library whose *entire selling point* is "high-performance JAX," this is the credibility gap. The fix is not decorator-sprinkling — it's restructuring the Newton/corrector step into a single `jax.lax.while_loop` / `jax.lax.scan` that JIT-compiles as one unit.

4. **Python-loop continuation.** The outer stepping is a Python `for`. That's acceptable (branches are inherently sequential, and you want per-step introspection for bifurcation detection), but the *inner* Newton solve must be jittable for the performance story to hold.

**Bottom line on code:** the math is right and tested; the "high-performance" and "autodiff-native" claims in the README are *not yet true on the hot path*. That's the single most important thing to fix before publishing, because it's your differentiation.

---

## 4. The documentation sprawl problem

This is why you feel lost. `notes/` contains **13 markdown files**, several overlapping:

```
BIFURCATION_DETECTION_IMPROVED.md   DEVELOPMENT.md          NEXT_STEPS.md
BISECTION_STATUS.md                 DOCUMENTATION_STATUS.md  PROFILING_ANALYSIS.md
BORDERED_NEWTON_VERIFIED.md         IMPLEMENTATION_STATUS.md PROFILING_RESULTS.md
                                    GPU_SETUP_COMPLETE.md    STRUCTURE.md
                                    INSTALL.md               TODO.md
```

`IMPLEMENTATION_STATUS.md`, `TODO.md`, `NEXT_STEPS.md`, and `DEVELOPMENT.md` all describe "what's done / what's next" with different answers. `PROFILING_ANALYSIS.md` and `PROFILING_RESULTS.md` are near-duplicates. `GPU_SETUP_COMPLETE.md` is empty (0 bytes).

**Recommendation:** these are working notes, not documentation. Collapse them:
- Keep **this file** + **one** `ROADMAP.md` as the single source of truth for status.
- Archive the rest into `notes/archive/` (don't delete — they have real profiling data worth keeping).
- User-facing docs live in `docs/` (Sphinx), not `notes/`.

Also: at review time the root README still had placeholder author, repository,
and citation metadata. These were fixed during the v0.1.0 release preparation.

---

## 5. Recommendation: ship a minimal v0.1.0 now

You asked directly: *"publish something minimal that works, then add features — or wait for a big release?"*

**Ship minimal. Do not wait.** Reasons:
- Waiting for a "big release" is how research software dies half-finished. You already feel out of touch after a break — a shipped v0.1.0 is a fixed point you can return to.
- A published package with a clear scope ("equilibrium continuation in JAX") is *useful and honest*. A half-implemented broad package is neither.
- Real users surface real priorities. You'll learn more from 3 users of v0.1.0 than from 3 more months of speculative features.

**Define v0.1.0 as exactly this — and cut everything else from the public API:**

- ✅ Equilibrium continuation (natural + pseudo-arclength)
- ✅ Fold + Hopf detection with refinement
- ✅ Stability (eigenvalues) along the branch
- ✅ Plotting bifurcation diagrams
- ✅ 3–4 worked examples (pitchfork, Lorenz, one neural-mass — you already have these)
- ✅ **JIT'd Newton/corrector** (the credibility fix)
- ✅ Honest README (scope stated plainly: "equilibria only, for now")
- 🚫 Hide/mark-experimental: periodic orbits, Floquet, BVP, normal forms, codim-2

Everything hidden stays in the codebase behind a clear "experimental / not yet implemented" boundary so you don't ship stubs that raise `NotImplementedError` in users' faces.

---

## 6. Step-by-step plan

### Phase 0 — Tidy up (½ day)
1. Move all `notes/*.md` except this review into `notes/archive/`. Create `ROADMAP.md` (short, single source of truth).
2. Fix README placeholders (name, URL, citation). State the v0.1.0 scope explicitly.
3. Delete empty `GPU_SETUP_COMPLETE.md`.

### Phase 1 — Make the core honest & fast (1–2 weeks)
4. Replace finite-difference `df/dp` with autodiff (`jacfwd` w.r.t. parameter). — *correctness + "autodiff-native" claim*
5. Fix the `try/except`-on-`jnp.linalg.solve` bug; handle singular Jacobians via explicit conditioning check or `lstsq`.
6. Rewrite the Newton/corrector inner loop as a jittable `jax.lax.while_loop`. Benchmark before/after (you already have `profile_continuation.py`).
7. Add a GPU smoke test (even just "runs on GPU without error" on a 50-D system).
8. Get the *core* modules (`core/`, `solvers/`, `stability/eigenvalue.py`) to >85% coverage. Ignore periodic/BVP/floquet coverage for now.

### Phase 2 — Package & document for release (1 week)
9. Wire up Sphinx docs in `docs/` (you have `conf.py` + readthedocs config already). Write: install, quickstart, one full tutorial, API reference.
10. Verify `pyproject.toml` builds a clean wheel; test `pip install` in a fresh env.
11. Tag **v0.1.0**, publish to **TestPyPI first**, then PyPI.
12. Get a Zenodo DOI (you already have the badge stubbed). Optional: draft a JOSS paper — this field values citable software.

### Phase 3 — Grow based on demand (ongoing)
13. Only *now* pick up periodic orbits + Floquet (v0.2.0) — this is genuinely the highest-value next feature and the natural sequel.
14. Branch switching, two-parameter continuation, codim-2 → v0.3.0+, driven by user requests.

---

## 7. Where to start *tomorrow*

Do these three, in order:
1. **Phase 0.1** — collapse the notes into one roadmap so you stop feeling lost. (1 hour)
2. **Phase 1.6** — JIT the Newton loop and re-run your profiler. This proves the core thesis of the whole project. If JAX doesn't buy you a real speedup here, that's important to know early. (biggest single item)
3. **Phase 1.4/1.5** — the two correctness fixes, which are small and make the library trustworthy.

Once those three are done, the release is mostly packaging and prose — and you'll have concrete evidence the "high-performance autodiff continuation" pitch is true.

---

## 8. One-line answer to your questions

- *Worth making?* Yes — real gap, but scope it as a focused research tool, not a MATCONT clone.
- *Right direction?* Code yes, project hygiene no — fix the notes sprawl and the performance-claim gap.
- *Publish minimal or wait?* **Publish minimal now.** Equilibria-only v0.1.0.
- *Where to start?* Collapse notes → JIT the Newton loop → fix the two bugs → package → ship.
