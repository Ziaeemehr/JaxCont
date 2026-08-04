# Hopf Normal Form / First Lyapunov Coefficient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `l1` (Hopf criticality: super- vs subcritical) to JaxCont, via a differentiable
Hopf-point solver (`hopf_point`) and Kuznetsov's first-Lyapunov-coefficient formula
(`lyapunov_coefficient`), wired into `Hopf.refine()` so every detected Hopf point self-classifies.

**Architecture:** New module `src/jaxcont/bifurcations/hopf_normal_form.py`, sibling to
`fold_solve.py`, reusing the same `solvers/implicit.py:differentiable_root` extended-system
pattern. `hopf_point` does the Newton solve (differentiable via implicit function theorem);
`lyapunov_coefficient` is pure algebra on top (directional derivatives via `jax.jvp`, no
iteration), so the two compose with correct end-to-end gradients through ordinary chain-rule
composition — `lyapunov_coefficient` never calls `jnp.linalg.eig` itself.

**Tech Stack:** JAX (`jax.numpy`, `jax.jvp`, `jax.jacfwd`, `jax.lax.stop_gradient`), pytest,
Julia + BifurcationKit.jl v0.5.2 (already installed in this dev environment) for the independent
cross-check.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md` — read it first; it
  documents two non-obvious bugs already found and fixed during verification (see "Implementation
  notes verified during planning" section there). The exact code in this plan already incorporates
  both fixes — do not reintroduce the naive `q1·q2` phase constraint or the real `2n×2n`-block
  left-eigenvector construction described there as broken.
- All new functions must accept/return dtype matching the input `u`'s dtype for real quantities
  (float32 by default, float64 only if the caller has `jax_enable_x64` on) and use
  `complex64`/`complex128` accordingly for complex intermediates — mirrors `fold_solve.py`'s
  existing `p_guess = jnp.asarray(p_guess, u_guess.dtype)` convention. Do not hardcode
  `complex128`/`float64`.
- `l1 < 0` is supercritical, `l1 > 0` is subcritical, `|l1| < l1_tolerance` is `"degenerate"` — use
  these exact string values (`"supercritical"`, `"subcritical"`, `"degenerate"`), matching the
  spec.
- Every new public function needs a docstring citing Kuznetsov (*Elements of Applied Bifurcation
  Theory*, 3rd ed.) where the design spec does, matching this project's existing citation style in
  `fold_solve.py`/`solvers/implicit.py`.
- Run the full test suite (`pytest`) after every task and confirm no regressions before moving on
  — this project's established practice (every roadmap entry records a full-suite pass).

---

### Task 1: `hopf_point` / `hopf_parameter` — differentiable Hopf-point solver

**Files:**
- Create: `src/jaxcont/bifurcations/hopf_normal_form.py`
- Test: `tests/test_hopf_normal_form.py`

**Interfaces:**
- Consumes: `jaxcont.solvers.implicit.differentiable_root(G, x0, theta, *, tol, max_iter) -> Array`
  (existing, unchanged).
- Produces: `hopf_point(f, u_guess, p_guess, args=None, *, tol=1e-8, max_iter=50) ->
  (u, p, q1, q2, omega0)` (all `Array`); `hopf_parameter(f, u_guess, p_guess, args=None, *,
  tol=1e-8, max_iter=50) -> Array` (scalar `p`). Both consumed by Task 2's `lyapunov_coefficient`
  and Task 5's `Hopf.refine()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hopf_normal_form.py`:

```python
"""
Tests for jaxcont.bifurcations.hopf_normal_form: the differentiable Hopf-
point solver (hopf_point/hopf_parameter) and the first Lyapunov coefficient
(lyapunov_coefficient). See docs/superpowers/specs/2026-08-04-hopf-normal-
form-design.md.
"""

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.hopf_normal_form import hopf_point, hopf_parameter


def _textbook_hopf(u, p, args):
    # Standard supercritical-Hopf textbook example (Kuznetsov Sec 3.2/3.4):
    # exact Hopf at u=(0,0), p=0, omega0=1, l1=-1.
    x, y = u[0], u[1]
    r2 = x**2 + y**2
    return jnp.array([-y + x * (p - r2), x + y * (p - r2)])


def test_hopf_point_recovers_exact_hopf_of_textbook_example():
    u, p, q1, q2, omega0 = hopf_point(
        _textbook_hopf, jnp.zeros(2), 0.05, tol=1e-10, max_iter=50,
    )
    assert jnp.allclose(u, jnp.zeros(2), atol=1e-5)
    assert jnp.isclose(float(p), 0.0, atol=1e-5)
    assert jnp.isclose(float(omega0), 1.0, atol=1e-5)
    # q1+i*q2 is a unit vector (G4's normalization).
    assert jnp.isclose(jnp.dot(q1, q1) + jnp.dot(q2, q2), 1.0, atol=1e-5)


def _hopf_with_param_shift(u, p, shift):
    x, y = u[0], u[1]
    r2 = x**2 + y**2
    p_eff = p - shift
    return jnp.array([-y + x * (p_eff - r2), x + y * (p_eff - r2)])


def test_hopf_parameter_grad_matches_finite_difference():
    # p*(shift) = shift exactly for this family, so d(hopf_parameter)/d(shift) = 1
    # exactly -- a strong, non-trivial gradient check (not a coincidental 0).
    def p_star(shift):
        return hopf_parameter(
            _hopf_with_param_shift, jnp.zeros(2), 0.05, shift, tol=1e-10, max_iter=50,
        )

    grad = jax.grad(p_star)(0.1)
    eps = 1e-4
    fd = (p_star(0.1 + eps) - p_star(0.1 - eps)) / (2 * eps)
    assert jnp.isclose(grad, fd, atol=1e-3)
    assert jnp.isclose(grad, 1.0, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hopf_normal_form.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.bifurcations.hopf_normal_form'`

- [ ] **Step 3: Write the implementation**

Create `src/jaxcont/bifurcations/hopf_normal_form.py`:

```python
"""
Differentiable Hopf-point solver + first Lyapunov coefficient (Kuznetsov's
normal-form formula), via the extended system + implicit diff.

A Hopf point of ``f(u, p; args) = 0`` is the solution of the extended system

    G1: f(u, p)                  = 0   (n eqs)  equilibrium
    G2: J(u,p)*q1 + omega*q2     = 0   (n eqs)  Re[(J - i*omega)(q1+i*q2)] = 0
    G3: J(u,p)*q2 - omega*q1     = 0   (n eqs)  Im[(J - i*omega)(q1+i*q2)] = 0
    G4: q1.q1 + q2.q2 - 1        = 0   (1 eq)   unit norm
    G5: q1_seed.q2 - q2_seed.q1  = 0   (1 eq)   phase anchored to the seed

in the unknowns ``x = (u, p, q1, q2, omega)`` (dimension ``3n+2``), where
``q1 + i*q2`` is the right eigenvector for the critical eigenvalue
``i*omega``. The extended-system Newton solve and its implicit-function-
theorem gradient live in :func:`jaxcont.solvers.implicit.differentiable_root`,
shared with ``fold_solve.py``; this module only builds the Hopf-specific
``G`` and initial guess.

G5's phase constraint uses the FIXED seed eigenvector (``q1_seed``,
``q2_seed``), not the live ``(q1, q2)``: the natural-looking self-
referential alternative ``G5 = q1.q2 = 0`` has a Jacobian (with respect to
``(q1, q2)``) that can become exactly singular depending on the
eigenvector's own components -- found and fixed during design, see
``docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md``. Anchoring
to a fixed reference makes G5's gradient the constant vector ``(q2_seed,
-q1_seed)``, which cannot vanish. Inside ``G``, the seed is recomputed
under ``jax.lax.stop_gradient`` -- not bare ``theta`` -- because the seed
comes from ``jnp.linalg.eig``, and JAX has no gradient rule for
non-symmetric eigenvectors; the seed is an arbitrary fixed reference, not a
differentiated quantity, so cutting its gradient path is correct, not a
workaround.

Public entry points:
- :func:`hopf_point`     -> (u*, p*, q1*, q2*, omega0*), differentiable in ``args``
- :func:`hopf_parameter` -> p*,                          differentiable in ``args``
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd, lax

from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def _pack(u, p, q1, q2, omega):
    return jnp.concatenate([u, jnp.reshape(p, (1,)), q1, q2, jnp.reshape(omega, (1,))])


def _unpack(x, n):
    return x[:n], x[n], x[n + 1:2 * n + 1], x[2 * n + 1:3 * n + 1], x[3 * n + 1]


def _seed(f, u, p, args, n):
    """Undifferentiated Newton seed: eigenvector of J(u,p) with smallest
    |Re(eigenvalue)|, matching bifurcations.events.Hopf's own selection
    rule. jnp.linalg.eig (not just eigvals) is needed for the eigenvector;
    like fold_solve.py's SVD-based null-vector seed, this is never itself
    differentiated."""
    jac = jacfwd(f, argnums=0)(u, p, args)
    evals, evecs = jnp.linalg.eig(jac)
    idx = jnp.argmin(jnp.abs(jnp.real(evals)))
    vec = evecs[:, idx]
    vec = vec / jnp.linalg.norm(vec)
    q1 = jnp.real(vec)
    q2 = jnp.imag(vec)
    omega = jnp.abs(jnp.imag(evals[idx]))
    return q1, q2, omega


def _extended_residual(x, f, args, n, u_guess, p_guess):
    """G(x, args) for the Hopf extended system."""
    u, p, q1, q2, omega = _unpack(x, n)
    q1_seed, q2_seed, _ = _seed(f, u_guess, p_guess, lax.stop_gradient(args), n)
    jac = jacfwd(f, argnums=0)(u, p, args)
    g1 = f(u, p, args)
    g2 = jac @ q1 + omega * q2
    g3 = jac @ q2 - omega * q1
    g4 = jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0
    g5 = jnp.dot(q1_seed, q2) - jnp.dot(q2_seed, q1)
    return jnp.concatenate([g1, g2, g3, jnp.reshape(g4, (1,)), jnp.reshape(g5, (1,))])


def hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Locate a Hopf point near ``(u_guess, p_guess)``, differentiable in ``args``.

    Returns ``(u*, p*, q1*, q2*, omega0*)``: the equilibrium, parameter,
    real and imaginary parts of the (unit) critical eigenvector, and the
    critical frequency.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _extended_residual(x, f, theta, n, u_guess, p_guess)

    def x0(theta):
        q1_0, q2_0, omega_0 = _seed(f, u_guess, p_guess, theta, n)
        return _pack(u_guess, p_guess, q1_0, q2_0, omega_0)

    x_star = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, q1, q2, omega = _unpack(x_star, n)
    return u, p, q1, q2, omega


def hopf_parameter(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Array:
    """
    Parameter value ``p*`` at the Hopf point -- a scalar, differentiable in ``args``.
    """
    _, p, _, _, _ = hopf_point(f, u_guess, p_guess, args, tol=tol, max_iter=max_iter)
    return p
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hopf_normal_form.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/bifurcations/hopf_normal_form.py tests/test_hopf_normal_form.py
git commit -m "feat: add differentiable hopf_point/hopf_parameter extended-system solver"
```

---

### Task 2: `lyapunov_coefficient` — first Lyapunov coefficient

**Files:**
- Modify: `src/jaxcont/bifurcations/hopf_normal_form.py`
- Test: `tests/test_hopf_normal_form.py`

**Interfaces:**
- Consumes: `hopf_point`'s output shape `(u, p, q1, q2, omega0)` (Task 1).
- Produces: `lyapunov_coefficient(f, u, p, q1, q2, omega0, args=None) -> Array` (scalar `l1`).
  Consumed by Task 3 (grad check), Task 4 (BK cross-check), Task 5 (`Hopf.refine()`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hopf_normal_form.py`:

```python
from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient


def test_lyapunov_coefficient_matches_exact_textbook_value():
    u, p, q1, q2, omega0 = hopf_point(
        _textbook_hopf, jnp.zeros(2), 0.05, tol=1e-10, max_iter=50,
    )
    l1 = lyapunov_coefficient(_textbook_hopf, u, p, q1, q2, omega0)
    assert jnp.isclose(float(l1), -1.0, atol=1e-4)


def test_lyapunov_coefficient_scales_linearly_with_cubic_coefficient():
    # dr/dt = mu*r + k*r^3 in polar form has l1 = k exactly by definition of
    # normal form -- scaling the cubic term by k must scale l1 by exactly k.
    def hopf_scaled(u, p, k):
        x, y = u[0], u[1]
        r2 = x**2 + y**2
        return jnp.array([-y + x * (p - k * r2), x + y * (p - k * r2)])

    for k in (1.0, 2.0, 0.5, -1.0):
        u, p, q1, q2, omega0 = hopf_point(
            hopf_scaled, jnp.zeros(2), 0.05, k, tol=1e-10, max_iter=50,
        )
        l1 = lyapunov_coefficient(hopf_scaled, u, p, q1, q2, omega0, k)
        assert jnp.isclose(float(l1), -k, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hopf_normal_form.py -v -k lyapunov_coefficient`
Expected: FAIL with `ImportError: cannot import name 'lyapunov_coefficient'`

- [ ] **Step 3: Write the implementation**

Append to `src/jaxcont/bifurcations/hopf_normal_form.py` (add `jvp` to the existing `from jax
import Array, jacfwd, lax` line, making it `from jax import Array, jacfwd, jvp, lax`):

```python
def lyapunov_coefficient(
    f: Callable[[Array, Array, PyTree], Array],
    u: Array,
    p: Array,
    q1: Array,
    q2: Array,
    omega0: Array,
    args: PyTree = None,
) -> Array:
    """
    First Lyapunov coefficient l1 at a Hopf point (u, p) with critical
    eigenvector q1 + i*q2 and frequency omega0 -- Kuznetsov's formula
    (Elements of Applied Bifurcation Theory, 3rd ed., eq. 3.19/10.16).
    l1 < 0 is supercritical (stable branching limit cycle), l1 > 0 is
    subcritical. Pure algebra (directional derivatives via jax.jvp plus two
    linear solves) -- no Newton iteration, differentiable wherever its
    inputs are, so composes with hopf_point's implicit-diff gradients via
    ordinary reverse-mode chain-rule composition; never calls
    jnp.linalg.eig itself.

    The left eigenvector p_left (Aᵀ p_left = -i*omega0*p_left, normalized
    so conj(p_left)@q == 1) is solved directly in complex arithmetic on the
    n x n matrix (Aᵀ + i*omega0*I) -- NOT via a real 2n x 2n block
    representation, which structurally always has repeated singular-value
    pairs and makes jnp.linalg.svd's gradient nan (found during design,
    see docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md).

    Verified: the standard supercritical textbook example (f = (-y + x(mu -
    x^2 - y^2), x + y(mu - x^2 - y^2)), exact l1 = -1 at the origin) and an
    independent BifurcationKit.jl v0.5.2 cross-check on a less-symmetric
    example -- see examples/BifurcationKit/04_hopf_normal_form.jl.
    """
    n = u.shape[0]
    complex_dtype = jnp.complex128 if u.dtype == jnp.float64 else jnp.complex64
    u_c = u.astype(complex_dtype)
    q = (q1 + 1j * q2).astype(complex_dtype)
    qbar = jnp.conj(q)

    A = jacfwd(lambda uu: f(uu, p, args))(u)
    Ac = A.astype(complex_dtype)
    eye_n = jnp.eye(n, dtype=complex_dtype)

    m_complex = Ac.T + 1j * omega0 * eye_n
    _, _, vh = jnp.linalg.svd(m_complex)
    p_left = jnp.conj(vh[-1, :])
    p_left = p_left / jnp.conj(jnp.vdot(p_left, q))

    def f_c(uu):
        return f(uu, p, args).astype(complex_dtype)

    def d1(uu, y):
        return jvp(f_c, (uu,), (y,))[1]

    def d2(uu, y, z):
        return jvp(lambda uu_: d1(uu_, y), (uu,), (z,))[1]

    def d3(uu, y, z, w):
        return jvp(lambda uu_: d2(uu_, y, z), (uu,), (w,))[1]

    B = lambda x, y: d2(u_c, x, y)
    C = lambda x, y, z: d3(u_c, x, y, z)

    term1 = jnp.vdot(p_left, C(q, q, qbar))
    a_inv_b = jnp.linalg.solve(Ac, B(q, qbar))
    term2 = 2.0 * jnp.vdot(p_left, B(q, a_inv_b))
    m2 = 2j * omega0 * eye_n - Ac
    m2_inv_b = jnp.linalg.solve(m2, B(q, q))
    term3 = jnp.vdot(p_left, B(qbar, m2_inv_b))

    return jnp.real(term1 - term2 + term3) / (4.0 * omega0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hopf_normal_form.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/bifurcations/hopf_normal_form.py tests/test_hopf_normal_form.py
git commit -m "feat: add lyapunov_coefficient (Kuznetsov's first Lyapunov coefficient)"
```

---

### Task 3: Gradient check for `lyapunov_coefficient`

**Files:**
- Test: `tests/test_hopf_normal_form.py`

**Interfaces:**
- Consumes: `hopf_point`, `lyapunov_coefficient` (Tasks 1-2), unchanged.
- Produces: nothing new — this task is verification-only, closing the "grad-ready" claim from the
  design spec's delivery-mechanism decision.

- [ ] **Step 1: Write the test**

Append to `tests/test_hopf_normal_form.py`:

```python
def test_lyapunov_coefficient_grad_matches_finite_difference():
    # l1(scale) = -scale exactly (see the scaling test above), so
    # d(l1)/d(scale) = -1 exactly -- a strong, non-trivial gradient check.
    def hopf_scaled(u, p, scale):
        x, y = u[0], u[1]
        r2 = x**2 + y**2
        return jnp.array([-y + x * (p - scale * r2), x + y * (p - scale * r2)])

    def l1_of_scale(scale):
        u, p, q1, q2, omega0 = hopf_point(
            hopf_scaled, jnp.zeros(2), 0.05, scale, tol=1e-10, max_iter=50,
        )
        return lyapunov_coefficient(hopf_scaled, u, p, q1, q2, omega0, scale)

    grad = jax.grad(l1_of_scale)(1.0)
    eps = 1e-4
    fd = (l1_of_scale(1.0 + eps) - l1_of_scale(1.0 - eps)) / (2 * eps)
    assert jnp.isclose(grad, fd, atol=1e-2)
    assert jnp.isclose(grad, -1.0, atol=1e-2)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_hopf_normal_form.py -v -k grad`
Expected: 2 passed (this one plus Task 1's `test_hopf_parameter_grad_matches_finite_difference`)

If this fails with `NotImplementedError: derivatives of non-symmetric eigenvectors are not
supported`, the `lax.stop_gradient` in `_extended_residual` (Task 1, Step 3) was dropped or
misapplied — re-check it wraps the `args` passed into `_seed`, not the outer `theta` used
elsewhere in `G`. If it fails with a `nan` gradient, the left-eigenvector solve in
`lyapunov_coefficient` (Task 2) was changed back to the real `2n×2n` block form — it must use the
complex `n×n` form as written.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hopf_normal_form.py
git commit -m "test: verify lyapunov_coefficient gradient against finite differences"
```

---

### Task 4: Independent BifurcationKit.jl cross-check

**Files:**
- Create: `examples/BifurcationKit/04_hopf_normal_form.jl`
- Test: `tests/test_hopf_normal_form.py`

**Interfaces:**
- Consumes: `hopf_point`, `lyapunov_coefficient` (Tasks 1-2), unchanged.
- Produces: nothing new for other tasks — this is the spec's required independent verification,
  matching the existing `examples/BifurcationKit/02_lorenz84.jl`/`03_neural_mass.jl` pattern.

- [ ] **Step 1: Create the Julia cross-check script**

Create `examples/BifurcationKit/04_hopf_normal_form.jl`:

```julia
using BifurcationKit
const BK = BifurcationKit

# Asymmetric Hopf example used to cross-validate
# jaxcont.bifurcations.hopf_normal_form.lyapunov_coefficient (see
# docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md). Deliberately
# non-symmetric (unlike the standard textbook polar-coordinates example) so
# the cubic-form B-based correction terms in Kuznetsov's l1 formula are
# actually exercised, not identically zero.
function Fbp(u, p)
    x, y = u
    mu = p.mu
    return [mu*x - y + x^2 - x^3 - x*y^2,
            x + mu*y + x*y - y^3]
end

par = (mu = -1.0,)
u0 = [0.0, 0.0]
prob = BK.BifurcationProblem(Fbp, u0, par, (@optic _.mu))
opts = ContinuationPar(p_min = -2.0, p_max = 2.0, ds = 0.01, dsmax = 0.05,
                        n_inversion = 8, max_bisection_steps = 25, nev = 2)
br = continuation(prob, PALC(), opts; normC = norminf, bothside = true)

hopfidx = [i for (i, bp) in enumerate(br.specialpoint) if bp.type == :hopf]
for i in hopfidx
    hp = get_normal_form(prob, br, i)
    # BifurcationKit.jl's `b` plays the role of the amplitude-equation cubic
    # coefficient -- its own source (NormalForms.jl) calls it "the Lyapunov
    # coefficient" in an internal warning message. It equals 2*l1 in
    # Kuznetsov's normalization: verified against the exact-known textbook
    # example first (l1=-1 there; BK independently gives b=-2.0 exactly)
    # before trusting this relationship on this asymmetric example.
    println("Hopf at p=", hp.p, " omega0=", hp.ω, "  BK b=", real(hp.nf.b),
            "  l1 = b/2 = ", real(hp.nf.b) / 2)
end
```

- [ ] **Step 2: Run the script and confirm the reference value**

Run: `julia examples/BifurcationKit/04_hopf_normal_form.jl`
Expected output (BifurcationKit.jl v0.5.2, `julia -e 'using Pkg; Pkg.status()'` should show this
version installed):
```
Hopf at p=5.392241290723402e-7 omega0=1.0  BK b=-1.7500010784482587  l1 = b/2 = -0.8750005392241294
```
If the printed `p`/`b` differ from this at more than the ~1e-6 level, something about the
continuation options changed the located point — do not silently adjust the reference value in
the test below without first checking why.

- [ ] **Step 3: Write the failing test**

Append to `tests/test_hopf_normal_form.py`:

```python
def test_lyapunov_coefficient_matches_bifurcationkit_jl_independent_run():
    # Independent cross-check against BifurcationKit.jl v0.5.2's own hopf
    # normal form (examples/BifurcationKit/04_hopf_normal_form.jl, run
    # 2026-08-04): Hopf at p=5.392241290723402e-7, omega0=1.0, BK's own
    # normal-form b=-1.7500010784482587, giving l1=b/2=-0.8750005392241294.
    def f(u, p, args):
        x, y = u[0], u[1]
        return jnp.array([
            p * x - y + x**2 - x**3 - x * y**2,
            x + p * y + x * y - y**3,
        ])

    u, p, q1, q2, omega0 = hopf_point(f, jnp.zeros(2), 0.05, tol=1e-10, max_iter=50)
    l1 = lyapunov_coefficient(f, u, p, q1, q2, omega0)

    bk_l1_reference = -0.8750005392241294
    assert jnp.isclose(float(omega0), 1.0, atol=1e-6)
    assert jnp.isclose(float(l1), bk_l1_reference, atol=1e-4)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hopf_normal_form.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add examples/BifurcationKit/04_hopf_normal_form.jl tests/test_hopf_normal_form.py
git commit -m "test: cross-validate lyapunov_coefficient against BifurcationKit.jl v0.5.2"
```

---

### Task 5: Wire `hopf_point`/`lyapunov_coefficient` into `Hopf.refine()`

**Files:**
- Modify: `src/jaxcont/bifurcations/events.py`
- Modify: `tests/test_bifurcations.py`

**Interfaces:**
- Consumes: `jaxcont.bifurcations.hopf_normal_form.hopf_point`,
  `jaxcont.bifurcations.hopf_normal_form.lyapunov_coefficient` (Tasks 1-2).
- Produces: `Hopf` gains an `l1_tolerance: float = 1e-6` field; `Hopf.refine()`'s returned
  `EventHit.info` gains `"omega0"`, `"l1"`, `"criticality"` keys, and `"method"` becomes
  `"extended_system"` (was `"bisection"`).

- [ ] **Step 1: Update the existing test to the new behavior**

In `tests/test_bifurcations.py`, replace `test_hopf_refine_converges_to_exact_zero_via_three_way_bisection`
(currently at lines 72-90) with:

```python
def test_hopf_refine_converges_via_extended_system():
    # This rhs is purely linear in u, so its 2nd/3rd derivatives are zero
    # everywhere -- l1 must come out exactly 0 (degenerate), giving this
    # test double duty as the "degenerate" edge case from the design spec.
    def rhs(u, p):
        x, y = u[0], u[1]
        return jnp.array([p * x - 0.1 * y, 0.1 * x + p * y])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), -0.05))
    right = BranchPoint(p=0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), 0.05))
    hit = hopf.refine(left, right, (3, 4), rhs, tolerance=1e-8, max_iterations=50)
    assert hit.kind == "hopf"
    assert abs(hit.p) < 1e-6
    assert hit.info["method"] == "extended_system"
    assert jnp.isclose(hit.info["omega0"], 0.1, atol=1e-4)
    assert jnp.isclose(hit.info["l1"], 0.0, atol=1e-4)
    assert hit.info["criticality"] == "degenerate"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_bifurcations.py::test_hopf_refine_converges_via_extended_system -v`
Expected: FAIL (old `refine()` still returns `method="bisection"`, no `omega0`/`l1`/`criticality`
keys)

- [ ] **Step 3: Rewrite `Hopf.refine()`**

In `src/jaxcont/bifurcations/events.py`:

1. Add the import (near the top, alongside the existing `from jaxcont.bifurcations.fold_solve
   import fold_point`):

```python
from jaxcont.bifurcations.hopf_normal_form import hopf_point, lyapunov_coefficient
```

2. Add `l1_tolerance` to the `Hopf` dataclass (currently `kind: str = "hopf"` and
   `tolerance: float = 1e-6`):

```python
@dataclass(frozen=True)
class Hopf(Event):
    """..."""  # existing docstring unchanged

    kind: str = "hopf"
    tolerance: float = 1e-6
    l1_tolerance: float = 1e-6
```

3. Replace the entire `refine` method body (currently the bisection loop) with:

```python
    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        u, p, q1, q2, omega0 = hopf_point(
            lambda u, p, _args: rhs(u, p), u_guess, p_guess,
            tol=tolerance, max_iter=max_iterations,
        )
        l1 = lyapunov_coefficient(lambda u, p, _args: rhs(u, p), u, p, q1, q2, omega0)
        if abs(l1) < self.l1_tolerance:
            criticality = "degenerate"
        else:
            criticality = "supercritical" if l1 < 0 else "subcritical"
        return EventHit(
            kind="hopf", p=float(p), u=u, index=index,
            info={"omega0": float(omega0), "l1": float(l1),
                  "criticality": criticality, "method": "extended_system"},
        )
```

(The `lambda u, p, _args: rhs(u, p)` adapter matches `Fold.refine()`'s existing pattern just above
in the same file — `detect_events`'s generic `rhs` is 2-arg, `hopf_point` wants the 3-arg
`f(u,p,args)` shape. `Hopf`'s existing class docstring does not mention "bisection" anywhere, so it
needs no update.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bifurcations.py -v`
Expected: all pass, including the rewritten test. Also run the full suite to check nothing else in
`events.py`'s test coverage broke:

Run: `pytest tests/ -v`
Expected: all pass (aside from anything already `slow`/`gpu`-marked and deselected by default,
matching the project's normal `pytest` invocation).

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/bifurcations/events.py tests/test_bifurcations.py
git commit -m "feat: wire hopf_point/lyapunov_coefficient into Hopf.refine()"
```

---

### Task 6: Regression check — `example_02_lorenz.py` / `example_05_neural_mass.py`

**Files:**
- None modified (verification-only task). If it uncovers a mismatch, stop and report — do not
  silently edit `bk_reference` tables to make the diff go away without understanding why the
  location moved.

**Interfaces:**
- Consumes: the rewritten `Hopf.refine()` from Task 5.
- Produces: nothing new — confirms the regression risk flagged in the design spec ("Hopf
  locations may shift slightly now that `u` is a converged equilibrium rather than an
  interpolation") did not break the already-published, cross-validated BifurcationKit.jl
  comparison numbers in these two example scripts.

- [ ] **Step 1: Run `example_02_lorenz.py` headless and inspect the comparison table**

Run: `MPLBACKEND=Agg python examples/example_02_lorenz.py 2>&1 | tail -20`
Expected: exit code 0, and the printed `JaxCont / BifurcationKit.jl (reference)` table shows a
`<-> hopf F=...` or `<-> bp/fold F=...` match on every line — none should say
`(no close match; see note above)`. There should be exactly 3 `hopf` matches and 1 `bp/fold` match
(per the script's existing `bk_reference` list: `1.546648` fold, `1.619658`/`2.467222`/`2.859876`
hopf).

- [ ] **Step 2: Run `example_05_neural_mass.py` headless and inspect its comparison table**

Run: `MPLBACKEND=Agg python examples/example_05_neural_mass.py 2>&1 | tail -20`
Expected: exit code 0, all printed bifurcations show a `<->` match against that script's own
`bk_reference` table (read the script first to see its exact reference values and expected count —
do not assume it matches example_02's).

- [ ] **Step 3: If either script shows a mismatch**

Stop. Read the printed JaxCont parameter value against the `bk_reference` list in that script,
check whether it's still within the existing `abs(r[1] - bif["parameter"]) < 0.01` tolerance the
script already uses. If it's genuinely outside tolerance, this is a real finding — report it rather
than loosening the tolerance or editing the reference table to hide it. (Based on the design
spec's own reasoning, a shift here is expected to be small — `u` moving from a linear
interpolation to a converged equilibrium — and likely to *improve* agreement, not break it; but
this must be confirmed by running the scripts, not assumed.)

- [ ] **Step 4: Record the result**

No commit needed for this task if both scripts pass cleanly — this is a verification gate, not a
code change. If you needed to fix something in Task 5 as a result of this check, go back and amend
that task's commit-worthy diff, then re-run Steps 1-2 here to confirm the fix worked, before moving
to Task 7.

---

### Task 7: Export `hopf_point`/`hopf_parameter`/`lyapunov_coefficient` from `jaxcont`

**Files:**
- Modify: `src/jaxcont/__init__.py`
- Test: `tests/test_functional_api.py` (or create `tests/test_hopf_normal_form.py`'s final test —
  either location is fine; this plan adds it to the existing normal-form test file for locality)

**Interfaces:**
- Consumes: `hopf_point`, `hopf_parameter`, `lyapunov_coefficient` (Tasks 1-2), unchanged.
- Produces: `jaxcont.hopf_point`, `jaxcont.hopf_parameter`, `jaxcont.lyapunov_coefficient` as
  top-level public names.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hopf_normal_form.py`:

```python
import jaxcont as jc


def test_hopf_normal_form_functions_are_exported_at_top_level():
    assert jc.hopf_point is hopf_point
    assert jc.hopf_parameter is hopf_parameter
    assert jc.lyapunov_coefficient is lyapunov_coefficient
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_hopf_normal_form.py -v -k exported`
Expected: FAIL with `AttributeError: module 'jaxcont' has no attribute 'hopf_point'`

- [ ] **Step 3: Update `src/jaxcont/__init__.py`**

Find the existing block:
```python
# Differentiable fold solver (reverse-mode grad of a fold location via the
# implicit function theorem -- see examples/example_07_differentiable.py)
from jaxcont.bifurcations.fold_solve import fold_point, fold_parameter
```
and add directly below it:
```python

# Differentiable Hopf-point solver + first Lyapunov coefficient (Hopf
# criticality) -- see docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md
from jaxcont.bifurcations.hopf_normal_form import (
    hopf_point, hopf_parameter, lyapunov_coefficient,
)
```

Then in the `__all__` list, find:
```python
    "fold_point",
    "fold_parameter",
```
and add directly below it:
```python
    "hopf_point",
    "hopf_parameter",
    "lyapunov_coefficient",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hopf_normal_form.py -v`
Expected: 9 passed

Then run the full suite one more time to confirm the `__init__.py` change didn't break any
existing import-surface test:

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/__init__.py tests/test_hopf_normal_form.py
git commit -m "feat: export hopf_point/hopf_parameter/lyapunov_coefficient from jaxcont"
```

---

## Post-plan follow-up (not part of this plan, for the roadmap)

After all 7 tasks land, update `notes/ROADMAP.md`'s v0.3.0+ section to check off "Normal forms /
Lyapunov coefficient l1 (Hopf criticality)" and add a dated entry describing what shipped, matching
every other roadmap entry's style (what was built, what was found/fixed during verification, what
was explicitly left out — the fold coefficient `a`, `jc.normal_form(sol, event)` dispatcher, GH
detection). This plan does not include that edit as a task since it's documentation bookkeeping
that happens once the real work is confirmed merged, not before.
