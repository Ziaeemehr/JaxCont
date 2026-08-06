r"""
PRC/dPRC validation via an independent shooting method
=========================================================

``examples/MatCont``'s ``MC-PRC-001`` case cross-validates :func:`prc_curve`
against real MatCont 7.6 output, but MatCont's own "dPRC" turned out to be a
time-derivative (:math:`d(\mathrm{PRC})/dt`), not the parameter derivative
:func:`dprc_curve` computes (:math:`d(\mathrm{PRC})/dp`) -- so dPRC has no
MatCont cross-check. This script closes that gap with a genuinely
independent *second numerical method*: classical IVP shooting (the textbook
way most PRC software computes it), rather than another collocation-based
tool.

**Method.** Given a periodic orbit already located by JaxCont's own
``periodic_orbit_problem`` (this script validates the *adjoint/PRC
computation*, not periodic-orbit-finding -- the same scope boundary
``MC-PRC-001`` used against MatCont), integrate the fundamental-matrix
variational equation :math:`\dot\Phi = J(x(t))\Phi` forward one period with
``scipy.integrate.solve_ivp`` to get the monodromy matrix :math:`\Phi(T)`
independently of collocation. The periodic adjoint seed :math:`Z(0)` is
:math:`\Phi(T)^\top`'s eigenvector for eigenvalue 1, normalized
:math:`Z(0)\cdot f(x_0,p)=\omega`; the rest of the period comes from
integrating the adjoint equation :math:`\dot Z = -J(t)^\top Z` *backward*.
dPRC is a central finite difference of this whole shooting computation
across :math:`p\pm\varepsilon` (using JaxCont's own re-converged
:math:`x_0(p)` at each perturbed :math:`p`, the same construction
:func:`dprc_curve`'s own unit test uses).

This is a one-off validation script (not wired into ``tests/``/CI) -- see
``docs/superpowers/specs/2026-08-05-prc-dprc-design.md``.
"""

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.signal import find_peaks

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.stability.prc import dprc_curve, prc_curve


def shooting_prc(rhs, jac, x0, T, args, n_out=400):
    """Shooting-based iPRC via the adjoint method (see module docstring).
    Returns ``(t_out, Z_out, Phi_T)``, ``Z_out`` shape ``(n_out, n)``."""
    n = len(x0)

    def state_and_fundamental_rhs(t, y):
        x = y[:n]
        Phi = y[n:].reshape(n, n)
        dx = np.asarray(rhs(t, x, *args))
        dPhi = jac(t, x, *args) @ Phi
        return np.concatenate([dx, dPhi.flatten()])

    y0 = np.concatenate([x0, np.eye(n).flatten()])
    sol_fwd = solve_ivp(
        state_and_fundamental_rhs, (0, T), y0,
        rtol=1e-12, atol=1e-14, method="DOP853", dense_output=True,
    )
    Phi_T = sol_fwd.y[n:, -1].reshape(n, n)

    eigvals, eigvecs = eig(Phi_T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    Z0 = np.real(eigvecs[:, idx])
    f0 = np.asarray(rhs(0.0, x0, *args))
    omega = 2 * np.pi / T
    Z0 = Z0 * (omega / (Z0 @ f0))

    def adjoint_rhs_reverse(tau, Z):
        # dZ/dt = -J(t)^T Z, integrated in reverse time tau = T - t, so
        # dZ/dtau = +J(T-tau)^T Z; x(t) via solve_ivp's own dense
        # (continuous) interpolant -- a coarse manual re-interpolation of a
        # handful of saved points was verified during prototyping to
        # dominate the error otherwise (~2.6e-2 vs ~6e-13 with dense_output).
        t = T - tau
        return jac(t, sol_fwd.sol(t)[:n], *args).T @ Z

    tau_eval = np.linspace(0, T, n_out)
    sol_bwd = solve_ivp(
        adjoint_rhs_reverse, (0, T), Z0, t_eval=tau_eval,
        rtol=1e-12, atol=1e-14, method="DOP853",
    )
    t_out = T - tau_eval[::-1]
    Z_out = sol_bwd.y.T[::-1]
    return t_out, Z_out, Phi_T


def _resample_to_mesh(t_out, Z_out, ntst, T):
    t_mesh = np.arange(ntst) * T / ntst
    return np.stack(
        [np.interp(t_mesh, t_out, Z_out[:, k]) for k in range(Z_out.shape[1])], axis=1
    )


# %%
# Part 1: sheared circle system -- a closed-form triple-check
# --------------------------------------------------------------
# ``r' = r(rho - r^2), theta' = omega + c(rho - r^2)`` (c != 0, unlike
# ``tests/test_prc.py``'s c=0 circle system) has logarithmic-spiral
# isochrons (Guckenheimer 1975): asymptotic phase
# ``phi(r,theta) = theta - c*ln(r/sqrt(rho))``, giving on the limit cycle
# (r=sqrt(rho)):
#
# .. math::
#     Z(\theta) = \frac{1}{\sqrt{\rho}}
#     \begin{pmatrix} -(\sin\theta + c\cos\theta) \\ \cos\theta - c\sin\theta \end{pmatrix}
#
# (derived and verified during this script's own design: matches shooting
# to 1.7e-12). This gives THREE independent computations of the same
# curve -- closed form, JaxCont's collocation (``prc_curve``), and
# shooting -- agreeing with each other is much stronger evidence than any
# pair alone.

C_SHEAR = 0.4
OMEGA = 1.0


def sheared_circle_rhs(u, p, args):
    x, y = u[0], u[1]
    rho = p
    rad = rho - (x * x + y * y)
    theta_dot = OMEGA + C_SHEAR * rad
    return jnp.array([rad * x - theta_dot * y, rad * y + theta_dot * x])


def _sheared_circle_rhs_np(t, u, rho):
    x, y = u
    rad = rho - (x * x + y * y)
    theta_dot = OMEGA + C_SHEAR * rad
    return [rad * x - theta_dot * y, rad * y + theta_dot * x]


def _sheared_circle_jac_np(t, u, rho):
    x, y = u
    rad = rho - (x * x + y * y)
    theta_dot = OMEGA + C_SHEAR * rad
    df1dx = rad - 2 * x * x - y * (C_SHEAR * -2 * x)
    df1dy = -2 * x * y - theta_dot - y * (C_SHEAR * -2 * y)
    df2dx = -2 * x * y + theta_dot + x * (C_SHEAR * -2 * x)
    df2dy = rad - 2 * y * y + x * (C_SHEAR * -2 * y)
    return np.array([[df1dx, df1dy], [df2dx, df2dy]])


rng = np.random.default_rng(0)
t_traj = np.sort(rng.uniform(0, 5.5, size=40))
t_traj[0] = 0.0
theta_guess = 2 * np.pi * t_traj / 5.5 + 0.3
u_traj = jnp.asarray(np.stack([0.8 * np.cos(theta_guess), 0.8 * np.sin(theta_guess)], axis=1))
mesh_circle = Collocation(ntst=10, ncol=4)
rho0 = 1.0
prob_circle = periodic_orbit_problem(
    sheared_circle_rhs, u_traj, jnp.asarray(t_traj), 2 * np.pi / OMEGA, rho0, mesh_circle
)

with jax.default_matmul_precision("float32"):
    Z_jaxcont_circle = np.asarray(prc_curve(sheared_circle_rhs, mesh_circle, prob_circle.u0, prob_circle.p0))
    dZ_jaxcont_circle = np.asarray(dprc_curve(prob_circle))

n = 2
mesh_states = np.asarray(prob_circle.u0[: mesh_circle.ntst * n]).reshape(mesh_circle.ntst, n)
theta_mesh = np.arctan2(mesh_states[:, 1], mesh_states[:, 0])
Z_closed = np.stack(
    [-(np.sin(theta_mesh) + C_SHEAR * np.cos(theta_mesh)), np.cos(theta_mesh) - C_SHEAR * np.sin(theta_mesh)],
    axis=1,
) / np.sqrt(rho0)

T_circle = float(prob_circle.u0[-1])
t_shoot, Z_shoot_circle, _ = shooting_prc(
    _sheared_circle_rhs_np, _sheared_circle_jac_np, mesh_states[0], T_circle, (rho0,)
)
Z_shoot_at_mesh_circle = _resample_to_mesh(t_shoot, Z_shoot_circle, mesh_circle.ntst, T_circle)

print("Sheared circle system (rho=1, c=0.4):")
print("  JaxCont vs closed form,   max |err| =", np.max(np.abs(Z_jaxcont_circle - Z_closed)))
print("  closed form vs shooting,  max |err| =", np.max(np.abs(Z_closed - Z_shoot_at_mesh_circle)))
print("  JaxCont vs shooting,      max |err| =", np.max(np.abs(Z_jaxcont_circle - Z_shoot_at_mesh_circle)))

# dPRC: finite difference of the shooting method across rho +/- eps, using
# JaxCont's own re-converged x0(rho) at each perturbed rho (same
# construction dprc_curve's own unit test uses against the closed form).
eps = 0.01


def _build_circle(rho):
    return periodic_orbit_problem(sheared_circle_rhs, u_traj, jnp.asarray(t_traj), 2 * np.pi / OMEGA, rho, mesh_circle)


def _shoot_Z_at_mesh_circle(prob, rho):
    ms = np.asarray(prob.u0[: mesh_circle.ntst * n]).reshape(mesh_circle.ntst, n)
    T = float(prob.u0[-1])
    t_out, Z_out, _ = shooting_prc(_sheared_circle_rhs_np, _sheared_circle_jac_np, ms[0], T, (rho,))
    return _resample_to_mesh(t_out, Z_out, mesh_circle.ntst, T)


Z_hi_circle = _shoot_Z_at_mesh_circle(_build_circle(rho0 + eps), rho0 + eps)
Z_lo_circle = _shoot_Z_at_mesh_circle(_build_circle(rho0 - eps), rho0 - eps)
dZ_shoot_fd_circle = (Z_hi_circle - Z_lo_circle) / (2 * eps)
print("  dprc_curve vs shooting finite-difference, max |err| =",
      np.max(np.abs(dZ_jaxcont_circle - dZ_shoot_fd_circle)))

# %%
# Part 2: Van der Pol -- a real system with no closed form
# ------------------------------------------------------------
# Same system/mu=1 starting point as ``example_10_van_der_pol_limit_cycle.py``.
# No analytic PRC exists here, so shooting is the sole independent
# reference against ``prc_curve``/``dprc_curve``.


def van_der_pol_rhs(u, p, args):
    x, y = u[0], u[1]
    mu = p
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


def _vdp_rhs_np(t, u, mu):
    x, y = u
    return [y, mu * (1.0 - x**2) * y - x]


def _vdp_jac_np(t, u, mu):
    x, y = u
    return np.array([[0.0, 1.0], [-2 * mu * x * y - 1.0, mu * (1.0 - x**2)]])


mu0 = 1.0
raw_solution = solve_ivp(
    _vdp_rhs_np, [0.0, 60.0], [0.5, 0.0], args=(mu0,), max_step=0.01, dense_output=True
)
tail_times = np.linspace(40.0, 60.0, 4000)
peak_times = tail_times[find_peaks(raw_solution.sol(tail_times)[0])[0]]
period_guess = float(np.mean(np.diff(peak_times)))
cycle_start_time = peak_times[-2]
t_trajectory = np.linspace(0.0, period_guess, 60, endpoint=False)
u_trajectory = jnp.asarray(raw_solution.sol(t_trajectory + cycle_start_time).T)
t_trajectory_j = jnp.asarray(t_trajectory)

mesh_vdp = Collocation(ntst=20, ncol=4)


def _build_vdp(mu):
    return periodic_orbit_problem(van_der_pol_rhs, u_trajectory, t_trajectory_j, period_guess, mu, mesh_vdp)


prob_vdp = _build_vdp(mu0)
with jax.default_matmul_precision("float32"):
    Z_jaxcont_vdp = np.asarray(prc_curve(van_der_pol_rhs, mesh_vdp, prob_vdp.u0, prob_vdp.p0))
    dZ_jaxcont_vdp = np.asarray(dprc_curve(prob_vdp))

mesh_states_vdp = np.asarray(prob_vdp.u0[: mesh_vdp.ntst * n]).reshape(mesh_vdp.ntst, n)
T_vdp = float(prob_vdp.u0[-1])
t_shoot_vdp, Z_shoot_vdp, _ = shooting_prc(_vdp_rhs_np, _vdp_jac_np, mesh_states_vdp[0], T_vdp, (mu0,))
Z_shoot_at_mesh_vdp = _resample_to_mesh(t_shoot_vdp, Z_shoot_vdp, mesh_vdp.ntst, T_vdp)

print("\nVan der Pol (mu=1):")
print("  JaxCont vs shooting PRC, max |err| =", np.max(np.abs(Z_jaxcont_vdp - Z_shoot_at_mesh_vdp)))

eps_vdp = 0.02


def _shoot_Z_at_mesh_vdp(prob, mu):
    ms = np.asarray(prob.u0[: mesh_vdp.ntst * n]).reshape(mesh_vdp.ntst, n)
    T = float(prob.u0[-1])
    t_out, Z_out, _ = shooting_prc(_vdp_rhs_np, _vdp_jac_np, ms[0], T, (mu,))
    return _resample_to_mesh(t_out, Z_out, mesh_vdp.ntst, T)


Z_hi_vdp = _shoot_Z_at_mesh_vdp(_build_vdp(mu0 + eps_vdp), mu0 + eps_vdp)
Z_lo_vdp = _shoot_Z_at_mesh_vdp(_build_vdp(mu0 - eps_vdp), mu0 - eps_vdp)
dZ_shoot_fd_vdp = (Z_hi_vdp - Z_lo_vdp) / (2 * eps_vdp)
print("  dprc_curve vs shooting finite-difference, max |err| =",
      np.max(np.abs(dZ_jaxcont_vdp - dZ_shoot_fd_vdp)))

# %%
# Plot: JaxCont (collocation) vs shooting, both systems
# ---------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].plot(theta_mesh, Z_jaxcont_circle[:, 0], "o", label="JaxCont prc_curve")
axes[0, 0].plot(theta_mesh, Z_shoot_at_mesh_circle[:, 0], "x", label="shooting")
theta_closed_loop = np.append(theta_mesh, theta_mesh[0])
Z_closed_loop = np.append(Z_closed[:, 0], Z_closed[0, 0])
axes[0, 0].plot(theta_closed_loop, Z_closed_loop, "-", alpha=0.5, label="closed form")
axes[0, 0].set_title("Sheared circle: iPRC (x-component)")
axes[0, 0].set_xlabel(r"$\theta$")
axes[0, 0].legend()

axes[0, 1].plot(theta_mesh, dZ_jaxcont_circle[:, 0], "o", label="JaxCont dprc_curve")
axes[0, 1].plot(theta_mesh, dZ_shoot_fd_circle[:, 0], "x", label="shooting finite-diff")
axes[0, 1].set_title("Sheared circle: dPRC (x-component)")
axes[0, 1].set_xlabel(r"$\theta$")
axes[0, 1].legend()

t_mesh_vdp = np.arange(mesh_vdp.ntst) * T_vdp / mesh_vdp.ntst
axes[1, 0].plot(t_mesh_vdp, Z_jaxcont_vdp[:, 0], "o", label="JaxCont prc_curve")
axes[1, 0].plot(t_mesh_vdp, Z_shoot_at_mesh_vdp[:, 0], "x", label="shooting")
axes[1, 0].set_title("Van der Pol: iPRC (x-component)")
axes[1, 0].set_xlabel("t")
axes[1, 0].legend()

axes[1, 1].plot(t_mesh_vdp, dZ_jaxcont_vdp[:, 0], "o", label="JaxCont dprc_curve")
axes[1, 1].plot(t_mesh_vdp, dZ_shoot_fd_vdp[:, 0], "x", label="shooting finite-diff")
axes[1, 1].set_title("Van der Pol: dPRC (x-component)")
axes[1, 1].set_xlabel("t")
axes[1, 1].legend()

for ax in axes.flat:
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=0)

plt.tight_layout()
plt.savefig(
    "images/example_14_prc_shooting_validation.png",
    dpi=180,
    bbox_inches="tight",
)
plt.show()
