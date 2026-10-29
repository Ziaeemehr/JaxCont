r"""
Lorenz-84: folds, Hopf curves and a Bogdanov–Takens point
=============================================================

The system
----------
This extended Lorenz atmosphere model has states :math:`X,Y,Z,U`:

.. math::
   \dot X &= -Y^2-Z^2-\alpha X+\alpha F-\gamma U^2,\\
   \dot Y &= XY-\beta XZ-Y+G,\\
   \dot Z &= \beta XY+XZ-Z,\\
   \dot U &= -\delta U+\gamma UX+T.

We fix :math:`(\alpha,\beta,\gamma,\delta,G)=(0.25,1,0.987,1.04,0.25)`.
The equilibrium branch varies :math:`F` at :math:`T=0.04`; the fold and
Hopf curves vary both :math:`F,T`. This is the four-state extension used in
MatCont's `numerical bifurcation tutorial, section 7.1
<https://webspace.science.uu.nl/~kouzn101/NMB_tutorial.pdf>`_,
which extends Lorenz's low-order atmosphere model.
The symmetry :math:`(U,T)\mapsto(-U,-T)` reflects solutions across :math:`T=0`;
we select the positive-:math:`T` Bogdanov–Takens point.
"""

# %%
# What to look for
# ----------------
# The F-branch turns at LP, where a real eigenvalue passes through zero.
# Its returning sheet contains three H points, where a conjugate eigenvalue
# pair crosses the imaginary axis. The spectral abscissa panel shows
# stability: negative means stable, positive means unstable. These sheets
# are already unstable, so a Hopf crossing changes the unstable directions
# without forcing the largest real eigenvalue through zero. Line styles
# identify tools throughout; they do not encode stability.
#
# In the parameter plane the fold and Hopf curves meet at BT, a double-zero
# eigenvalue with a single eigenvector. The Hopf frequency tends to zero
# there: beyond BT the extended chart describes a neutral saddle rather than
# an oscillatory instability, so the physical Hopf curve ends. On the fold
# curve, ZH combines a zero eigenvalue with an imaginary pair; CP is a cusp,
# where the quadratic fold coefficient vanishes. The green curve's BT label
# is BifurcationKit's unrefined guess; the refined BT and its two small
# eigenvalues are compared separately below. The refined-point panel
# magnifies their splitting: at a defective double zero, round-off may
# separate eigenvalues along real or imaginary directions. This splitting
# alone does not indicate a different bifurcation.

import argparse
import sys
import tempfile
from pathlib import Path

if '__file__' in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=Path(tempfile.gettempdir()) / 'jaxcont' / 'generated')
options = parser.parse_args() if __name__ == '__main__' and '__file__' in globals() else parser.parse_args([])
# The reviewed Newton controls require double precision in either gallery mode.
import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import jaxcont as jc
from examples.validation.cases.systems import lorenz84 as model
from examples.validation.cases.python_cases.lorenz84 import equilibrium_tables, fold_tables, run
from examples.validation.reporting import make_report, save_checked_figure, combine_reports
from examples.validation.visualize import (
    ReportTable, render_comparison, render_lorenz_plane, summarize_report,
)
import examples.validation as validation_package

reference_dir = Path(validation_package.__file__).resolve().parent / 'reference'

# %%
# The JaxCont computation
# -----------------------
# The shared model above is transcribed in ``model.lorenz84_rhs``. Correct
# the prescribed seed, continue both directions, and refine native LP/H
# events in the table adapter. These are fresh solves, independent of the
# committed external solutions.

seed = jnp.array(model.F_SWEEP['seed_state'])
for _ in range(5):
    seed -= jnp.linalg.solve(model.lorenz84_jacobian(seed, 3.),
                            model.lorenz84_rhs(seed, 3., model.FIXED_ARGS))
problem = jc.bif_problem(model.lorenz84_rhs, u0=seed, p0=3., args=model.FIXED_ARGS)
settings = jc.ContinuationPar(ds=.002, ds_max=.05, max_steps=200,
                              newton_tol=1e-12, theta=1.)
equilibria = {
    name: jc.continuation(problem, p_span=(3., end), settings=settings,
                          events=[jc.Fold(), jc.Hopf()])
    for name, end in [('decreasing', -1.5), ('increasing', 3.0000001)]
}
equilibrium = equilibrium_tables(equilibria)
fold_seed = model.select_event(equilibrium[1], 'LP', model.EVENT_WINDOWS['fold_seed'])
hopf_seed = model.select_event(equilibrium[1], 'H', model.EVENT_WINDOWS['hopf_seed'])
curve_settings = jc.ContinuationPar(ds=.001, ds_min=1e-5, ds_max=.015,
    max_steps=400, theta=1., newton_tol=1e-12, compute_stability=False)
fold_problem = jc.fold_curve_problem(model.lorenz84_ft_rhs,
    jnp.array([fold_seed[f'state_{i}'] for i in range(4)]),
    jnp.array([fold_seed['F'], .04]), free=1, formulation='extended',
    tol=1e-12, args=model.CONSTANTS)
fold = jc.continuation(fold_problem, p_span=(.04, -.01), settings=curve_settings,
    events=[jc.BogdanovTakens(raw_f=model.lorenz84_ft_rhs, free=1, curve='fold', args=model.CONSTANTS),
            jc.ZeroHopf(raw_f=model.lorenz84_ft_rhs, free=1, curve='fold', args=model.CONSTANTS),
            jc.Cusp(raw_f=model.lorenz84_ft_rhs, free=1, curve='fold', args=model.CONSTANTS)])
hopf_problem = jc.hopf_curve_problem(model.lorenz84_ft_rhs,
    jnp.array([hopf_seed[f'state_{i}'] for i in range(4)]),
    jnp.array([hopf_seed['F'], .04]), free=1, formulation='extended',
    tol=1e-12, args=model.CONSTANTS)
hopf = jc.continuation(hopf_problem, p_span=(.04, -.01), settings=curve_settings,
    events=[jc.BogdanovTakens(raw_f=model.lorenz84_ft_rhs, free=1, curve='hopf', args=model.CONSTANTS)])
bt_seed = next(e for e in fold.events if e.kind == 'bogdanov_takens'
               and .01 <= float(e.info['p'][1]) <= .03)
bt = jc.bogdanov_takens_point(model.lorenz84_ft_rhs, bt_seed.u,
    bt_seed.info['p'], args=model.CONSTANTS, tol=1e-12)
computed = dict(equilibrium=equilibrium, fold=fold_tables(fold), hopf=hopf, bt=bt)
# The adapter keeps only the Hopf side of BT and packages the refined point.
cases = [run('lorenz84.' + suffix, computed=computed)
         for suffix in ('eq-f', 'fold-ft', 'hopf-ft', 'bt-ft')]

# %%
# Comparison figures
# ------------------
# All external series are reviewed v2 bundles, read offline. The parameter
# plane overlays both curves and the independently refined BT; its error
# panels use the promotion comparator's sheet matching, coordinate choice
# and sparser native samples. Differences are divided by their unchanged
# registry budgets so the horizontal limit is shared across observables.
# Full native branches and labels are shown; acceptance uses the reviewed
# windows printed in each footer. LP lies just below the equilibrium
# matching window, and the Hopf curve boundary at BT is checked separately
# by the refined-point run. An unrefined curve BT remains pending evidence
# even when curves overlap.

reports = [make_report(case, reference_dir) for case in cases]
report = combine_reports(reports, case_id='lorenz84: eq-f / fold-ft / hopf-ft / bt-ft')
overlay, error = render_comparison(cases[0], reports[0], reference_dir)
plane = render_lorenz_plane(cases[1:], report, reference_dir)
fold_overlay, fold_error = render_comparison(cases[1], reports[1], reference_dir)
hopf_overlay, hopf_error = render_comparison(cases[2], reports[2], reference_dir)
bt_overlay, bt_error = render_comparison(cases[3], reports[3], reference_dir)
# sphinx_gallery_start_ignore
# The combined plane replaces the separate curve views. Point differences
# are in the report; a sampled branch error plot is uninformative for a point.
import matplotlib.pyplot as plt
for unused_figure in (fold_overlay, hopf_overlay, bt_error):
    plt.close(unused_figure)
# sphinx_gallery_end_ignore
for index, (figure, figure_report) in enumerate([
    (overlay, reports[0]), (error, reports[0]), (plane, report),
    (fold_error, reports[1]), (hopf_error, reports[2]), (bt_overlay, reports[3]),
]):
    save_checked_figure(figure, figure_report, options.output_dir / f'plot_lorenz84_{index}.png')

# %%
# Quantitative summary
# --------------------
# The same evidence drives the banners, per-metric verdicts and sidecars.
# Pending curve-event refinement is shown explicitly, without treating a
# nearby unrefined guess as a successful event comparison.

ReportTable(report)

# %%
# Takeaway
# --------

print(summarize_report(report))

# %%
# The overlays show how the equilibrium instabilities connect to the fold
# and Hopf curves in the parameter plane. Curve-event guesses remain a
# limitation of the comparison, while the separately refined BT provides
# independent evidence about the meeting point and its degeneracy.
