# JaxCont

**Vectorize whole continuation sweeps with `jax.vmap`, and differentiate
bifurcation locations with `jax.grad`.**

[![Test](https://github.com/Ziaeemehr/JaxCont/actions/workflows/tests.yml/badge.svg)](https://github.com/Ziaeemehr/JaxCont/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/jaxcont/badge/?version=latest)](https://jaxcont.readthedocs.io/latest/)
[![PyPI version](https://img.shields.io/pypi/v/jaxcont.svg)](https://pypi.org/project/jaxcont/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

JaxCont is an equilibrium continuation and bifurcation-analysis library built
around JAX transformations. Its whole-loop pseudo-arclength engine is a pure,
compiled computation: use `vmap` to compute ensembles of branches in one
batched kernel, `jacfwd` to differentiate through a sweep, or the implicit
`fold_parameter` solver for reverse-mode gradients of a fold location.

```python
# One compiled kernel computes a branch for every design value.
branches = jax.vmap(run_branch)(design_values)

# A fold location can participate in gradient-based inverse design.
dp_dtheta = jax.grad(
    lambda theta: jc.fold_parameter(f, u_guess, p_guess, theta)
)(theta)
```

The v0.1 series deliberately focuses on equilibria: natural and
pseudo-arclength continuation, fold and Hopf detection with refinement,
linear stability, and bifurcation diagrams. Periodic orbits, Floquet
multipliers, boundary-value problems, branch switching, and two-parameter
continuation are not part of the supported v0.1 API.

## Installation

JaxCont requires Python 3.9 or newer.

```bash
pip install jaxcont
```

For a development checkout:

```bash
git clone https://github.com/Ziaeemehr/JaxCont.git
cd JaxCont
python -m pip install -e ".[dev]"
```

JAX's platform-specific accelerator installation is documented in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick start

Continue the positive branch of `u² + p = 0` through its fold at `p = 0`:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxcont as jc

def saddle_node(u, p, args):
    return u**2 + p

problem = jc.bif_problem(
    saddle_node,
    u0=jnp.array([1.0]),
    p0=-1.0,
    state_names=["u"],
    param_name="p",
)
result = jc.continuation(
    problem,
    p_span=(-1.0, 0.2),
    settings=jc.ContinuationPar(ds=0.03, max_steps=200),
    events=[jc.Fold()],
)

print(result.branch.params)
print([(event.kind, event.p) for event in result.events])

result.plot(annotate=True, title="Saddle-node bifurcation")
plt.show()
```

`PseudoArclength(engine="scan")` is the default algorithm. It uses the
whole-loop compiled engine, computes stability in a vectorized post-pass, and
refines requested fold/Hopf events. Use `jc.Natural()` for natural-parameter
continuation or `jc.PseudoArclength(engine="legacy")` only when comparing with
the compatibility implementation.

See the [quickstart](https://jaxcont.readthedocs.io/en/latest/quickstart.html),
[Sphinx-Gallery examples](https://jaxcont.readthedocs.io/en/latest/auto_examples/index.html),
[`example_06_vmap_sweep.py`](examples/example_06_vmap_sweep.py), and
[`example_07_differentiable.py`](examples/example_07_differentiable.py) for the
full `vmap`, `jacfwd`, and inverse-design stories.

## Development

```bash
python -m pytest
make docs
python -m build
python -m twine check dist/*
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). The project
roadmap and supported scope live in [notes/ROADMAP.md](notes/ROADMAP.md).
The version-gated MatCont/BifurcationKit cross-validation plan and initial
MATLAB producers live in
[validation/VALIDATION_EXAMPLES.md](validation/VALIDATION_EXAMPLES.md).

## Citation

If JaxCont supports your research, cite the archived release using the DOI in
the GitHub/Zenodo release record. Citation metadata is also provided in
[`CITATION.cff`](CITATION.cff). Until the first archive is minted:

```bibtex
@software{ziaeemehr_jaxcont_2026,
  author  = {Ziaeemehr, Abolfazl},
  title   = {JaxCont: Differentiable Continuation and Bifurcation Analysis in JAX},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Ziaeemehr/JaxCont}
}
```

JaxCont is distributed under the [MIT License](LICENSE).
