<p align="center">
  <img src="docs/images/logo/logo.svg" alt="JaxCont" width="340">
</p>

<p align="center"><strong>Differentiable continuation and bifurcation analysis in JAX.</strong></p>

<p align="center">
  <a href="https://github.com/Ziaeemehr/JaxCont/actions/workflows/tests.yml"><img src="https://github.com/Ziaeemehr/JaxCont/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://jaxcont.readthedocs.io/latest/"><img src="https://readthedocs.org/projects/jaxcont/badge/?version=latest" alt="Documentation"></a>
  <a href="https://pypi.org/project/jaxcont/"><img src="https://img.shields.io/pypi/v/jaxcont.svg" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
</p>

JaxCont turns an equilibrium continuation sweep into a JAX program. Follow
branches through folds with pseudo-arclength continuation, detect and refine
fold/Hopf events, and compose the analysis with `jax.jit`, `jax.vmap`, and
automatic differentiation.

## Why JaxCont?

- **Transform the whole analysis.** Batch ensembles of branches with `vmap`
  and differentiate bifurcation locations for inverse design.
- **Stay on the branch.** Adaptive pseudo-arclength continuation passes folds;
  natural continuation is available when the parameter remains regular.
- **See and validate the result.** Plot stability-aware bifurcation diagrams
  and eigenvalue trajectories with JaxCont events plus MatCont,
  BifurcationKit, or analytic reference markers.

The supported v0.1 surface focuses deliberately on equilibria. Periodic
orbits, branch switching, and two-parameter continuation remain future work.

## Installation

JaxCont requires Python 3.9 or newer.

```bash
pip install jaxcont
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
when selecting CPU, GPU, or TPU support.

## Quick start

Continue `u² + p = 0` through its fold at `p = 0`:

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

print([(event.kind, event.p) for event in result.events])
result.plot(annotate=True, title="Saddle-node bifurcation")
plt.show()
```

The same result can drive a stability-spectrum plot:

```python
from jaxcont.viz import plot_eigenvalues

plot_eigenvalues(result, shade_stability=True)
```

## Explore

- [Quickstart](https://jaxcont.readthedocs.io/en/latest/quickstart.html)
- [Example gallery](https://jaxcont.readthedocs.io/en/latest/auto_examples/index.html)
- [`example_03_van_der_pol.py`](examples/example_03_van_der_pol.py) — Hopf
  crossing with JaxCont and MatCont markers
- [`example_06_vmap_sweep.py`](examples/example_06_vmap_sweep.py) — batched
  continuation with `vmap`
- [`example_07_differentiable.py`](examples/example_07_differentiable.py) —
  differentiable bifurcation analysis

## Development

```bash
git clone https://github.com/Ziaeemehr/JaxCont.git
cd JaxCont
python -m pip install -e ".[dev]"
python -m pytest
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md), the
[roadmap](notes/ROADMAP.md), and the
[cross-validation plan](validation/VALIDATION_EXAMPLES.md).

## Citation

If JaxCont supports your research, cite the archived release using its
GitHub/Zenodo DOI. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
Until the first archive is minted:

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
