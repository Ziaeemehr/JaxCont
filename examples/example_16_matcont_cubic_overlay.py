r"""
Visual comparison with MatCont
==============================

Numerical tolerances are essential, but an overlay gives a faster first check:
do two continuation packages recover the same branch geometry and the same
bifurcations?

This example runs JaxCont's registered ``MC-EQ-001`` validation case and plots
its cubic S-curve directly over the reviewed MatCont 7.6 branch. It does not
need MATLAB at runtime: the normalized MatCont CSV files are committed under
``examples/MatCont/reference`` with provenance and integrity metadata.

The equation is

.. math::

    0 = r + x - \frac{x^3}{3},

with folds at :math:`(r,x)=(2/3,-1)` and :math:`(-2/3,1)`.
"""

# %%
# Generate the overlay
# --------------------
# ``render_case_overlay`` calls the same registered Python case used by the
# systematic MatCont validation CLI. The blue curve is freshly computed by
# JaxCont; orange open circles come from the reviewed MatCont artifact.

import sys
from pathlib import Path

# sphinx-gallery executes this script without defining __file__ (its exec
# namespace deliberately omits it -- see sphinx_gallery.gen_rst), and in that
# context conf.py has already put the repository root on sys.path. Only
# standalone execution (``python examples/example_16_...py``) needs this.
try:
    _repository_root = Path(__file__).resolve().parents[1]
except NameError:
    _repository_root = None
if _repository_root is not None and str(_repository_root) not in sys.path:
    sys.path.insert(0, str(_repository_root))

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

try:
    import examples.MatCont  # noqa: F401
except ModuleNotFoundError:
    # A notebook downloaded from the docs gallery and run standalone (e.g.
    # uploaded to Google Colab) has no repository checkout, so
    # `examples.MatCont` isn't importable. Fetch just that package --
    # including its committed reference/ data, which needs no MATLAB/MatCont
    # install to read -- using only the standard library, since neither svn
    # nor git is guaranteed to be present in a notebook runtime.
    import io
    import urllib.request
    import zipfile

    with urllib.request.urlopen(
        "https://github.com/Ziaeemehr/JaxCont/archive/refs/heads/main.zip"
    ) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    prefix = next(
        name for name in archive.namelist() if name.endswith("/examples/MatCont/")
    )
    for member in archive.namelist():
        if member.startswith(prefix) and not member.endswith("/"):
            target = Path("examples/MatCont") / member[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))

    sys.path.insert(0, str(Path.cwd()))

from examples.MatCont.visualize import render_case_overlay

output_path = Path("images") / "matcont_cubic_overlay.png"
figure = render_case_overlay(
    "MC-EQ-001",
    output_path=output_path,
    parameter_name=r"$r$",
    state_name=r"$x$",
    title="Cubic S-curve: JaxCont vs MatCont 7.6",
)

# %%
# How to read the figure
# ----------------------
# JaxCont and MatCont use different adaptive meshes, so individual sample
# points are not expected at identical parameter values. Agreement means that
# the independently sampled curves lie on top of one another and that both
# packages place the two limit points (``LP``) at the same locations.
#
# The automated validation remains authoritative: it compares monotone branch
# segments by interpolation, requires unique event matches, checks stability
# and spectra, and applies the tolerances declared in ``cases.json``. This
# figure is a transparent visual companion to those numerical checks.

print(f"Saved JaxCont/MatCont overlay to {output_path}")
plt.show()
