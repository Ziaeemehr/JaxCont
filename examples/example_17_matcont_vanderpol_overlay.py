r"""
Van der Pol Hopf comparison with MatCont
=========================================

The Van der Pol equilibrium remains at the origin while its stability changes
at the Hopf point.  That makes the spectral crossing more informative than the
geometrically trivial equilibrium branch: the lower panel shows the largest
real part of the eigenvalues passing through zero.

This page runs JaxCont's registered ``MC-EQ-002`` validation case and compares
it with reviewed MatCont 7.6 CSV artifacts.  MATLAB is not required at runtime.
"""

# %%
# Generate the branch and stability overlays
# ------------------------------------------
# Blue curves are freshly computed by JaxCont and orange open circles are the
# committed MatCont reference mesh.  The circular and cross markers identify
# the Hopf event reported independently by each solver.

import sys
from pathlib import Path

# sphinx-gallery executes this script without defining __file__ (its exec
# namespace deliberately omits it -- see sphinx_gallery.gen_rst), and in that
# context conf.py has already put the repository root on sys.path. Only
# standalone execution (``python examples/example_17_...py``) needs this.
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

from examples.MatCont.visualize import render_equilibrium_overlay

figure = render_equilibrium_overlay(
    "MC-EQ-002",
    output_path=Path("images") / "matcont_vanderpol_overlay.png",
    parameter_name=r"$\mu$",
    state_name=r"$x$",
    title="Van der Pol Hopf: JaxCont vs MatCont 7.6",
)

# %%
# Reading the stability crossing
# ------------------------------
# The equilibrium coordinates are zero throughout this continuation, so the
# upper branch overlay is deliberately unremarkable.  The lower panel supplies
# the useful evidence: both solvers locate the spectral-abscissa zero crossing
# at the Hopf bifurcation.  The PASS annotation is only added after the
# registered numerical comparison succeeds.

print(
    "Saved JaxCont/MatCont Van der Pol overlay to "
    "images/matcont_vanderpol_overlay.png"
)
plt.show()
