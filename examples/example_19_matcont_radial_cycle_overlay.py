r"""
Radial periodic orbit comparison with MatCont
==============================================

The analytic polar normal form

.. math::

    r' = r(\rho-r^2), \qquad \theta' = 1

has a periodic orbit for positive :math:`\rho`.  Its expected radius is
:math:`\sqrt{\rho}`, its period is :math:`2\pi`, and its nontrivial Floquet
multiplier is :math:`\exp(-4\pi\rho)`.

This page runs JaxCont's registered ``MC-LC-001`` case and compares those
three observables against the reviewed MatCont 7.6 adaptive-mesh artifacts.
MATLAB is not required at runtime.
"""

# %%
# Generate the periodic-orbit comparison
# ---------------------------------------
# The top panel shows the state-0 minimum/maximum envelope, the middle panel
# compares periods, and the lower panel removes the one trivial Floquet
# multiplier nearest +1 before comparing the remaining modulus.

import sys
from pathlib import Path

# sphinx-gallery executes this script without defining __file__ (its exec
# namespace deliberately omits it -- see sphinx_gallery.gen_rst), and in that
# context conf.py has already put the repository root on sys.path. Only
# standalone execution (``python examples/example_19_...py``) needs this.
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

from examples.MatCont.visualize import render_periodic_overlay

figure = render_periodic_overlay(
    "MC-LC-001",
    output_path=Path("images") / "matcont_radial_cycle_overlay.png",
    parameter_name=r"$\rho$",
    title="Radial periodic orbit: JaxCont vs MatCont 7.6",
)

print(
    "Saved JaxCont/MatCont radial-cycle overlay to "
    "images/matcont_radial_cycle_overlay.png"
)
plt.show()
