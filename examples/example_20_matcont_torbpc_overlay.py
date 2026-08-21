r"""
torBPC periodic branch diagnostic with MatCont
===============================================

This page is intentionally diagnostic. It runs JaxCont's registered
``MC-LC-002`` case beside the reviewed MatCont 7.6 torBPC artifacts without
altering either solver's event locations, extrema, periods, or Floquet
multipliers. The automated validator currently fails this case, so the figure
exposes known mismatches rather than presenting the branches as agreement.

The envelope and period panels show actual detected JaxCont events separately
from the MatCont LPC, NS, and PD locations. In the multiplier plane,
``JaxCont near`` means the spectrum at the JaxCont branch point nearest the
corresponding MatCont event parameter; it does not mean JaxCont detected that
event there. MATLAB is not required at runtime.
"""

# %%
# Generate the torBPC diagnostic
# ------------------------------
# Event-specific colors connect the reference locations in the parameter
# panels to their MatCont and nearest-parameter JaxCont spectra.

import sys
from pathlib import Path

# sphinx-gallery executes this script without defining __file__ (its exec
# namespace deliberately omits it -- see sphinx_gallery.gen_rst), and in that
# context conf.py has already put the repository root on sys.path. Only
# standalone execution (``python examples/example_20_...py``) needs this.
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
    "MC-LC-002",
    output_path=Path("images") / "matcont_torbpc_overlay.png",
    parameter_name=r"$\nu$",
    state_name=r"$x$ envelope",
    title="torBPC periodic branch: JaxCont and MatCont 7.6 diagnostic",
)
plt.show()
