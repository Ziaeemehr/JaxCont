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

_repository_root = Path(__file__).resolve().parents[1]
if str(_repository_root) not in sys.path:
    sys.path.insert(0, str(_repository_root))

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from examples.MatCont.visualize import render_periodic_overlay

figure = render_periodic_overlay(
    "MC-LC-002",
    output_path=Path("images") / "matcont_torbpc_overlay.png",
    parameter_name=r"$\nu$",
    state_name=r"$x$ envelope",
    title="torBPC periodic branch: JaxCont and MatCont 7.6 diagnostic",
)
plt.show()
