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

from pathlib import Path

import matplotlib.pyplot as plt

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
