r"""
Adaptive-control Hopf comparison with MatCont
=============================================

The adaptive-control model also has a geometrically trivial equilibrium branch.
Its stability crossing carries the meaningful continuation information, so the
spectral-abscissa panel is more revealing than the state coordinates alone.

The page runs the registered ``MC-EQ-003`` JaxCont case against reviewed
MatCont 7.6 CSV data.  It uses no MATLAB runtime dependency.
"""

# %%
# Generate the branch and stability overlays
# ------------------------------------------
# The two solvers retain their native adaptive meshes.  Blue is the fresh
# JaxCont calculation; orange open circles are the reviewed MatCont artifact.

import sys
from pathlib import Path

_repository_root = Path(__file__).resolve().parents[1]
if str(_repository_root) not in sys.path:
    sys.path.insert(0, str(_repository_root))

import matplotlib.pyplot as plt

from examples.MatCont.visualize import render_equilibrium_overlay

figure = render_equilibrium_overlay(
    "MC-EQ-003",
    output_path=Path("images") / "matcont_adaptive_control_overlay.png",
    parameter_name=r"$\alpha$",
    state_name=r"$x_1$",
    title="Adaptive-control Hopf: JaxCont vs MatCont 7.6",
)

# %%
# Reading the stability crossing
# ------------------------------
# The upper panel confirms the common equilibrium branch, but the lower panel
# is the decisive visual comparison: it shows the leading eigenvalue's real
# part crossing zero at Hopf.  JaxCont's numerical comparison must pass before
# the figure is annotated with PASS.

print(
    "Saved JaxCont/MatCont adaptive-control overlay to "
    "images/matcont_adaptive_control_overlay.png"
)
plt.show()
