# Pinned BifurcationKit regeneration

The committed Project/Manifest pin BifurcationKit 0.8.8. Do not re-resolve the
Manifest. These headless producers do not load Plots or Revise and do not read
JaxCont solutions. Install/instantiate only through the separately opt-in
`setup.jl`; running a producer or a test never installs packages.

From the repository root:

```sh
source ~/envs/jaxcont/bin/activate
export JAX_PLATFORMS=cpu
export JULIA_BIN="$HOME/.juliaup/bin/julia"
export JULIA_DEPOT_PATH="$PWD/.julia-depot:$HOME/.julia"
python3 -m examples.validation.run_validation --regenerate bifurcationkit \
  --case lorenz84.eq-f --output-dir examples/validation/generated/candidates < /dev/null
```

Other implemented cases are `cubic.eq-fold`, `radial.cycle-floquet`,
`hopf-normal-form.l1`, `lorenz84.fold-ft`, `lorenz84.hopf-ft`, and
`lorenz84.bt-ft`. Scripts also accept `--case ID --output-dir EXACT_DIRECTORY`
directly. The CLI writes `<output-dir>/<case-id>/bifurcationkit/`; MATLAB's
Lorenz-84 drivers use the corresponding `matcont/` directory. Numerical CSVs,
source/Manifest hashes, runtime controls, native conventions and raw logs stay
in these unreviewed candidates. Promotion is a separate reviewed operation.

Run real exports with `python3 -m pytest -m julia -q < /dev/null`; the default
pytest selection does not invoke Julia. A selected missing/broken runtime is a
failure. Native Floquet values are log multipliers and are explicitly converted
to multipliers; radial flow-mode identification uses FloquetGEV eigenvectors aligned with
the physical vector field at phase zero. The asymmetric Hopf coefficient is native
BK complex cubic `b`, without an implicit convention conversion.

The reviewed Lorenz-84 contract currently blocks standalone BT selection:
independent runs locate positive-T BT near F=1.4467167, T=0.02094017, outside
the frozen F window [1.5,2]. That route writes failure evidence and exits
nonzero. The frozen 30-step codim2 controls also need not cover T=-0.01;
metadata records observed/expected domains and `coverage_status: PARTIAL`.
A Hopf chart boundary is never relabeled as a refined BT detection.
