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
`adaptx.cycle-pd-switch` (`adaptx_pd.jl`), `torbpc.cycle-lpc-pd-ns`
(`torbpc_ns.jl`), `hopf-normal-form.l1`, `lorenz84.fold-ft`,
`lorenz84.hopf-ft`, and `lorenz84.bt-ft`. Scripts also accept `--case ID --output-dir EXACT_DIRECTORY`
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

Lorenz-84 codimension two: the BT window is F in [1.40, 1.50], T in
[0.01, 0.03]; `lorenz84_codim2.jl` refines the positive-T BT (near
F=1.4467167, T=0.02094017) inside it and exports `lorenz84.bt-ft`, which is
promoted and agrees with JaxCont and MatCont. The codim-2 controls take 400
steps so the Hopf chart covers T up to 0.04. BifurcationKit's BT labels on the
fold and Hopf curves have native status `guess`; they are exported as such and
never compared as refined points. A Hopf chart boundary is never relabeled as
a refined BT detection.
