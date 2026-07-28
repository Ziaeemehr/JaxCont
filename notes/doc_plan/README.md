# Practical bifurcation guide

This directory contains the plan and the first LaTeX edition of the JaxCont
practical bifurcation handbook.

- `plan.md`: long-term content plan.
- `jaxcont_practical_bifurcation_guide.tex`: editable LaTeX source.
- `jaxcont_practical_bifurcation_guide.pdf`: compiled handbook.

Build with:

```bash
make
```

The v0.2 tutorial edition covers the supported equilibrium workflow and the
fixed-mesh periodic-orbit workflow, including Floquet stability,
period-doubling, and Neimark--Sacker screening. Branch switching, automatic
Hopf-to-cycle initialization, adaptive periodic meshes, two-parameter
continuation, and global-orbit analysis remain clearly labelled future-version
placeholders. Existing repository-generated figures are reused through
relative paths, so build the document from this directory.
