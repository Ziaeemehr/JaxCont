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

The first edition concentrates on the equilibrium-continuation workflow that
JaxCont currently supports. Later workflow stages are included as conceptual
orientation and are explicitly marked as not yet implemented in the public
JaxCont API. Existing repository-generated figures are reused through relative
paths, so build the document from this directory.
