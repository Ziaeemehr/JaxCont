# Technical presentation

A beginner-friendly, 16:9 Beamer deck labeled **JaxCont v0.3.1+** walks
through continuation fundamentals, the public API, periodic orbits and
Floquet stability, Hopf refinement, direct codimension-two point solvers,
visualization, phase-response curves, and validation evidence.

The deck distinguishes three source statuses throughout: features
**available in v0.3.1**, features on **current main — planned for v0.4**
(including `prc_curve`, `branch_prc`, and `dprc_curve`), and **future
work — not implemented**.

[**Download the slides (PDF)**](_static/jaxcont_technical_presentation.pdf)

```{raw} html
<embed src="_static/jaxcont_technical_presentation.pdf" type="application/pdf"
       width="100%" height="600px" />
```

The PDF is built from the LaTeX source under
[`notes/technical_presentation/`](https://github.com/Ziaeemehr/JaxCont/tree/main/notes/technical_presentation)
whenever the documentation is built, so it is never committed to the
repository as a binary artifact. If the embedded viewer above is blank, your
browser may be blocking inline PDFs — use the download link instead.

```{note}
Building the deck locally requires `latexmk`, `xelatex`, and ImageMagick's
`convert`. If those tools are unavailable, the Sphinx build still succeeds;
only this page's PDF is skipped.
```
