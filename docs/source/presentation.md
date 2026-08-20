# Technical presentation

A beginner-friendly, 16:9 Beamer deck for **JaxCont v0.4.0** walks
through continuation fundamentals, the public API, periodic orbits and
Floquet stability, Hopf refinement, direct codimension-two point solvers,
seeded two-parameter fold/Hopf curves, visualization, phase-response curves,
release migration contracts, and validation evidence. The validation gallery
shows all five MatCont visual comparisons while preserving the systematic
`MC-LC-002` failure and its experimental capability boundary.

The deck distinguishes released **v0.4.0 capabilities** from **future work —
not implemented**. PRC/dPRC and fold/Hopf two-parameter continuation are
included in the v0.4.0 capability surface. LPC/PD/NS detection is explicitly
marked experimental while the strict `MC-LC-002` comparison remains failing.

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
Building the deck locally requires `latexmk`, `pdflatex`, and ImageMagick's
`convert`. If those tools are unavailable, the Sphinx build still succeeds;
only this page's PDF is skipped.
```
