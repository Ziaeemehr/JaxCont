LATEXMK ?= latexmk
CONVERT ?= convert
JOB := jaxcont_technical_presentation
TEX := $(JOB).tex
PDF := $(JOB).pdf
SETUP := presentation_setup.tex
CHAPTERS := $(wildcard chapters/*.tex)
ASSETS := $(wildcard assets/*)
LOGO_SVG := ../../docs/images/logo/logo.svg
LOGO_PNG := jaxcont_logo.png

.PHONY: all clean distclean rebuild verify help

all: $(PDF)

$(PDF): $(TEX) $(SETUP) $(CHAPTERS) $(ASSETS) $(LOGO_PNG)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX)

$(LOGO_PNG): $(LOGO_SVG)
	$(CONVERT) -background none -density 180 $< -resize 900x $@

# Remove generated LaTeX intermediates while preserving the compiled PDF.
clean:
	$(LATEXMK) -c $(TEX)
	$(RM) $(JOB).nav $(JOB).snm $(JOB).synctex.gz $(JOB).vrb

# Remove intermediates and the compiled PDF.
distclean: clean
	$(RM) $(PDF)

rebuild: distclean all

verify: rebuild
	@test "$$(pdfinfo $(PDF) | awk '/^Pages:/ {print $$2}')" -gt 0
	@! rg -n "LaTeX Error|Undefined control sequence|File .* not found|There were undefined references" $(JOB).log
	@pdftotext $(PDF) - | rg -q "JaxCont v0.3.1"
	@pdftotext $(PDF) - | rg -q "Current main"
	@pdftotext $(PDF) - | rg -q "Phase-response"

help:
	@echo "make          Build $(PDF)"
	@echo "make clean    Remove build intermediates, preserving $(PDF)"
	@echo "make distclean Remove build intermediates and $(PDF)"
	@echo "make rebuild  Clean everything and rebuild $(PDF)"
