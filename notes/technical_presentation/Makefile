LATEXMK ?= latexmk
CONVERT ?= convert
JOB := jaxcont_technical_presentation
TEX := $(JOB).tex
PDF := $(JOB).pdf
LOGO_SVG := ../../docs/images/logo/logo.svg
LOGO_PNG := jaxcont_logo.png

.PHONY: all clean distclean rebuild help

all: $(PDF)

$(PDF): $(TEX) $(LOGO_PNG)
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

help:
	@echo "make          Build $(PDF)"
	@echo "make clean    Remove build intermediates, preserving $(PDF)"
	@echo "make distclean Remove build intermediates and $(PDF)"
	@echo "make rebuild  Clean everything and rebuild $(PDF)"
