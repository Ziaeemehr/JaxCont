LATEXMK ?= latexmk
JOB := engine_consolidation
TEX := $(JOB).tex
PDF := $(JOB).pdf

.PHONY: all clean distclean rebuild help

all: $(PDF)

$(PDF): $(TEX)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX)

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
