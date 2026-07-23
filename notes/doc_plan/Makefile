LATEXMK ?= latexmk
JOB := jaxcont_practical_bifurcation_guide

.PHONY: all clean distclean rebuild

all: $(JOB).pdf

$(JOB).pdf: $(JOB).tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(JOB).tex

clean:
	$(LATEXMK) -c $(JOB).tex
	$(RM) $(JOB).synctex.gz

distclean: clean
	$(RM) $(JOB).pdf

rebuild: distclean all
