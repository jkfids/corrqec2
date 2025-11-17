# Makefile using latexmk for efficient LaTeX builds with biblatex+biber

PDFS = main.pdf notes.pdf

all: $(PDFS)

main.pdf: paper/main.tex paper/references.bib
	cd paper && latexmk -pdf -bibtex -interaction=nonstopmode main.tex

notes.pdf: paper/notes.tex paper/references.bib
	cd paper && latexmk -pdf -bibtex -interaction=nonstopmode notes.tex


# Clean only auxiliary files, keep PDFs and .bbl files
clean:
	cd paper && rm -f *.aux *.bcf *.blg *.log *.out *.run.xml *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.gz *SAVE-ERROR

.PHONY: all clean
