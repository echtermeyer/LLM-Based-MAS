#!/bin/bash
cd "$(dirname "$0")"
rm -f thesis.aux thesis.log thesis.out thesis.toc thesis.lof thesis.lot thesis.fls thesis.fdb_latexmk thesis.bbl thesis.blg
pdflatex -interaction=nonstopmode thesis.tex
bibtex thesis
pdflatex -interaction=nonstopmode thesis.tex
pdflatex -interaction=nonstopmode thesis.tex
pdflatex -interaction=nonstopmode thesis.tex
echo "Done."
