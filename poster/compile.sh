#!/bin/bash
# Compile the poster

# Run pdflatex twice for references
pdflatex -interaction=nonstopmode poster.tex
pdflatex -interaction=nonstopmode poster.tex

echo "Poster compiled! Output: poster.pdf"
