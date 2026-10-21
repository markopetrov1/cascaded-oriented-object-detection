#!/bin/bash
# Highlighted-changes PDF for the resubmission (WP12).
# latexdiff marks every addition in a unique near-black colour (RGB 0,0,3), which
# survives math, tables and captions where soul's \hl breaks; deletions are not
# printed. scripts/r1_highlight_pdf.py then puts a yellow highlight annotation
# over exactly the glyphs drawn in that colour. Output: paper/r1_highlighted.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
W=paper/build_diff
mkdir -p $W
cat > $W/preamble.tex <<'TEX'
%DIF PREAMBLE: additions in colour RGB(0,0,3), highlighted in yellow afterwards; deletions omitted
\RequirePackage{xcolor}
\definecolor{difadd}{RGB}{0,0,3}
\providecommand{\DIFadd}[1]{{\protect\color{difadd}#1}}
\providecommand{\DIFdel}[1]{}
\providecommand{\DIFaddbegin}{}
\providecommand{\DIFaddend}{}
\providecommand{\DIFdelbegin}{}
\providecommand{\DIFdelend}{}
\providecommand{\DIFmodbegin}{}
\providecommand{\DIFmodend}{}
\providecommand{\DIFaddFL}[1]{\DIFadd{#1}}
\providecommand{\DIFdelFL}[1]{\DIFdel{#1}}
\providecommand{\DIFaddbeginFL}{}
\providecommand{\DIFaddendFL}{}
\providecommand{\DIFdelbeginFL}{}
\providecommand{\DIFdelendFL}{}
%DIF END PREAMBLE
TEX
latexdiff --flatten --math-markup=whole --graphics-markup=none --preamble=$W/preamble.tex \
    paper_r0_submitted/main.tex paper/main.tex > $W/r1_highlighted.tex 2> $W/latexdiff.log
# latexdiff wraps \cmidrule's arguments in markup, which booktabs cannot parse; unwrap them.
python3 - "$W/r1_highlighted.tex" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(r"\\cmidrule\\DIFadd(?:FL)?\{(\([a-z]+\))\}\{\\DIFadd(?:FL)?\{([0-9-]+)\}\}", r"\\cmidrule\1{\2}", t)
t = re.sub(r"\\cmidrule\\DIFadd(?:FL)?\{\{([0-9-]+)\}\}", r"\\cmidrule{\1}", t)
open(p, "w").write(t)
PY
cp $W/r1_highlighted.tex paper/r1_highlighted.tex
cd paper
latexmk -pdf -interaction=nonstopmode -outdir=build_diff r1_highlighted.tex > build_diff/latexmk.log 2>&1 || true
rm -f r1_highlighted.tex
cd ..
.venv/bin/python scripts/r1_highlight_pdf.py paper/build_diff/r1_highlighted.pdf paper/r1_highlighted.pdf
