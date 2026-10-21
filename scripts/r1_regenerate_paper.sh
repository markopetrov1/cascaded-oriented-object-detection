#!/bin/bash
# Regenerate everything the revised manuscript reads from reports/r1, then build it.
# Run twice and compare paper/generated_*.tex and reports/r1/figures/*.pdf to check that the
# chain is reproducible byte for byte (WP13). The evaluations that produce reports/r1 are run
# by the scripts listed in scripts/r1_regenerate_results.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
python3 scripts/prior_work_prediction.py > /dev/null
$PY scripts/r1_misc_facts.py > /dev/null
$PY scripts/paper_figure_teaser.py > /dev/null
$PY scripts/paper_figure_architecture.py > /dev/null
$PY scripts/paper_figure_taxonomy.py > /dev/null
$PY scripts/r1_paper_figures.py > /dev/null
$PY scripts/r1_paper_numbers.py
python3 scripts/r1_number_changes.py > /dev/null
$PY scripts/r1_paper_tables.py > /dev/null
paper/r1_parts/assemble.sh
(cd paper && ./compile_paper.sh)
echo "pages: $(pdfinfo paper/main.pdf | awk '/Pages/ {print $2}')"
