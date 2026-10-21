#!/usr/bin/env python3
"""Build paper/r1_overleaf.zip: everything the revised manuscript needs, nothing else.

Collects main.tex, the files it \\input s, the figures and photos it includes,
the bibliography and style, the IEEE Access class with its spot-colour package,
Type 1 fonts, font maps and logo artwork. Then unpacks the zip into an empty
directory and compiles it there (pdflatex, bibtex, pdflatex x2), so a missing
file fails here rather than on Overleaf.

    python scripts/r1_make_overleaf_zip.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
OUT = PAPER / "r1_overleaf.zip"
CLASS_FILES = ["ieeeaccess.cls", "spotcolor.sty", "IEEEtran.bst", "logo.png", "notaglinelogo.png", "bullet.png"]


def needed(main: Path) -> list[str]:
    tex = main.read_text()
    tex = re.sub(r"(?<!\\)%.*", "", tex)                    # drop comments
    files = {"main.tex", "refs.bib"}
    for name in re.findall(r"\\input\{([^}]+)\}", tex):
        files.add(name if name.endswith(".tex") else name + ".tex")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        files.add(name)
    files.update(CLASS_FILES)
    files.update(p.name for p in PAPER.iterdir() if p.suffix in {".pfb", ".tfm", ".fd", ".map"})
    return sorted(files)


def main() -> int:
    files = needed(PAPER / "main.tex")
    missing = [f for f in files if not (PAPER / f).exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    if OUT.exists():
        OUT.rename(OUT.with_suffix(".previous.zip"))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(PAPER / f, f)
    print(f"wrote {OUT} ({len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB)")
    env = dict(os.environ, PATH=f"{Path.home()}/.TinyTeX/bin/x86_64-linux:" + os.environ["PATH"])
    with tempfile.TemporaryDirectory() as d:
        zipfile.ZipFile(OUT).extractall(d)
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
            subprocess.run(cmd, cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log = Path(d, "main.log").read_text(errors="replace")
        errors = [l for l in log.splitlines() if l.startswith("!")]
        undefined = [l for l in log.splitlines() if "undefined" in l and "Font" not in l]
        pages = re.findall(r"Output written on main.pdf \((\d+) pages", log)
        print(f"clean build from the zip: pages={pages[-1] if pages else '?'} errors={len(errors)} "
              f"undefined={len(undefined)}")
        if errors or undefined or not pages:
            print("\n".join(errors[:5] + undefined[:5]))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
