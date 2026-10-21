#!/usr/bin/env python3
"""Add yellow highlight annotations over every glyph drawn in the diff colour.

scripts/r1_make_diff.sh has latexdiff draw added text in RGB(0,0,3), which is
visually black and used nowhere else. This finds those glyph runs and places a
yellow highlight annotation over each line's run, so additions read as yellow
highlighting in any PDF viewer.

    python scripts/r1_highlight_pdf.py in.pdf out.pdf
"""
from __future__ import annotations

import sys

import pymupdf

MARK = 0x000003


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    doc = pymupdf.open(src)
    n_spans = 0
    for page in doc:
        rects = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                run = None
                for span in line["spans"]:
                    if span["color"] == MARK and span["text"].strip():
                        r = pymupdf.Rect(span["bbox"])
                        run = r if run is None else run | r
                        n_spans += 1
                    elif run is not None:
                        rects.append(run)
                        run = None
                if run is not None:
                    rects.append(run)
        for r in rects:
            a = page.add_highlight_annot(r)
            a.set_colors(stroke=(1, 0.93, 0.2))
            a.update()
    doc.save(dst, garbage=3, deflate=True)
    print(f"highlighted {n_spans} spans on {len(doc)} pages -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
