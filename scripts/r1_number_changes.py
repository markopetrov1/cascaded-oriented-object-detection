#!/usr/bin/env python3
"""Every generated number of the submission against the revision (WP13).

Reads the macro files of the submitted manuscript (paper_r0_submitted/generated_numbers.tex)
and the revision (paper/generated_numbers.tex) and writes reports/r1/number_changes.md:
macros present in both with their old and new values, then the macros the revision
removed and added. A macro that kept its name can still have changed its definition
(for example a saving now chosen on held-out scenes), so the table records values,
and the changelog records why each one moved.

    python scripts/r1_number_changes.py
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "paper_r0_submitted/generated_numbers.tex"
NEW = REPO / "paper/generated_numbers.tex"
OUT = REPO / "reports/r1/number_changes.md"
MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*(?:%.*)?$")


def macros(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = MACRO.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def cell(v: str) -> str:
    return v.replace("{,}", ",").replace("|", "\\|") or " "


def main() -> int:
    old, new = macros(OLD), macros(NEW)
    both = [k for k in old if k in new]
    changed = [k for k in both if old[k] != new[k]]
    removed = [k for k in old if k not in new]
    added = [k for k in new if k not in old]
    lines = ["# Generated numbers, submission against revision", "",
             f"Source: `{OLD.relative_to(REPO)}` ({len(old)} macros) and `{NEW.relative_to(REPO)}` "
             f"({len(new)} macros), written by `scripts/r1_number_changes.py`.", "",
             "The revision changed the protocol behind most numbers: predictions at confidence 0.001 instead of "
             "0.25, image-level AP after merging tiles, a 101-point AP budget applied on held-out scenes "
             "(two-fold cross-fitting), the shared-backbone cost for fused gates, and one timing protocol. "
             "A macro that kept its name may therefore measure something slightly different; the changelog "
             "gives the reason for each change.", "",
             f"## Kept, value changed ({len(changed)} of {len(both)} kept)", "",
             "| Macro | Submission | Revision |", "|---|---|---|"]
    lines += [f"| `{k}` | {cell(old[k])} | {cell(new[k])} |" for k in changed]
    lines += ["", f"## Kept, value unchanged ({len(both) - len(changed)})", "",
              ", ".join(f"`{k}`" for k in both if k not in changed) or "none", "",
              f"## Removed ({len(removed)})", "", "| Macro | Submission |", "|---|---|"]
    lines += [f"| `{k}` | {cell(old[k])} |" for k in removed]
    lines += ["", f"## Added ({len(added)})", "", "| Macro | Revision |", "|---|---|"]
    lines += [f"| `{k}` | {cell(new[k])} |" for k in added]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(both)} kept ({len(changed)} changed), {len(removed)} removed, {len(added)} added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
