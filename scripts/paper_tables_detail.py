#!/usr/bin/env python3
"""Emit the detail tables the manuscript needs, all from reports/ and runs/.

Kept separate from paper_tables.py so the sweep table and these can be
regenerated independently. Writes paper/generated_tables_detail.tex containing:

  tab:speed     measured cost of every component on the target GPU
  tab:ece       calibration error per gate backbone and calibration map
  tab:strat     stratified behaviour across four axes at a stated threshold
  tab:codesign  gate quality of the joint variants against independent training

Anything whose source file is missing is skipped rather than faked.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

OUT = Path("paper/generated_tables_detail.tex")
GATES = ["mbv3small", "mbv3large", "resnet18", "resnet50", "effb0", "tiny"]
PRETTY_GATE = {"mbv3small": "MobileNetV3-small", "mbv3large": "MobileNetV3-large",
               "resnet18": "ResNet-18", "resnet50": "ResNet-50",
               "effb0": "EfficientNet-B0", "tiny": "tiny ConvNet"}
CALIBS = ["identity", "temperature", "platt", "isotonic", "context_adaptive"]
PRETTY_CALIB = {"identity": "ident.", "temperature": "temp.", "platt": "Platt",
                "isotonic": "isot.", "context_adaptive": "ctx.-ad."}
# The stratified sweep was run with this configuration; stated so the numbers
# are reproducible rather than attributed to an unnamed "natural" operating point.
STRAT_GLOB = "reports/cascade/gate_resnet18_temperature_strat_{}.stratified_{}.json"
STRAT_TAU = 0.5


def best_pr_auc(run: str) -> float | None:
    f = Path("runs") / run / "history.json"
    if not f.exists():
        return None
    hist = json.load(open(f))
    vals = [e.get("val/pr_auc") or e.get("val_pr_auc") or 0 for e in hist]
    return max(vals) if vals else None


def table_speed() -> list[str]:
    rows = []
    for f in sorted(glob.glob("reports/speed/*.json")):
        d = json.load(open(f))
        lab = d.get("label", os.path.basename(f))
        if "_b" in lab:            # batch-size sweep lives in its own figure
            continue
        rows.append((lab, d.get("gflops_per_image"), d.get("ms_per_image_gpu")))
    if not rows:
        return []
    rows.sort(key=lambda r: r[1] or 0)
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Measured per-image cost of every component on the target GPU, "
         r"an NVIDIA Quadro RTX 8000. Gates run at input size $256$ and detectors "
         r"at $1024$. Latency is the median over warmup plus fifty "
         r"CUDA-event-synchronised iterations at batch one.}",
         r"\label{tab:speed}", r"\small",
         r"\begin{tabular}{lrr}", r"\toprule",
         r"Component & GFLOPs/img & ms/img \\", r"\midrule"]
    for lab, g, ms in rows:
        name = lab.replace("gate_", "gate / ").replace("_", " ")
        L.append(f"{name} & ${g:.2f}$ & ${ms:.2f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def table_ece() -> list[str]:
    data = {}
    for g in GATES:
        f = Path(f"reports/calibration/ships/gate_{g}/reliability_summary.json")
        if f.exists():
            data[g] = json.load(open(f))
    if not data:
        return []
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Expected calibration error of each gate on DOTA-1.5 ships "
         r"under each calibration map. The three classical maps reduce it "
         r"substantially and none of them moves the cascade frontier "
         r"(\S\ref{sec:whatdoesnt}). The context-adaptive variant is a decision "
         r"rule rather than a probability map, and decalibrates by design. "
         r"}",
         r"\label{tab:ece}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l" + "r" * len(CALIBS) + "}", r"\toprule",
         "Gate & " + " & ".join(PRETTY_CALIB[c] for c in CALIBS) + r" \\", r"\midrule"]
    for g in GATES:
        if g not in data:
            continue
        cells = []
        for c in CALIBS:
            v = data[g].get(c, {}).get("ece")
            cells.append(f"${v:.3f}$" if v is not None else "---")
        L.append(f"{PRETTY_GATE[g]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def table_strat() -> list[str]:
    blocks = []
    for stratum, title in (("imagesource", "Sensor source"),
                           ("size_bucket", "Largest object"),
                           ("gsd_bucket", "Ground sampling distance"),
                           ("boundary", "Tile boundary")):
        f = STRAT_GLOB.format(stratum, stratum)
        if not os.path.exists(f):
            continue
        rows = [r for r in json.load(open(f))
                if abs(r["threshold"] - STRAT_TAU) < 1e-9 and r.get("bucket")
                and r.get("mAP@0.50") is not None]
        for r in sorted(rows, key=lambda r: str(r["bucket"])):
            blocks.append((title, str(r["bucket"]), r["n_tiles"], r["mAP@0.50"],
                           r["filter_rate"], r["gate_recall_on_positive_tiles"]))
        blocks.append(None)   # rule between strata
    if not blocks:
        return []
    # Six columns and sixteen rows do not fit one IEEE column, so this one spans
    # both. table* floats to the top of a page; dblfloatfix lets it reach a bottom.
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Stratified behaviour on DOTA-1.5 ships, for the ResNet-18 gate "
         r"under temperature scaling at $\tau = " + f"{STRAT_TAU}" + r"$. Sensor "
         r"source spreads detection accuracy by more than an order of magnitude "
         r"while the gate filters most aggressively exactly where the detector is "
         r"least useful.}",
         r"\label{tab:strat}", r"\footnotesize", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Axis & Bucket & Tiles & mAP$@0.5$ & Filter & Gate rec. \\", r"\midrule"]
    last = None
    for b in blocks:
        if b is None:
            L.append(r"\midrule")
            continue
        title, bucket, n, m, fr, rec = b
        shown = title if title != last else ""
        last = title
        L.append(f"{shown} & {bucket.replace('_',' ')} & ${n}$ & ${m:.3f}$ & "
                 f"${fr:.3f}$ & ${rec:.2f}$ \\\\")
    while L[-1] == r"\midrule":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return L


def table_codesign() -> list[str]:
    indep = {g: best_pr_auc(f"gate_{g}") for g in GATES}
    distil = {g: best_pr_auc(f"gate_ships_{g}_distill") for g in GATES}
    joint = {"Shared backbone": best_pr_auc("codesign_shared_resnet18"),
             "Early exit": best_pr_auc("codesign_early_exit_resnet18"),
             "Gumbel-relaxed": best_pr_auc("codesign_relaxed_resnet18")}
    if not any(distil.values()) and not any(joint.values()):
        return []
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Gate ranking quality on DOTA-1.5 ships. Upper block: "
         r"independent binary training against detector-to-gate distillation, per "
         r"backbone. Lower block: the three joint architectures of our own, all on "
         r"a ResNet-18 trunk, against the independently trained ResNet-18. Both "
         r"failures are architectural; a head placed on the detector's own "
         r"backbone succeeds (\S\ref{sec:fusing}).}",
         r"\label{tab:codesign}", r"\small",
         r"\begin{tabular}{lrr}", r"\toprule",
         r"Gate & Binary PR-AUC & Distilled PR-AUC \\", r"\midrule"]
    for g in GATES:
        if indep[g] is None:
            continue
        d = f"${distil[g]:.3f}$" if distil.get(g) else "---"
        L.append(f"{PRETTY_GATE[g]} & ${indep[g]:.3f}$ & {d} \\\\")
    L += [r"\midrule", r"\multicolumn{3}{l}{\textit{Joint architectures, ResNet-18 trunk}} \\"]
    base = indep.get("resnet18")
    for name, v in joint.items():
        if v is None:
            continue
        L.append(f"{name} & ${v:.3f}$ & \\\\")
    if base:
        L.append(f"Independent (reference) & ${base:.3f}$ & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return L


def main() -> int:
    parts = ["% Generated by scripts/paper_tables_detail.py -- do not edit by hand.", ""]
    made = []
    for fn, name in ((table_speed, "tab:speed"), (table_ece, "tab:ece"),
                     (table_strat, "tab:strat"), (table_codesign, "tab:codesign")):
        block = fn()
        if block:
            parts += block
            made.append(name)
        else:
            print(f"  skipped {name} (source data missing)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print(f"  wrote {OUT} with {len(made)} tables: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
