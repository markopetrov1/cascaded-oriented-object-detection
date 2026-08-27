#!/usr/bin/env python3
"""Wall-clock latency table: cascade vs single-stage YOLO11n at matched-compute imgsz.

Combines the measured per-model ms/img from reports/speed/*.json with the
filter rate at the cascade's best operating point per class. Produces:
  cascade_total_ms = gate_ms × n_tiles + detector_ms × n_tiles_passed

vs YOLO11n single-stage at imgsz {768, 896, 1024}, reading detector ms scaled
by (imgsz/1024)^2 from the YOLO11m baseline (proxy — true value depends on
hardware utilization but the scaling is approximately right).
"""
from __future__ import annotations
import json, glob
from collections import defaultdict


def load_speed() -> dict:
    out = {}
    for f in glob.glob("reports/speed/*.json"):
        d = json.load(open(f))
        out[d["label"]] = {"gflops": d["gflops_per_image"], "ms": d["ms_per_image_gpu"]}
    return out


def cascade_best(glob_pat: str):
    rows = []
    seen = set()
    for f in glob.glob(glob_pat):
        if "stratif" in f or "codesign" in f or "distill" in f or "smoke" in f:
            continue
        for r in json.load(open(f)):
            lbl = r.get('label', '')
            calib = r.get('calibration', '')
            if calib and lbl.endswith('_' + calib):
                continue
            key = (lbl, calib, round(r['threshold'], 3))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    real = [r for r in rows if r.get('gate') != 'oracle']
    fp = next((r for r in real if r['threshold'] == 0.0), None)
    threshold = fp['mAP@0.50'] * 0.97
    cand = [r for r in real if r['mAP@0.50'] >= threshold]
    # Best = max compute saved (min total_gflops), not max filter_rate, since
    # gates have very different compute costs.
    return fp, min(cand, key=lambda r: r['total_gflops'])


def main() -> int:
    speed = load_speed()
    yolo11m = speed["yolo11m_obb"]
    print("=" * 100)
    print("WALL-CLOCK LATENCY (median GPU ms/img × n_tiles) — cascade vs matched-compute YOLO11n")
    print("=" * 100)

    rows = []
    for cls, glob_pat, n_tiles in [
        ("ships",         "reports/cascade/*.json",              5297),
        ("planes",        "reports/planes/cascade/*.json",       5174),
        ("small_vehicle", "reports/small_vehicle/cascade/*.json", 5297),
    ]:
        fp, best = cascade_best(glob_pat)
        gate_label = best.get('label', '')
        backbone = next((b for b in ['resnet50','resnet18','mbv3large','mbv3small','effb0','tiny'] if b in gate_label), '?')
        gate_ms = speed.get(f"gate_{backbone}", {"ms": 0})["ms"]
        det_ms_full = yolo11m["ms"]
        n_pass = best['n_accepted']
        cascade_total_ms = gate_ms * n_tiles + det_ms_full * n_pass
        print(f"\n[{cls.upper()}]")
        print(f"  {'Approach':<48} {'GPU ms total':>13} {'GPU s total':>12}")
        print(f"  {'YOLO11m full-pass':<48} {det_ms_full * n_tiles:>13,.0f} {det_ms_full * n_tiles / 1000:>12.1f}")
        print(f"  {'Cascade ('+gate_label+')':<48} {cascade_total_ms:>13,.0f} {cascade_total_ms/1000:>12.1f}")
        rows.append({"class": cls, "approach": "cascade", "ms_total": cascade_total_ms, "n_tiles": n_tiles, "n_passed": n_pass})

        # YOLO11n at each imgsz — scale ms by FLOPs-ratio relative to YOLO11m@1024
        for imgsz in [768, 896, 1024]:
            n_gflops = 6.6 * (imgsz / 1024) ** 2
            # Approximate ms by linearly scaling from YOLO11m's measured ms via FLOPs ratio
            # (real hardware utilization may differ; flag in paper)
            n_ms_proxy = det_ms_full * n_gflops / yolo11m["gflops"]
            total_ms = n_ms_proxy * n_tiles
            ratio = total_ms / cascade_total_ms
            print(f"  {'YOLO11n single-stage @ imgsz='+str(imgsz):<48} {total_ms:>13,.0f} {total_ms/1000:>12.1f}  ({ratio:.2f}x cascade)")
            rows.append({"class": cls, "approach": f"yolo11n_imgsz{imgsz}", "ms_total": total_ms, "n_tiles": n_tiles})

    import json as _json
    from pathlib import Path
    out = Path("reports/wallclock_table.json")
    out.write_text(_json.dumps(rows, indent=2))
    print(f"\n  Wrote {out}")
    print("\n  CAVEAT: YOLO11n ms estimates are proxies (FLOPs-scaled from YOLO11m measurement);")
    print("  actual hardware utilization (memory-bandwidth bound vs compute bound) may shift these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
