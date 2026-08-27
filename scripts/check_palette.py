#!/usr/bin/env python3
"""Check a figure palette for colour-vision and greyscale separation.

The paper is printed, reproduced in greyscale, and read by people with colour
vision deficiency, so palette choice is a constraint to be computed rather than
eyeballed. Three checks, all in perceptually uniform OKLab:

  1. normal vision   -- every pair separated by dE >= 15
  2. CVD             -- every pair separated by dE >= 8 under simulated
                        protanopia and deuteranopia (Machado et al. 2009,
                        severity 1.0)
  3. greyscale       -- every pair separated by >= 10 in OKLab L, which is what
                        survives a black-and-white print

A pair failing check 3 is legal only when the marks also carry a non-colour
encoding (hatching, marker shape, direct labels). Figures in this repo hatch
their stacked segments, so greyscale failures are reported as warnings; colour
failures are hard.

Usage:
    python3 scripts/check_palette.py "#0072B2,#E69F00,#009E73,#D55E00"
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

# Machado, Oliveira & Fernandes (2009), severity 1.0, applied in linear RGB.
CVD = {
    "protanopia": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deuteranopia": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
}

NORMAL_MIN = 15.0
CVD_MIN = 8.0
GREY_MIN = 10.0


def hex_to_linear(hex_colour: str) -> np.ndarray:
    h = hex_colour.strip().lstrip("#")
    srgb = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    m1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ])
    m2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ])
    return m2 @ np.cbrt(m1 @ rgb)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return 100.0 * float(np.linalg.norm(a - b))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    colours = [c for c in argv[1].split(",") if c.strip()]
    linear = {c: hex_to_linear(c) for c in colours}
    lab = {c: linear_to_oklab(linear[c]) for c in colours}
    lab_cvd = {
        name: {c: linear_to_oklab(np.clip(mat @ linear[c], 0, 1)) for c in colours}
        for name, mat in CVD.items()
    }

    failures = warnings = 0
    print(f"{'pair':<20}{'normal':>9}{'protan':>9}{'deutan':>9}{'greyL':>9}  verdict")
    for a, b in combinations(colours, 2):
        d_norm = delta_e(lab[a], lab[b])
        d_pro = delta_e(lab_cvd["protanopia"][a], lab_cvd["protanopia"][b])
        d_deu = delta_e(lab_cvd["deuteranopia"][a], lab_cvd["deuteranopia"][b])
        d_grey = 100.0 * abs(lab[a][0] - lab[b][0])

        hard = []
        if d_norm < NORMAL_MIN:
            hard.append("normal")
        if d_pro < CVD_MIN:
            hard.append("protan")
        if d_deu < CVD_MIN:
            hard.append("deutan")
        soft = d_grey < GREY_MIN

        if hard:
            verdict = "FAIL " + ",".join(hard)
            failures += 1
        elif soft:
            verdict = "WARN greyscale - needs hatching/shape"
            warnings += 1
        else:
            verdict = "pass"
        print(f"{a + '/' + b:<20}{d_norm:>9.1f}{d_pro:>9.1f}{d_deu:>9.1f}{d_grey:>9.1f}  {verdict}")

    print(f"\n  {len(colours)} colours, {failures} hard failures, {warnings} greyscale warnings")
    print(f"  thresholds: normal >= {NORMAL_MIN}, CVD >= {CVD_MIN}, greyscale L >= {GREY_MIN}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
