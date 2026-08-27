#!/usr/bin/env python3
"""Measure GFLOPs and per-image latency for a detector or gate model.

Two modes:
  - Detector (YOLO-OBB): ``--mode detector --weights <best.pt> --imgsz 1024``
  - Gate (timm/torch):   ``--mode gate --gate-config <yaml> --weights <best.pt> --imgsz 256``

Outputs a single JSON with ``gflops_per_image``, ``ms_per_image_gpu``,
``ms_per_image_cpu``, and the parameters used to obtain them, so downstream
``eval_cascade.py`` can substitute the measured values for the previously
hardcoded constants (``--gate-flops-g``, ``--detector-flops-g``, etc.).

Methodology:
  - GFLOPs: ``thop.profile`` on a single forward (× 1e-9 to convert to G).
  - GPU latency: warmup ``--warmup`` iters, then time ``--iters`` forwards
    with ``torch.cuda.synchronize`` + ``time.perf_counter`` and report the
    median per-image ms across iterations (median is robust to outliers).
  - CPU latency: same loop on ``cpu`` device. Skipped by default for the
    detector (very slow); enable with ``--include-cpu``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", choices=("detector", "gate"), required=True)
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--gate-config", help="Required when --mode gate; provides backbone name")
    p.add_argument("--imgsz", type=int, required=True, help="Square input size (1024 detector, 256 gate)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch for latency measurement. Reported ms is per-image (total / batch).")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--include-cpu", action="store_true",
                   help="Also measure CPU latency. Off by default (detector is very slow on CPU).")
    p.add_argument("--label", required=True, help="Tag stored in the JSON for identification")
    p.add_argument("--out", required=True, help="Output JSON path")
    return p.parse_args()


def _build_gate(gate_config_path: str, weights: str, device: torch.device) -> torch.nn.Module:
    from src.gate import build_gate_model
    cfg = yaml.safe_load(open(gate_config_path))
    backbone = cfg.get("backbone") or cfg.get("model", {}).get("backbone")
    if backbone is None:
        raise SystemExit(f"Could not find 'backbone' in {gate_config_path}")
    model = build_gate_model(backbone, pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict") or state.get("model") or state
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def _build_detector(weights: str, device: torch.device):
    from ultralytics import YOLO
    model = YOLO(weights, task="obb")
    model.model.to(device).eval()
    return model


def _measure_flops(model: torch.nn.Module, imgsz: int, device: torch.device) -> float:
    from thop import profile
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        macs, _params = profile(model, inputs=(x,), verbose=False)
    return float(macs) * 2.0 * 1e-9


def _time_loop(model: torch.nn.Module, imgsz: int, batch: int, device: torch.device, warmup: int, iters: int) -> float:
    x = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize(device)
        per_iter_ms: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize(device)
            per_iter_ms.append((time.perf_counter() - t0) * 1000.0 / batch)
    return float(statistics.median(per_iter_ms))


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if args.mode == "gate":
        if not args.gate_config:
            raise SystemExit("--gate-config required for --mode gate")
        model = _build_gate(args.gate_config, args.weights, device)
    else:
        wrapper = _build_detector(args.weights, device)
        model = wrapper.model

    gflops = _measure_flops(model, args.imgsz, device)
    ms_gpu = _time_loop(model, args.imgsz, args.batch_size, device, args.warmup, args.iters)

    ms_cpu: float | None = None
    if args.include_cpu:
        cpu_dev = torch.device("cpu")
        # Need a CPU copy; some YOLO wrappers keep state on device, so deep-copy via state_dict reload.
        if args.mode == "gate":
            model_cpu = _build_gate(args.gate_config, args.weights, cpu_dev)
        else:
            wrapper_cpu = _build_detector(args.weights, cpu_dev)
            model_cpu = wrapper_cpu.model
        ms_cpu = _time_loop(model_cpu, args.imgsz, args.batch_size, cpu_dev, max(2, args.warmup // 4), max(10, args.iters // 5))

    out = {
        "label": args.label,
        "mode": args.mode,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "device": args.device,
        "gflops_per_image": gflops,
        "ms_per_image_gpu": ms_gpu,
        "ms_per_image_cpu": ms_cpu,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[benchmark] {args.label}: {gflops:.2f} GFLOPs/image, {ms_gpu:.2f} ms/image (GPU)"
          + (f", {ms_cpu:.2f} ms/image (CPU)" if ms_cpu is not None else ""))
    print(f"[benchmark] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
