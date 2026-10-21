#!/usr/bin/env python3
"""One timing protocol for every component (reviewer concerns R1.1 and R2.2).

The first submission mixed measurements: Table 1 reported gates at batch 32 and
detectors at batch 8 under a caption saying batch 1, and the text used batch-1
values from a different file. Here every component is measured the same way:

  * forward pass only, on a device-resident input (no decoding, no host transfer);
  * CUDA events around each call, 50 warm-up and 200 timed iterations, median and
    interquartile range; wall-clock with synchronisation is recorded alongside;
  * FP32 and FP16 (autocast), cudnn.benchmark off (the PyTorch default);
  * gates at 256 px for batch 1, 8, 32, 128; detectors at 1024 px for batch 1, 4, 8, 16;
  * GFLOPs from thop (multiply-accumulates x 2) at batch 1.

Also measures the YOLO11m-OBB backbone prefix (layers 0..k) and the matching
suffix for the taps the fused gate could use, and the GPU resize of a 1024-px
tile to the gate's 256-px input.

Writes reports/r1/speed/components.json. Run on an otherwise idle card.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from thop import profile  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.gate import build_gate_model  # noqa: E402
from src.oan import OANHead  # noqa: E402

GATES = {
    "mbv3small": ("mobilenetv3_small_100", "runs/gate_mbv3small/best.pt"),
    "mbv3large": ("mobilenetv3_large_100", "runs/gate_mbv3large/best.pt"),
    "mbv3large_sweep": ("mobilenetv3_large_100", "runs/gate_sweep_ship/best.pt"),
    "resnet18": ("resnet18", "runs/gate_resnet18/best.pt"),
    "resnet50": ("resnet50", "runs/gate_resnet50/best.pt"),
    "effb0": ("efficientnet_b0", "runs/gate_effb0/best.pt"),
    "tiny": ("tiny", "runs/gate_tiny/best.pt"),
}
DETECTORS = {
    "yolo11m_multiclass": "runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/weights/best.pt",
    "yolo11m_ships_p50": "runs/obb/runs/baseline_yolo11m_dota_ships_obb_p50/weights/best.pt",
    "yolo11m_oan_joint": "runs/obb/runs/oan/oan_ships_lambda3/weights/best.pt",
    "yolo11n_ships": "runs/obb/runs/baseline_yolo11n_dota_ships_obb/weights/best.pt",
}
TAPS = [2, 4, 6, 8, 10]


def timed(fn, warmup: int, iters: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ev, wall = [], []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1000)
            ev.append(s.elapsed_time(e))
    q = statistics.quantiles(ev, n=4)
    return {"median_ms": statistics.median(ev), "q25_ms": q[0], "q75_ms": q[2],
            "wall_median_ms": statistics.median(wall)}


def gflops(model: nn.Module, x: torch.Tensor) -> float:
    macs, _ = profile(copy.deepcopy(model).float().cpu(), inputs=(x.float().cpu(),), verbose=False)
    return 2 * macs / 1e9


class Prefix(nn.Module):
    """Layers 0..k of a YOLO model (the backbone part is a plain sequence)."""

    def __init__(self, model: nn.Module, k: int):
        super().__init__()
        assert all(m.f == -1 for m in model.model[:k + 1])
        self.layers = nn.Sequential(*model.model[:k + 1])

    def forward(self, x):
        return self.layers(x)


def run_prefix_suffix(model: nn.Module, x: torch.Tensor, k: int):
    """Full forward split at layer k: returns (saved outputs after the prefix, suffix callable)."""
    y = []
    out = x
    for m in model.model[:k + 1]:
        out = m(out)
        y.append(out if m.i in model.save or m.i == k else None)

    def suffix(y_saved=y, start=k + 1):
        yy = list(y_saved)
        o = yy[k]
        for m in model.model[start:]:
            if m.f != -1:
                o = yy[m.f] if isinstance(m.f, int) else [o if j == -1 else yy[j] for j in m.f]
            o = m(o)
            yy.append(o if m.i in model.save else None)
        return o
    return y, suffix


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", default="reports/r1/speed/components.json")
    args = p.parse_args()
    dev = torch.device(f"cuda:{args.device}")
    # Make it the current device: torch.cuda.synchronize() and CUDA events otherwise refer to
    # cuda:0 and do not wait for, or time, work running on another card.
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = False
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    out = {"protocol": {"timer": "CUDA events around each call; wall-clock with synchronize also recorded",
                        "warmup": args.warmup, "iters": args.iters, "statistic": "median, IQR",
                        "included": "model forward on a device-resident tensor; no decoding, NMS or transfer",
                        "cudnn_benchmark": False, "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(dev), "nvidia_smi_before": smi},
           "gates": {}, "detectors": {}, "prefix": {}, "resize": {}}

    for name, (bb, w) in GATES.items():
        m = build_gate_model(bb, pretrained=False)
        ck = torch.load(REPO / w, map_location="cpu", weights_only=False)
        m.load_state_dict(ck.get("model", ck))
        m = m.to(dev).eval()
        entry = {"backbone": bb, "weights": w, "imgsz": 256,
                 "gflops": gflops(m, torch.zeros(1, 3, 256, 256)), "fp32": {}, "fp16": {}}
        for b in (1, 8, 32, 128):
            x = torch.rand(b, 3, 256, 256, device=dev)
            entry["fp32"][b] = timed(lambda: m(x), args.warmup, args.iters)
            def f16():
                with torch.autocast("cuda", dtype=torch.float16):
                    m(x)
            entry["fp16"][b] = timed(f16, args.warmup, args.iters)
        out["gates"][name] = entry
        print(name, entry["gflops"], {b: round(v["median_ms"] / b, 3) for b, v in entry["fp32"].items()}, flush=True)
        del m; torch.cuda.empty_cache()

    for name, w in DETECTORS.items():
        m = YOLO(str(REPO / w), task="obb").model.float().to(dev).eval()
        entry = {"weights": w, "imgsz": 1024, "gflops": gflops(m, torch.zeros(1, 3, 1024, 1024)),
                 "fp32": {}, "fp16": {}}
        for b in (1, 4, 8, 16):
            x = torch.rand(b, 3, 1024, 1024, device=dev)
            entry["fp32"][b] = timed(lambda: m(x), args.warmup, args.iters)
            def f16():
                with torch.autocast("cuda", dtype=torch.float16):
                    m(x)
            entry["fp16"][b] = timed(f16, args.warmup, args.iters)
        out["detectors"][name] = entry
        print(name, entry["gflops"], {b: round(v["median_ms"] / b, 3) for b, v in entry["fp32"].items()}, flush=True)

        if name == "yolo11m_ships_p50":
            # backbone prefix and suffix costs for the fused-gate taps
            for k in TAPS:
                pre = Prefix(m, k)
                feat = pre(torch.zeros(1, 3, 1024, 1024, device=dev))
                head = OANHead(int(feat.shape[1])).to(dev).eval()
                pk = {"prefix_gflops": gflops(pre, torch.zeros(1, 3, 1024, 1024)),
                      "head_gflops": gflops(head, feat.detach()), "feature_shape": list(feat.shape[1:]),
                      "fp32": {}}
                for b in (1, 16):
                    x = torch.rand(b, 3, 1024, 1024, device=dev)
                    pk["fp32"][b] = {"prefix": timed(lambda: pre(x), args.warmup, args.iters)}
                    with torch.inference_mode():
                        y, suffix = run_prefix_suffix(m, x, k)
                        ref = m(x)
                        got = suffix()
                        r0 = ref[0] if isinstance(ref, (list, tuple)) else ref
                        g0 = got[0] if isinstance(got, (list, tuple)) else got
                        pk.setdefault("suffix_max_abs_diff", float((r0 - g0).abs().max()))
                    pk["fp32"][b]["suffix"] = timed(suffix, args.warmup, args.iters)
                    fb = pre(x)
                    pk["fp32"][b]["head"] = timed(lambda: head(fb), args.warmup, args.iters)
                out["prefix"][k] = pk
                print("tap", k, round(pk["prefix_gflops"], 2), round(pk["head_gflops"], 3),
                      "suffix diff", pk["suffix_max_abs_diff"], flush=True)
        del m; torch.cuda.empty_cache()

    for b in (1, 32):
        x = torch.rand(b, 3, 1024, 1024, device=dev)
        out["resize"][b] = timed(lambda: F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False),
                                 args.warmup, args.iters)
    out["protocol"]["nvidia_smi_after"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,clocks.sm",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    Path(REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
