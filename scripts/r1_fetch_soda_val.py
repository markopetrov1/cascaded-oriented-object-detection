#!/usr/bin/env python3
"""Fetch only the SODA-A validation images from the public HuggingFace mirror.

SODA-A (Cheng et al., CC BY-NC 4.0) ships all splits in one 11 GB Images.zip at
huggingface.co/datasets/torchgeo/soda-a. The zip is read over HTTP range
requests, so only its central directory and the 576 validation members are
transferred and stored (about 2.5 GB). No account is needed. Annotations.zip
must already be extracted to <out>/Annotations/val.

    python scripts/r1_fetch_soda_val.py --out data/raw/SODA-A
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/torchgeo/soda-a/resolve/main/Images.zip"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()                     # follow the CDN redirect once
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception as e:                       # transient network errors
                if attempt == 4:
                    raise
                print(f"retry after {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/SODA-A")
    args = p.parse_args()
    out = Path(args.out)
    want = sorted(f.stem for f in (out / "Annotations/val").glob("*.json"))
    dest = out / "Images/val"
    dest.mkdir(parents=True, exist_ok=True)
    f = HTTPRangeFile(URL)
    print(f"Images.zip {f.size / 1e9:.2f} GB at {f.url[:60]}...", flush=True)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=16 << 20))
    members = {Path(n).stem: n for n in z.namelist() if n.lower().endswith((".jpg", ".png"))}
    missing = [w for w in want if w not in members]
    if missing:
        raise SystemExit(f"{len(missing)} val images not in the zip, e.g. {missing[:3]}")
    done = 0
    for w in want:
        target = dest / Path(members[w]).name
        if target.exists() and target.stat().st_size == z.getinfo(members[w]).file_size:
            done += 1
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(z.read(members[w]))
        tmp.rename(target)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(want)}", flush=True)
    print(f"fetched {done} validation images into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
