#!/usr/bin/env python3
"""Download HRSC2016-MS for cross-dataset evaluation.

Pulls the multi-scale variant published by Chen et al. (2022) — the dataset
attached to the MSSDet paper — from the official Google Drive share linked in
https://github.com/wmchen/HRSC2016-MS:

    https://drive.google.com/file/d/1UslulCCx8GoTflm1gpfIGZeXIsCAdMG5/view

Default flow::

    pip install gdown
    python scripts/download_hrsc2016.py     # downloads + extracts to data/raw/HRSC2016/

Override flags:
  --gdrive-id <ID>        use a different Google Drive file id
  --url <https://...>     use a direct HTTP(S) zip/tar URL
  --out-dir <DIR>         destination (default: data/raw/HRSC2016)
  --keep-archive          don't delete the downloaded archive after extraction
  --skip-validate         bypass the layout sanity check

The post-extraction layout is whatever the upstream archive ships; the
validator just spot-counts images and annotation files. Convert to the
cascade pipeline's tile + YOLO-OBB layout afterwards via
``scripts/prepare_hrsc.py`` (TODO).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

DEFAULT_GDRIVE_ID = "1UslulCCx8GoTflm1gpfIGZeXIsCAdMG5"  # wmchen/HRSC2016-MS official share


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--gdrive-id", default=DEFAULT_GDRIVE_ID,
                     help=f"Google Drive file id (default: {DEFAULT_GDRIVE_ID}, the wmchen/HRSC2016-MS share).")
    src.add_argument("--url", help="Direct HTTP(S) URL to a zip/tar archive (overrides --gdrive-id).")
    p.add_argument("--out-dir", default="data/raw/HRSC2016",
                   help="Where to extract (default: data/raw/HRSC2016)")
    p.add_argument("--keep-archive", action="store_true",
                   help="Keep the downloaded archive after extraction (default: delete it).")
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip the post-extraction layout check.")
    return p.parse_args()


def _gdrive_download(file_id: str, dest: Path) -> Path:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required. Install with:  pip install gdown")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] gdown id={file_id} -> {dest}")
    out = gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
    if out is None:
        sys.exit(
            "gdown returned None — the file may be quota-blocked or require manual confirmation.\n"
            f"Open https://drive.google.com/file/d/{file_id}/view in a browser, click Download,\n"
            f"then re-run with --url <direct-url> or save the archive at {dest} and re-run."
        )
    return Path(out)


def _http_download(url: str, dest: Path) -> Path:
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        sys.exit("requests and tqdm are required for --url. Install with:  pip install requests tqdm")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] GET {url} -> {dest}")
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest


def _extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} -> {out_dir}")
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    elif any(name.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
    else:
        sys.exit(f"Unrecognized archive format: {archive.name}")


def _validate(out_dir: Path) -> bool:
    """Spot-count images and annotations; report whatever folder structure shipped."""
    images = (
        list(out_dir.rglob("*.bmp"))
        + list(out_dir.rglob("*.jpg"))
        + list(out_dir.rglob("*.jpeg"))
        + list(out_dir.rglob("*.png"))
    )
    xmls = list(out_dir.rglob("*.xml"))
    txts = list(out_dir.rglob("*.txt"))
    print(f"[validate] images: {len(images)}")
    print(f"[validate] xml annotations: {len(xmls)}")
    print(f"[validate] txt files (DOTA-style annotations or splits): {len(txts)}")
    # Show a couple of top-level entries so caller knows what landed
    top = sorted({p.relative_to(out_dir).parts[0] for p in out_dir.rglob("*") if p.is_file()})[:10]
    print(f"[validate] top-level entries: {top}")
    if len(images) < 100:
        print("[validate] WARNING: <100 images — download may be incomplete.")
        return False
    if not xmls and not txts:
        print("[validate] WARNING: no .xml or .txt annotation files found.")
        return False
    return True


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        suffix = ".zip"
        for ext in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar", ".zip"):
            if args.url.lower().endswith(ext):
                suffix = ext
                break
        archive = _http_download(args.url, out_dir / f"hrsc2016_download{suffix}")
    else:
        archive = _gdrive_download(args.gdrive_id, out_dir / "hrsc2016_download.zip")

    _extract(archive, out_dir)
    if not args.keep_archive and archive.exists():
        archive.unlink()
        print(f"[cleanup] removed archive {archive.name}")

    if not args.skip_validate:
        ok = _validate(out_dir)
        if not ok:
            print("[validate] proceed with caution; re-run with --skip-validate to bypass.")
            return 1

    print(f"[done] HRSC2016-MS ready at {out_dir}")
    print("[next] convert to cascade pipeline format with scripts/prepare_hrsc.py (TODO)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
