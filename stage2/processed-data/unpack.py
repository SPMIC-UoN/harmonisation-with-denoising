#!/usr/bin/env python3
"""
unpack.py - Unpack the processed data into the folder layout the pipeline and the notebooks expect.

  python unpack.py --out /path/to/derivatives

Extracts every zip in this folder, joins the files that were split to stay below GitHub's file-size limit
(*.part1, *.part2, ...) and checks every file against checksums.sha256. Standard library only.
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="Unpack the processed data.")
    p.add_argument("--out", type=Path, required=True, help="folder to create or add to (use it as RR_DERIV)")
    out = p.parse_args().out
    zips = sorted(HERE.glob("*.zip"))
    want = dict(reversed(x.split("  ", 1)) for x in (HERE / "checksums.sha256").read_text().splitlines())
    if not zips:
        sys.exit("no zip files found next to unpack.py")
    parts = {}                                     # joined name -> number of parts
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                base, sep, k = n.rpartition(".part")
                parts[base if sep and k.isdigit() else n] = max(parts.get(base, 0), int(k)) if sep and k.isdigit() else 0
    clash = [n for n in want if (out / n).exists()]
    if clash:
        sys.exit(f"{len(clash)} files already exist in {out}, e.g. {clash[0]}; choose an empty folder")
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    for n, k in parts.items():
        if k:
            with open(out / n, "wb") as fh:
                for i in range(1, k + 1):
                    q = out / f"{n}.part{i}"
                    fh.write(q.read_bytes())
                    q.unlink()
    bad = [n for n, h in want.items()
           if not (out / n).is_file() or hashlib.sha256((out / n).read_bytes()).hexdigest() != h]
    extra = set(parts) - set(want)
    if bad or extra:
        sys.exit(f"{len(bad)} files missing or different from checksums.sha256, {len(extra)} not listed: {(bad + sorted(extra))[:3]}")
    print(f"{len(want)} files unpacked to {out}; all checksums match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
