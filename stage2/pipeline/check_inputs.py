#!/usr/bin/env python3
"""
check_inputs.py - Structural checks on the tables written by build_inputs.py (no values are printed).

  - H1: every (subject, unit) has exactly one row per arm; rows per subject = kept checkerboard voxels / ROIs from
        counts.json (/ drops); expected numbers of subjects; pooled z has mean 0, SD 1 over both arms.
  - H2/H4: one row per unit and per pair; expected number of pairs and repeats.
  - H3: every (subject, unit, arm) has exactly one row per scanner; the same units in both arms and in every
        subject (valid in every session of every subject); cells balanced.
  - counts.json:  NaN patterns identical across arms and equal to the _missing masks (0 differences).
Exit code 1 and a list of failures if anything is off.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read(p: Path):
    with gzip.open(p, "rt") as fh:
        return list(csv.DictReader(fh))


def check(inputs: Path, n_subjects: int, n_scanners: int, n_pairs: int, n_rois: int) -> list[str]:
    C = json.loads((inputs / "counts.json").read_text())
    fail = []
    ok = lambda cond, msg: None if cond else fail.append(msg)

    ok(len(C["repeat_pairs"]) == n_pairs, f"repeat pairs {len(C['repeat_pairs'])} != {n_pairs}")
    for where, d in C.get("d41_checks", {}).items():
        ok(d["nan_arms_differ"] == 0 and d["nan_vs_missing_mask_differ"] == 0, f"NaN pattern mismatch {where}")

    for m in ("FA", "MD", "MK"):
        for lvl in "ab":
            p = inputs / f"h1{lvl}_{m}.csv.gz"
            if not p.is_file():
                fail.append(f"missing {p.name}"); continue
            rows = read(p)
            per = Counter((r["subject"], r["unit"]) for r in rows)
            ok(set(per.values()) == {2}, f"{p.name}: (subject, unit) without exactly 2 arms")
            ok(Counter(r["den"] for r in rows) == Counter({"0": len(rows) // 2, "1": len(rows) // 2}),
               f"{p.name}: arms unbalanced")
            subs = Counter(r["subject"] for r in rows)
            ok(len(subs) == n_subjects, f"{p.name}: {len(subs)} subjects")
            for s, n in subs.items():
                u = C["units"][f"H1/{s}/all/{m}"]
                exp = (u["checkerboard_kept"] - u["cov_nonpos_mean_voxels"] if lvl == "a"
                       else u["rois"] - u["rois_empty_or_dropped"] - u["rois_cov_undefined"])
                ok(n == 2 * exp, f"{p.name} {s}: {n} rows, expected {2 * exp}")
                if lvl == "b":
                    ok(u["rois"] == n_rois, f"{s}: {u['rois']} ROIs != {n_rois}")
            y = np.array([float(r["y"]) for r in rows])
            ok(abs(y.mean()) < 1e-9 and abs(y.std() - 1) < 1e-9, f"{p.name}: pooled z not (0, 1)")

    for h in ("h2", "h4"):
        for lvl in "ab":
            for m in ("FA", "MD", "MK"):
                ps = sorted(inputs.glob(f"{h}{lvl}_sub-*_{m}.csv.gz"))
                ok(len(ps) == n_pairs, f"{h}{lvl} {m}: {len(ps)} pair tables")
                for p in ps:
                    rows = read(p)
                    t = C["tables"][p.name[:-7]]
                    ok(len(rows) == t["rows"] and len(rows) > 0, f"{p.name}: row count")
                    ok(len({r["unit"] for r in rows}) == len(rows), f"{p.name}: duplicate units")
                    if h == "h2":
                        ok(t["n_within"] == 6 and t["n_across"] == n_scanners - 1, f"{p.name}: sessions")
                    else:
                        ok(t["n_repeats"] == 6, f"{p.name}: repeats")

    for lvl in "ab":
        for m in ("FA", "MD", "MK"):
            p = inputs / f"h3{lvl}_{m}.csv.gz"
            if not p.is_file():
                fail.append(f"missing {p.name}"); continue
            rows = read(p)
            per = Counter((r["subject"], r["unit"], r["den"]) for r in rows)
            ok(set(per.values()) == {n_scanners}, f"{p.name}: (subject, unit, arm) without {n_scanners} scanners")
            units = defaultdict(set)
            for r in rows:
                units[(r["subject"], r["den"])].add(r["unit"])
            subs = {s for s, _ in units}
            ok(len(subs) == n_subjects, f"{p.name}: {len(subs)} subjects")
            common = C["units"][f"H3{lvl}/common/{m}"]["kept_all_subjects"]
            for s in subs:
                ok(units[(s, "0")] == units[(s, "1")], f"{p.name} {s}: unit sets differ between arms")
                ok(units[(s, "0")] == units[(sorted(subs)[0], "0")], f"{p.name} {s}: unit set differs between subjects")
                ok(len(units[(s, "0")]) == common, f"{p.name} {s}: {len(units[(s, '0')])} units, expected {common}")
            t = C["tables"][p.name[:-7]]
            ok(t["cells"] == n_subjects * n_scanners * 2, f"{p.name}: {t['cells']} cells")
    return fail


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--subjects", type=int, default=10)
    p.add_argument("--scanners", type=int, default=6)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--rois", type=int, default=1298)
    a = p.parse_args(argv)
    fail = check(a.inputs, a.subjects, a.scanners, a.pairs, a.rois)
    print("\n".join(fail) if fail else "all structural checks passed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
