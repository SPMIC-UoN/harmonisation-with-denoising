#!/usr/bin/env python3
"""
h3a_mk_persubject.py - Exploratory (Table S17): input table for voxel-wise H3 of MK (H3a) with validity required
only within each subject.

The registered H3a uses a voxel only if it is valid in all 85 sessions (both arms), which keeps 26.5% of the MNI
white-matter voxels for MK. Here a voxel enters for a subject if its MK is valid in all of THAT subject's sessions
(both arms), so different subjects can contribute different voxels. Everything else is as in build_inputs.py: the
same maps (MK_MNI2mm_valid), one voxel in eight, repeats averaged per scanner, the unit mean subtracted per condition
and one scaling constant shared by both conditions. The registered H3 model is then fitted with run_models.py.

  python h3a_mk_persubject.py --deriv-dir <deriv> --manifest <meta>/sessions_manifest.csv \
      --out <deriv>/analysis/inputs_h3a_mk_persubject

Writes <out>/h3a_MK.csv.gz and <out>/counts.json (voxels kept per subject and number of subjects per voxel).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import build_inputs as B
import inputs as I


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    D = B.Data(a.deriv_dir, a.manifest)
    D.pattern = "lattice"
    lat = D.img(D.deriv / "mmorf" / f"MNI152_2mm_wm_checkerboard-{D.pattern}.nii.gz", "MNI") > 0
    units = np.flatnonzero(lat)
    rows, counts, kept = defaultdict(list), {}, {}
    for sub in sorted(D.sessions):
        ses = [s for s, _ in D.sessions[sub]]
        scanners = [sc for _, sc in D.sessions[sub]]
        vals = {arm: np.stack([D.img(D.deriv / "mmorf" / arm / f"{sub}_{s}" / f"{B.MNI_FILES['MK']}.nii.gz", "MNI")[lat]
                               for s in ses]) for arm in B.ARMS}
        bad = {arm: ~np.isfinite(v) for arm, v in vals.items()}
        assert all((bad[B.ARMS[0]] == b).all() for b in bad.values()), "missing voxels differ between arms"
        keep = ~bad[B.ARMS[0]].any(axis=0)
        kept[sub] = int(keep.sum())
        for j, arm in enumerate(B.ARMS):
            avg, labels = I.average_by_group(vals[arm], scanners)
            for t, sc in enumerate(labels):
                n = kept[sub]
                rows["subject"] += [sub] * n
                rows["scanner"] += [sc] * n
                rows["unit"] += units[keep].tolist()
                rows["arm"] += [j] * n
                rows["y_raw"] += avg[t][keep].tolist()
    B.h3_write(a.out, "h3a_MK", rows, counts)

    per_unit = defaultdict(set)
    for u, s in zip(rows["unit"], rows["subject"]):
        per_unit[u].add(s)
    n_sub = np.array([len(v) for v in per_unit.values()])
    summary = dict(candidates=int(units.size), kept_per_subject=kept,
                   kept_per_subject_pct={s: 100 * k / units.size for s, k in kept.items()},
                   units_in_any_subject=int(n_sub.size), units_in_all_subjects=int((n_sub == len(kept)).sum()),
                   units_by_n_subjects={int(k): int((n_sub == k).sum()) for k in range(1, len(kept) + 1)}, table=counts)
    (a.out / "counts.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("table", "kept_per_subject_pct")}))


if __name__ == "__main__":
    main()
