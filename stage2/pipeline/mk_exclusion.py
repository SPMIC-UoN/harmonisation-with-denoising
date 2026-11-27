#!/usr/bin/env python3
"""
mk_exclusion.py - Descriptive (Supplementary Section 5, Figure S8): why the voxel-wise H3 analysis of MK (H3a)
keeps 26.5% of the candidate voxels.

  python mk_exclusion.py --deriv-dir <deriv> --out <deriv>/analysis/supplementary/mk_exclusion

Writes:
  mk_bins.csv        MK in template space (one WM voxel in eight, all sessions, each arm), % of voxel-sessions per bin
  mk_outside.json    summaries of the values outside 0 < MK <= 3
  per_session.csv    % of MNI candidate voxels invalid for MK in each session (and the share triggered by each arm)
  per_voxel.csv.gz   per MNI candidate voxel: sessions in which it is invalid, which arm triggered it, XTRACT tract,
                     and whether it is kept for H3a FA and MK
  by_tract.csv       % of candidate voxels kept per XTRACT tract (left and right combined), MK and FA
  summary.json       totals used in the text
Needs FSLDIR (XTRACT maximum-probability atlas, $FSLDIR/data/atlases/XTRACT).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine

ARMS = ("nondenoised", "denoised")
BINS = [-np.inf, -1000, -10, -1, -0.5, -0.1, 0, 3, 3.5, 4, 10, 1000, np.inf]
LABELS = ["<= -1,000", "-1,000 to -10", "-10 to -1", "-1 to -0.5", "-0.5 to -0.1", "-0.1 to 0", "0 to 3 (valid)",
          "3 to 3.5", "3.5 to 4", "4 to 10", "10 to 1,000", "> 1,000"]


def load(p):
    return np.asarray(nib.load(str(p)).dataobj)


def template_values(deriv: Path):
    """Raw dtifit --kurt MK at the template-space WM voxels (one in eight) of every session, per arm."""
    vals = {a: [] for a in ARMS}
    for s in sorted((deriv / "fits" / "nondenoised").glob("sub-*_ses-*")):
        sub = s.name.split("_")[0]
        m = load(deriv / "templates" / sub / "all" / "T_wm_checkerboard-lattice_2mm.nii.gz") > 0
        for a in ARMS:
            x = load(deriv / "fits" / a / s.name / "all" / "dki_allshells_kurt.nii.gz")[m].astype(float)
            vals[a].append(x[np.isfinite(x)])
    return {a: np.concatenate(v) for a, v in vals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    D, O = a.deriv_dir, a.out
    O.mkdir(parents=True, exist_ok=True)

    # 1. MK values in template space
    v = template_values(D)
    pd.DataFrame({arm: 100 * np.histogram(x, BINS)[0] / x.size for arm, x in v.items()},
                 index=pd.Index(LABELS, name="MK")).to_csv(O / "mk_bins.csv")
    out = {}
    for arm, x in v.items():
        lo, hi = x[x <= 0], x[x > 3]
        out[arm] = dict(n=int(x.size), pct_le0=100 * lo.size / x.size, pct_gt3=100 * hi.size / x.size,
                        le0_below_m1_pct=100 * np.mean(lo < -1), gt3_above_10_pct=100 * np.mean(hi > 10),
                        gt3_above_1000_pct=100 * np.mean(hi > 1000), pct_3_to_4=100 * np.mean((x > 3) & (x <= 4)))
    (O / "mk_outside.json").write_text(json.dumps(out, indent=1))

    # 2. MNI voxels (H3a candidates): invalid sessions and the arm that triggered the MK range rule
    mm = D / "mmorf"
    lat = load(mm / "MNI152_2mm_wm_checkerboard-lattice.nii.gz") > 0
    cand = np.flatnonzero(lat.ravel())
    ses = sorted(q.name for q in (mm / "nondenoised").glob("sub-*_ses-*"))
    get = lambda s, f: load(mm / "nondenoised" / s / f).ravel()[cand]   # noqa: E731  (identical in both arm folders)
    inv = np.array([np.isnan(get(s, "MK_MNI2mm_valid.nii.gz")) for s in ses])
    tn = np.array([get(s, "trigMK_nondenoised_MNI2mm.nii.gz") > 0 for s in ses]) & inv
    td = np.array([get(s, "trigMK_denoised_MNI2mm.nii.gz") > 0 for s in ses]) & inv
    kept_mk = ~inv.any(0)
    kept_fa = ~np.array([np.isnan(get(s, "FA_MNI2mm_valid.nii.gz")) for s in ses]).any(0)
    only_n, only_d = (tn & ~td).any(0) & ~td.any(0), (td & ~tn).any(0) & ~tn.any(0)
    trig = np.where(kept_mk, "kept", np.where(only_n, "non-denoised only", np.where(only_d, "denoised only", "both arms")))
    # "both arms": MK out of range in each arm (not necessarily in the same session), or excluded by the signal checks
    # that apply to both arms jointly (no MK range trigger in either arm)
    per = pd.DataFrame(dict(session=ses, pct_invalid=100 * inv.mean(1), pct_trig_nondenoised=100 * tn.mean(1),
                            pct_trig_denoised=100 * td.mean(1)))
    per.to_csv(O / "per_session.csv", index=False)

    # 3. XTRACT tract of each candidate voxel (maximum-probability atlas, 1 mm, at the 2 mm voxel centre)
    xd = Path(os.environ["FSLDIR"]) / "data" / "atlases"
    xi = nib.load(str(xd / "XTRACT" / "xtract-tract-atlases-maxprob5-1mm.nii.gz"))
    names = {int(i) + 1: n.strip() for i, n in re.findall(r'<label index="(\d+)"[^>]*>([^<]*)', (xd / "XTRACT.xml").read_text())}
    ijk = np.array(np.unravel_index(cand, lat.shape)).T
    w = np.rint(apply_affine(np.linalg.inv(xi.affine), apply_affine(nib.load(str(mm / "MNI152_2mm_wm_checkerboard-lattice.nii.gz")).affine, ijk))).astype(int)
    lab = np.asarray(xi.dataobj).astype(int)[w[:, 0], w[:, 1], w[:, 2]]
    tract = pd.Series([names.get(l, "outside XTRACT") for l in lab]).str.replace(r" [LR]$", "", regex=True)
    vox = pd.DataFrame(dict(unit=cand, n_invalid=inv.sum(0), trigger=trig, tract=tract, kept_MK=kept_mk, kept_FA=kept_fa))
    vox.to_csv(O / "per_voxel.csv.gz", index=False)
    vox.groupby("tract").agg(candidates=("unit", "size"), kept_MK_pct=("kept_MK", lambda x: 100 * x.mean()),
                             kept_FA_pct=("kept_FA", lambda x: 100 * x.mean())).sort_values("kept_MK_pct").to_csv(O / "by_tract.csv")

    s = dict(candidates=int(cand.size), kept=int(kept_mk.sum()), kept_pct=100 * kept_mk.mean(),
             invalid_per_session_mean=float(per.pct_invalid.mean()), invalid_per_session_min=float(per.pct_invalid.min()),
             invalid_per_session_max=float(per.pct_invalid.max()), excluded_in_one_session=int((inv.sum(0) == 1).sum()),
             trigger=vox.trigger.value_counts().to_dict(),
             signal_checks_only=int((~kept_mk & ~(tn | td).any(0)).sum()))
    (O / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
