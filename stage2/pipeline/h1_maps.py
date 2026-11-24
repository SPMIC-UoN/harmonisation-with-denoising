#!/usr/bin/env python3
"""
h1_maps.py - DESCRIPTIVE group maps of the H1 cross-scanner CoV in MNI space (for visualisation; not confirmatory).

Voxel level: per subject, CoV across the 6 scanners (repeats averaged) on ALL voxels of the subject WM mask
(the confirmatory H1a used the sparse checkerboard; the same formula and validity rules are used here: missing values, MK range,
mean <= 0 in either arm -> missing). Each subject map (template 2 mm grid) is taken to MNI152 2 mm with the
subject's FNIRT warp (07_masks_rois.py) by normalised trilinear interpolation (warp(value x valid) / warp(valid),
kept where warped validity >= 0.5), then averaged over subjects where at least --min-subjects are valid.
ROI level: the confirmatory H1b CoV values (inputs/h1b_*.csv.gz), averaged per ROI over subjects (ROI kept if present
in at least --min-subjects subjects) and painted onto the 1,298-label atlas (MNI152 1 mm).

Outputs (<deriv>/analysis/descriptives/h1_maps/), CoV in %:
  vox_{M}_{nondenoised,denoised,diff,nsub}_MNI2mm.nii.gz   diff = denoised - non-denoised (percentage points)
  roi_{M}_{nondenoised,denoised,diff,nsub}_MNI1mm.nii.gz
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

import build_inputs as B
import inputs as I

FSLDIR = Path("/usr/local/fsl/6.0.6.x")
MNI2 = FSLDIR / "data/standard/MNI152_T1_2mm_brain.nii.gz"
MNI1 = FSLDIR / "data/standard/MNI152_T1_1mm_brain.nii.gz"
ARMS = B.ARMS


def applywarp(src: Path, warp: Path, out: Path) -> np.ndarray:
    env = dict(os.environ, FSLDIR=str(FSLDIR), FSLOUTPUTTYPE="NIFTI_GZ")
    subprocess.run([str(FSLDIR / "bin/applywarp"), f"--in={src}", f"--ref={MNI2}", f"--warp={warp}",
                    "--interp=trilinear", f"--out={out}"], check=True, env=env, capture_output=True)
    return np.asarray(nib.load(str(out)).dataobj, dtype=np.float64)


def subject_maps(D: B.Data, sub: str, metric: str, tmp: Path) -> dict:
    sess = D.sessions[sub]
    scanners = [sc for _, sc in sess]
    u = B.subject_units(D, sub, "all", sess, metric, {}, "maps")
    cov = {}
    for arm in ARMS:
        per_sc, _ = I.average_by_group(u["vals"][arm][:, u["keep"]], scanners)
        cov[arm] = I.cov(per_sc) * 100
    a, b = I.missing_in_either_arm(cov[ARMS[0]], cov[ARMS[1]])
    ref = nib.load(str(D.template(sub, "all") / "T_template0_2mm.nii.gz"))
    idx = u["flat_idx"][u["keep"]]
    valid = np.isfinite(a)
    out = {}
    vimg = np.zeros(ref.shape[:3], np.float32)
    vimg.flat[idx[valid]] = 1
    vp = tmp / f"{sub}_{metric}_valid.nii.gz"
    nib.save(nib.Nifti1Image(vimg, ref.affine), str(vp))
    warp = D.template(sub, "all") / "T_to_MNI_warp.nii.gz"
    w_valid = applywarp(vp, warp, tmp / f"{sub}_{metric}_valid_mni.nii.gz")
    for arm, x in zip(ARMS, (a, b)):
        img = np.zeros(ref.shape[:3], np.float32)
        img.flat[idx[valid]] = x[valid]
        p = tmp / f"{sub}_{metric}_{arm}.nii.gz"
        nib.save(nib.Nifti1Image(img, ref.affine), str(p))
        w = applywarp(p, warp, tmp / f"{sub}_{metric}_{arm}_mni.nii.gz")
        out[arm] = np.where(w_valid >= 0.5, w / np.maximum(w_valid, 1e-6), np.nan)
    return out


def save(arr, like: nib.Nifti1Image, path: Path):
    hdr = like.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), like.affine, hdr), str(path))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Descriptive H1 CoV group maps in MNI space")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--min-subjects", type=int, default=8)
    p.add_argument("--jobs", type=int, default=10)
    a = p.parse_args(argv)
    out = a.deriv_dir / "analysis" / "descriptives" / "h1_maps"
    out.mkdir(parents=True, exist_ok=True)
    D = B.Data(a.deriv_dir, a.manifest)
    subs = sorted(D.sessions)
    mni2 = nib.load(str(MNI2))
    atlas_img = nib.load(str(a.deriv_dir / "atlas" / "xtract_sub_1298.nii.gz"))
    atlas = np.asarray(atlas_img.dataobj).astype(int)
    for m in B.METRICS:
        with tempfile.TemporaryDirectory() as tmp:
            with ThreadPoolExecutor(a.jobs) as ex:
                maps = list(ex.map(lambda s: subject_maps(D, s, m, Path(tmp)), subs))
        for arm in ARMS + ("diff",):
            stack = np.stack([(x[ARMS[1]] - x[ARMS[0]]) if arm == "diff" else x[arm] for x in maps])
            n = np.isfinite(stack).sum(0)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(stack, axis=0) if stack.size else stack
            mean = np.where(n >= a.min_subjects, mean, np.nan)
            save(mean, mni2, out / f"vox_{m}_{arm}_MNI2mm.nii.gz")
        save(n, mni2, out / f"vox_{m}_nsub_MNI2mm.nii.gz")
        # ROI level from the confirmatory H1b table
        t = B_read(a.deriv_dir / "analysis" / "inputs" / f"h1b_{m}.csv.gz")
        lab, den, cov = t["unit"].astype(int), t["den"].astype(int), t["cov"].astype(float) * 100
        roi = {}
        for k in np.unique(lab):
            s0, s1 = cov[(lab == k) & (den == 0)], cov[(lab == k) & (den == 1)]
            if s0.size >= a.min_subjects:
                roi[k] = (s0.mean(), s1.mean(), (s1 - s0).mean(), s0.size)
        for j, arm in enumerate(ARMS + ("diff", "nsub")):
            vol = np.full(atlas.shape, np.nan, np.float32)
            for k, v in roi.items():
                vol[atlas == k] = v[j]
            save(vol, atlas_img, out / f"roi_{m}_{arm}_MNI1mm.nii.gz")
        print(f"{m}: voxel maps (>= {a.min_subjects} subjects: {int((n >= a.min_subjects).sum())} MNI voxels), "
              f"ROI maps ({len(roi)} ROIs)", flush=True)
    return 0


def B_read(p: Path) -> dict:
    import csv
    import gzip
    with gzip.open(p, "rt") as fh:
        r = csv.reader(fh)
        head = next(r)
        cols = list(zip(*r))
    return {h: np.array(c) for h, c in zip(head, cols)}


if __name__ == "__main__":
    raise SystemExit(main())
