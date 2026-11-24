#!/usr/bin/env python3
"""
h2_maps.py - DESCRIPTIVE maps of the H2 change in bias (Delta Bias = Bias_den - Bias_nonden) per repeat pair, in MNI
space (for visualisation; not confirmatory). As h1_maps.py:

Voxel level: per repeat subject, Bias = mean of the 6 repeats - mean of the 5 other scanners on ALL voxels of the
subject WM mask (confirmatory H2a used the sparse checkerboard; same validity rules), Delta Bias as % of the
pair's mean metric value, taken to MNI152 2 mm with the subject's FNIRT warp (normalised trilinear interpolation).
ROI level: the confirmatory H2b Delta Bias (as % of the pair's mean metric value) painted onto the 1,298-label atlas.

Outputs (<deriv>/analysis/descriptives/h2_maps/): {vox,roi}_{pair}_{M}_dbias_{MNI2mm,MNI1mm}.nii.gz
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

import build_inputs as B
from h1_maps import MNI2, B_read, applywarp, save


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Descriptive H2 Delta Bias maps in MNI space")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.deriv_dir / "analysis" / "descriptives" / "h2_maps"
    out.mkdir(parents=True, exist_ok=True)
    D = B.Data(a.deriv_dir, a.manifest)
    mni2 = nib.load(str(MNI2))
    atlas_img = nib.load(str(a.deriv_dir / "atlas" / "xtract_sub_1298.nii.gz"))
    atlas = np.asarray(atlas_img.dataobj).astype(int)
    per_arm = B_read(a.deriv_dir / "analysis" / "descriptives" / "h2_bias_per_arm.csv.gz")
    for sub, sc in sorted(D.repeat_scanner.items()):
        sess = D.sessions[sub]
        rep = np.array([s == sc for _, s in sess])
        ref = nib.load(str(D.template(sub, "all") / "T_template0_2mm.nii.gz"))
        warp = D.template(sub, "all") / "T_to_MNI_warp.nii.gz"
        for m in B.METRICS:
            u = B.subject_units(D, sub, "all", sess, m, {}, "maps")
            v = {arm: u["vals"][arm][:, u["keep"]] for arm in B.ARMS}
            bias = {arm: v[arm][rep].mean(0) - v[arm][~rep].mean(0) for arm in B.ARMS}
            mean = np.mean([v[arm].mean() for arm in B.ARMS])
            d = 100 * (bias[B.ARMS[1]] - bias[B.ARMS[0]]) / mean
            idx = u["flat_idx"][u["keep"]]
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                for name, vals in (("valid", np.ones(idx.size)), ("val", d)):
                    img = np.zeros(ref.shape[:3], np.float32)
                    img.flat[idx] = vals
                    nib.save(nib.Nifti1Image(img, ref.affine), str(tmp / f"{name}.nii.gz"))
                wv = applywarp(tmp / "valid.nii.gz", warp, tmp / "valid_mni.nii.gz")
                wd = applywarp(tmp / "val.nii.gz", warp, tmp / "val_mni.nii.gz")
            save(np.where(wv >= 0.5, wd / np.maximum(wv, 1e-6), np.nan), mni2, out / f"vox_{sub}_{m}_dbias_MNI2mm.nii.gz")
            s = (per_arm["pair"] == sub) & (per_arm["metric"] == m) & (per_arm["level"] == "ROI")
            lab = per_arm["unit"][s].astype(int)
            dr = 100 * (per_arm["bias_den"][s].astype(float) - per_arm["bias_nonden"][s].astype(float)) \
                / per_arm["metric_mean"][s].astype(float)
            lut = np.full(atlas.max() + 1, np.nan, np.float32)
            lut[lab] = dr
            save(lut[atlas], atlas_img, out / f"roi_{sub}_{m}_dbias_MNI1mm.nii.gz")
        print(f"{sub}: maps written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
