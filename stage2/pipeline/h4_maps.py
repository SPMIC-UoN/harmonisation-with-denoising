#!/usr/bin/env python3
"""
h4_maps.py - DESCRIPTIVE MNI maps for H4 per repeat pair (for visualisation; not confirmatory). As h1_maps.py:

Voxel level, in each repeat subject's repeat-template space (all WM voxels; the confirmatory H4a used the
checkerboard; same validity rules):
  off   mean |off-resonance| (Hz) over the 6 repeats and both arms (as build_inputs.build_h4)
  dcov  scan-rescan CoV across the 6 repeats, denoised - non-denoised (percentage points of CoV)
Each map is taken to MNI152 2 mm with the repeat template's FNIRT warp (normalised trilinear interpolation).
ROI level: the confirmatory H4b table values (off_hz, cov_den - cov_nonden) painted onto the 1,298-label atlas.

Outputs (<deriv>/analysis/descriptives/h4_maps/): {vox,roi}_{pair}_{off,dcov_FA,dcov_MD,dcov_MK}_{MNI2mm,MNI1mm}.nii.gz
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

import build_inputs as B
import inputs as I
from h1_maps import MNI2, B_read, applywarp, save


def to_mni(vals: np.ndarray, idx: np.ndarray, ref, warp: Path, out: Path, mni2) -> None:
    ok = np.isfinite(vals)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, v in (("valid", np.ones(ok.sum())), ("val", vals[ok])):
            img = np.zeros(ref.shape[:3], np.float32)
            img.flat[idx[ok]] = v
            nib.save(nib.Nifti1Image(img, ref.affine), str(tmp / f"{name}.nii.gz"))
        wv = applywarp(tmp / "valid.nii.gz", warp, tmp / "valid_mni.nii.gz")
        wd = applywarp(tmp / "val.nii.gz", warp, tmp / "val_mni.nii.gz")
    save(np.where(wv >= 0.5, wd / np.maximum(wv, 1e-6), np.nan), mni2, out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Descriptive H4 maps (|off-resonance|, Delta scan-rescan CoV) in MNI space")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.deriv_dir / "analysis" / "descriptives" / "h4_maps"
    out.mkdir(parents=True, exist_ok=True)
    D = B.Data(a.deriv_dir, a.manifest)
    mni2 = nib.load(str(MNI2))
    atlas_img = nib.load(str(a.deriv_dir / "atlas" / "xtract_sub_1298.nii.gz"))
    atlas = np.asarray(atlas_img.dataobj).astype(int)
    for sub, rsc in sorted(D.repeat_scanner.items()):
        sess = [(s, sc) for s, sc in D.sessions[sub] if sc == rsc]
        ses = [s for s, _ in sess]
        tdir = D.template(sub, "repeat")
        rk = f"{sub}/repeat"
        ref = nib.load(str(tdir / "T_template0_2mm.nii.gz"))
        warp = tdir / "T_to_MNI_warp.nii.gz"
        wm = D.img(tdir / "T_wm_2mm.nii.gz", rk) > 0
        hz = np.concatenate([np.stack([np.abs(D.img(D.fit(arm, sub, s, "repeat", "field_hz"), rk)[wm]) for s in ses])
                             for arm in B.ARMS])
        to_mni(I.h4_offres(hz), np.flatnonzero(wm), ref, warp, out / f"vox_{sub}_off_MNI2mm.nii.gz", mni2)
        for m in B.METRICS:
            u = B.subject_units(D, sub, "repeat", sess, m, {}, "maps")
            c0, c1 = I.missing_in_either_arm(I.cov(u["vals"][B.ARMS[0]]), I.cov(u["vals"][B.ARMS[1]]))
            d = 100 * (c1 - c0)
            d[~u["keep"]] = np.nan
            to_mni(d, u["flat_idx"], ref, warp, out / f"vox_{sub}_dcov_{m}_MNI2mm.nii.gz", mni2)
            t = B_read(a.deriv_dir / "analysis" / "inputs" / f"h4b_{sub}_{m}.csv.gz")
            lab = t["unit"].astype(int)
            for name, v in ((f"dcov_{m}", 100 * (t["cov_den"].astype(float) - t["cov_nonden"].astype(float))),
                            ("off", t["off_hz"].astype(float))):
                lut = np.full(atlas.max() + 1, np.nan, np.float32)
                lut[lab] = v
                save(lut[atlas], atlas_img, out / f"roi_{sub}_{name}_MNI1mm.nii.gz")
        print(f"{sub}: maps written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
