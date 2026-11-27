#!/usr/bin/env python3
"""
07_masks_rois.py - Per-template WM masks, checkerboard masks and XTRACT ROIs in 2 mm template space
.

Per template (<deriv>/templates/<sub>/{all,repeat-<scanner>}; the 2 mm grid comes from 06_fit.py):
  wm     FAST (fast -t 1 -n 3) on T_template0 masked by the template brain mask (1 mm); WM PVE -> 2 mm grid (flirt -applyxfm identity,
         trilinear, FLIRT frame as 06_fit.py); WM mask = PVE >= 0.5 AND template brain mask
         sparse checkerboard on the 2 mm grid indices (i, j, k all even: 1 in 8 voxels, the sampling behind the
         6,800 voxels of Section 3.6) and WM x sparse checkerboard
  checkerboard  masks only: the checkerboard from the existing WM mask (FAST/ROIs untouched)
  mni    flirt -dof 12 (T_template0 -> MNI152_T1_1mm_brain), fnirt --config=T1_2_MNI152_2mm
         (ref MNI152_T1_2mm_brain, refmask MNI152_T1_2mm_brain_mask_dil), invwarp (MNI -> template)
  rois   each of the atlas labels -> binary -> applywarp (inverse warp, trilinear) onto the 2 mm
         grid -> >= 0.5 -> AND WM mask; ties (>= 0.5 in more than one ROI) excluded and counted;
         outputs a label image and per-ROI voxel counts

The atlas is verified against its pinned SHA-256. Only T1-template-derived images and atlas
images are read; no diffusion data or fitted maps are touched.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_VERSION = "1.1.0"
FSLDIR = Path("/usr/local/fsl/6.0.6.x")                                  # FSL 6.0.6.4 (checked)
ATLAS_SHA256 = "d3adee781d5e3f399fec4d7a0c0a63de3f1f112c8c32e70c2ef57f97b271412c"   # 1,298-label copy
PVE_THR = 0.5
ROI_THR = 0.5


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fsl(cmds: str, log: Path) -> None:
    """Run FSL commands (FSL 6.0.6.4 via the BRC set-up) and append their output to log."""
    env = dict(os.environ, FSLDIR=str(FSLDIR), FSLOUTPUTTYPE="NIFTI_GZ",
               PATH=f"{FSLDIR}/share/fsl/bin:{FSLDIR}/bin:" + os.environ.get("PATH", ""))
    script = f"set -e -o pipefail\nset -x\n{cmds}\n"
    with open(log, "a") as fh:
        rc = subprocess.run(["bash", "-c", script], stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc:
        raise RuntimeError(f"FSL commands failed ({rc}); see {log}")


CB_PATTERN = "lattice"                   # sparse checkerboard: i, j, k all even (1 in 8; Section 3.6, ~6,800 voxels)
CB_PATTERNS = (CB_PATTERN, "parity")     # parity: dense checkerboard (1 in 2), for Table S20


def checkerboard(shape: tuple, pattern: str) -> np.ndarray:
    """lattice: i, j, k all even (1 in 8, used); parity: (i+j+k) even (~50%, no face-adjacent kept voxels); block: pilot 2x2x2 block parity ((i//2+j//2+k//2) even, ~50%, face neighbours within blocks)."""
    i, j, k = np.meshgrid(*(np.arange(n) for n in shape[:3]), indexing="ij")
    if pattern == "parity":
        return (i + j + k) % 2 == 0
    if pattern == "block":
        return ((i // 2 + j // 2 + k // 2) % 2 == 0)
    if pattern == "lattice":
        return (i % 2 == 0) & (j % 2 == 0) & (k % 2 == 0)
    raise ValueError(pattern)


def save_like(arr: np.ndarray, ref: Path, out: Path, dtype) -> None:
    img = nib.load(str(ref))
    hdr = img.header.copy()
    hdr.set_data_dtype(dtype)
    hdr.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(arr.astype(dtype), img.affine, hdr), str(out))


# --------------------------------------------------------------------------
def stage_wm(d: Path, log: Path) -> dict:
    fsl(f"""cd {d}
fast -t 1 -n 3 -o T_template0_fast T_template0_masked
flirt -in T_template0_fast_pve_2 -ref T_template0_2mm -applyxfm -init $FSLDIR/etc/flirtsch/ident.mat \\
    -interp trilinear -out T_template0_wmpve_2mm
fslmaths T_template0_wmpve_2mm -thr {PVE_THR} -bin -mas T_template0_brainmask_2mm T_wm_2mm -odt char
""", log)
    return {"wm_voxels": int((np.asarray(nib.load(str(d / "T_wm_2mm.nii.gz")).dataobj) > 0).sum()),
            **stage_checkerboard(d)}


def stage_checkerboard(d: Path) -> dict:
    """Masks only: sparse (and dense) checkerboard of the existing T_wm_2mm; FAST, WM mask and ROIs are not touched."""
    wm = np.asarray(nib.load(str(d / "T_wm_2mm.nii.gz")).dataobj) > 0
    out = {"pattern": CB_PATTERN, "wm_voxels": int(wm.sum())}
    for pat in CB_PATTERNS:
        cb = checkerboard(wm.shape, pat)
        save_like(cb, d / "T_template0_2mm.nii.gz", d / f"T_checkerboard-{pat}_2mm.nii.gz", np.uint8)
        save_like(wm & cb, d / "T_template0_2mm.nii.gz", d / f"T_wm_checkerboard-{pat}_2mm.nii.gz", np.uint8)
        out[f"wm_checkerboard_{pat}_voxels"] = int((wm & cb).sum())
    return out


def stage_mni(d: Path, log: Path) -> dict:
    fsl(f"""cd {d}
flirt -in T_template0 -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -dof 12 -omat T_to_MNI_affine.mat
fnirt --in=T_template0 --aff=T_to_MNI_affine.mat --config=T1_2_MNI152_2mm \\
    --ref=$FSLDIR/data/standard/MNI152_T1_2mm_brain --refmask=$FSLDIR/data/standard/MNI152_T1_2mm_brain_mask_dil \\
    --cout=T_to_MNI_warp --iout=T_in_MNI_2mm --logout=T_to_MNI_fnirt.log
invwarp --ref=T_template0 --warp=T_to_MNI_warp --out=MNI_to_T_warp
""", log)
    return {}


def stage_rois(d: Path, atlas: Path, log: Path, threads: int) -> dict:
    if sha256_of(atlas) != ATLAS_SHA256:
        raise RuntimeError(f"atlas {atlas} does not match the pinned sha256")
    lab = np.asarray(nib.load(str(atlas)).dataobj)
    labels = [int(x) for x in np.unique(lab) if x > 0]
    ref = d / "T_template0_2mm.nii.gz"
    wm = np.asarray(nib.load(str(d / "T_wm_2mm.nii.gz")).dataobj) > 0
    shape = wm.shape
    with tempfile.TemporaryDirectory(dir=d) as tmp:
        tmp = Path(tmp)

        def warp_one(k: int) -> tuple[int, np.ndarray]:
            roi, out = tmp / f"roi{k}.nii.gz", tmp / f"roi{k}_2mm.nii.gz"
            save_like(lab == k, atlas, roi, np.uint8)
            fsl(f"applywarp --in={roi} --ref={ref} --warp={d}/MNI_to_T_warp --interp=trilinear --out={out} -d float",
                tmp / f"roi{k}.log")
            v = np.asarray(nib.load(str(out)).dataobj) >= ROI_THR
            roi.unlink(); out.unlink()
            return k, v

        hits = np.zeros(shape, np.int16)
        label_img = np.zeros(shape, np.int32)
        raw_counts = {}
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for f in as_completed([ex.submit(warp_one, k) for k in labels]):
                k, v = f.result()
                raw_counts[k] = int(v.sum())
                hits += v
                label_img[v] = k
    ties = hits > 1
    label_img[ties] = 0
    label_img[~wm] = 0
    save_like(label_img, ref, d / "T_xtract_rois_2mm.nii.gz", np.int32)
    counts = np.bincount(label_img.ravel(), minlength=max(labels) + 1)
    with open(d / "T_xtract_rois_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["label", "voxels_ge_0.5", "voxels_in_wm_no_ties"])
        for k in labels:
            w.writerow([k, raw_counts[k], int(counts[k])])
    return {"rois": len(labels), "tie_voxels": int(ties.sum()), "empty_rois_in_wm": int(sum(counts[k] == 0 for k in labels))}


def process(d: Path, atlas: Path, stages: list[str], threads: int, force: bool) -> dict:
    st_p = d / "masks_rois_status.json"
    st = json.loads(st_p.read_text()) if st_p.is_file() else {}
    log = d / "07_masks_rois.log"
    for stage, fn in (("wm", lambda: stage_wm(d, log)), ("checkerboard", lambda: stage_checkerboard(d)),
                      ("mni", lambda: stage_mni(d, log)), ("rois", lambda: stage_rois(d, atlas, log, threads))):
        if stage not in stages or (st.get(stage, {}).get("state") == "done" and not force):
            continue
        t0 = time.time()
        try:
            info = fn()
            st[stage] = {"state": "done", "finished": utcnow(), "runtime_s": round(time.time() - t0), **info}
        except Exception as exc:  # noqa: BLE001 - recorded, stops this template
            st[stage] = {"state": "failed", "error": str(exc), "finished": utcnow()}
            st_p.write_text(json.dumps(st, indent=2))
            return {"template": str(d), "state": "failed", "stage": stage, "error": str(exc)}
        st["generated_by"] = f"07_masks_rois.py v{SCRIPT_VERSION}"
        st_p.write_text(json.dumps(st, indent=2))
    return {"template": str(d), "state": "done", **{k: v for k, v in st.items() if isinstance(v, dict)}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="WM masks, checkerboards and XTRACT ROIs per subject template.")
    p.add_argument("--templates-dir", type=Path, required=True, help="<deriv>/templates")
    p.add_argument("--atlas", type=Path, required=True, help="Pinned xtract_sub.nii.gz")
    p.add_argument("--only", nargs="+", help="Template ids, e.g. sub-16981/all")
    p.add_argument("--stages", nargs="+", choices=("wm", "checkerboard", "mni", "rois"),
                   default=["wm", "mni", "rois"])
    p.add_argument("--parallel-templates", type=int, default=3)
    p.add_argument("--threads", type=int, default=6, help="Parallel ROI warps per template")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    dirs = sorted(x.parent for x in a.templates_dir.glob("sub-*/*/T_template0.nii.gz"))
    if a.only:
        dirs = [x for x in dirs if str(x.relative_to(a.templates_dir)) in set(a.only)]
    missing = [x for x in dirs if not (x / "T_template0_2mm.nii.gz").is_file()]
    if missing:
        print(f"2 mm grid missing (run 06_fit.py --stages grid first): {[str(m) for m in missing]}", file=sys.stderr)
        return 2
    ver = (FSLDIR / "etc/fslversion").read_text().split(":")[0].strip()
    if ver != "6.0.6.4":
        print(f"FSL at {FSLDIR} is {ver}, expected 6.0.6.4", file=sys.stderr)
        return 2
    print(f"{utcnow()} 07_masks_rois.py v{SCRIPT_VERSION} (sha256 {sha256_of(Path(__file__))}); {len(dirs)} templates")
    rc = 0
    with ThreadPoolExecutor(max_workers=a.parallel_templates) as ex:
        for f in as_completed([ex.submit(process, d, a.atlas, a.stages, a.threads, a.force) for d in dirs]):
            r = f.result()
            print(json.dumps(r, default=str))
            rc |= r["state"] != "done"
    return rc


if __name__ == "__main__":
    sys.exit(main())
