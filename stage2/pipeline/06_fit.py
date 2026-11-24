#!/usr/bin/env python3
"""
06_fit.py - Template-space diffusion fits: linear mapping of each arm's EDDY-corrected
data into 2 mm subject-template space, bvec rotation, and FSL dtifit fits.

Stages (all CPU; each job has a generated .sh, a .log and a status .json under <out>/_jobs/):
  grid   per template: 2 mm grid and brain mask in FSL's coordinate frame:
           flirt -in T_template0 -ref T_template0 -applyisoxfm 2 -interp spline -> T_template0_2mm
           brain mask = T_template0 > 1e-3, largest component; flirt -applyisoxfm 2 nearest -> 2 mm
  t1reg per template x session (shared by both arms): FLIRT 6 DOF
           T1_unbiased_brain -> T_template0 (1 mm)
  fit    per arm x session:
           1. check that the reverse b0 is the last volume, then drop it from data, bvals and
              EDDY's rotated bvecs together
           2. dMRI -> T1: BRC epi_reg_dof.sh --dof=6 exactly as BRC diff_reg.sh (per arm)
           3. per template of the session (all; repeat): concatenate dMRI->T1 and T1->template,
              resample once with spline into the 2 mm grid, clip negatives, rotate bvecs (bvecs.py)
           4. FA/MD (+ tensor for H3a): dtifit --save_tensor on b0 (b < 50) + b1000 (900-1100);
              MK: dtifit --kurt on all shells
           5. repeat templates only (H4): TOPUP field (Hz) with the same concatenated affine
  nonpos per session x template, once BOTH arms are fitted: validity of the fitted metrics. A voxel is missing (NaN)
         in both arms where, in either arm, the metric is non-finite or (FA, MD) <= 0:
           dti_b0b1000_{FA,MD}_valid.nii.gz, dki_allshells_kurt_valid.nii.gz
         The MK range (0 < MK <= 3) is applied in build_inputs.py and 08_mmorf.py, which also count the exclusions.
         Masks (missing_metric_{FA,MD,MK}) and combined counts (both arms together) in <out>/_missing/
Completion checks read headers and text files only. The maps are written, but never inspected or compared between arms here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bvecs import rotate_bvec_file  # noqa: E402

SCRIPT_VERSION = "1.0.0"
DEFAULT_BRC = Path("/usr/local/BRC_Pipeline/1.6.3")
EXPECTED_FSL = "6.0.6.4"
ARMS = ("nondenoised", "denoised")
B0_MAX = 50
B1000 = (900, 1100)
SHELL_TOL = 100
log = logging.getLogger("06_fit")
_lock = threading.Lock()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def header(p: Path) -> dict:
    """Header only: shape and affine (no voxel data)."""
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rb") as fh:
        h = nib.Nifti1Header.from_fileobj(io.BytesIO(fh.read(348)))
    return {"shape": tuple(int(x) for x in h.get_data_shape()), "affine": h.get_best_affine()}


# --------------------------------------------------------------------------
# Helpers called from the job scripts (text files only)
# --------------------------------------------------------------------------
def helper_drop_revpe(bvals: Path, bvecs: Path, n: int, revpe_bval: float, out_stem: Path) -> None:
    """the reverse b0 must be the last of n+1 volumes; write the first n bvals/bvecs."""
    b = np.loadtxt(bvals, ndmin=1)
    v = np.loadtxt(bvecs, ndmin=2)
    if len(b) != n + 1 or v.shape[1] != n + 1:
        raise SystemExit(f"expected {n + 1} volumes, found {len(b)} bvals / {v.shape[1]} bvecs")
    if b[-1] != revpe_bval or b[-1] >= B0_MAX:
        raise SystemExit(f"last volume b={b[-1]} is not the reverse b0 (expected {revpe_bval})")
    np.savetxt(f"{out_stem}.bval", b[:n][None, :], fmt="%d")
    np.savetxt(f"{out_stem}.bvec", v[:, :n], fmt="%.6f")


BRAIN_THR = 1e-3   # ANTs leaves a ~1e-10..1e-4 numerical background; brain intensities are ~6-9


def helper_brainmask(template: Path, out: Path) -> None:
    """template brain mask = T_template0 > 1e-3, largest connected component (header/affine kept)."""
    from scipy import ndimage
    img = nib.load(str(template))
    x = np.asarray(img.dataobj, dtype=np.float32)
    lab, n = ndimage.label(x > BRAIN_THR)
    if n == 0:
        raise SystemExit(f"{template}: empty brain mask")
    keep = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image((lab == keep).astype(np.uint8), img.affine, hdr), str(out))


def helper_select_b0_b1000(stem: Path, out_stem: Path) -> None:
    """indices of b0 (b < 50) and b1000 (900-1100) volumes; subset bvals/bvecs; index list to .vols."""
    b = np.loadtxt(f"{stem}.bval", ndmin=1)
    v = np.loadtxt(f"{stem}.bvec", ndmin=2)
    idx = [i for i, x in enumerate(b) if x < B0_MAX or B1000[0] <= x <= B1000[1]]
    nb0 = sum(b[i] < B0_MAX for i in idx)
    if nb0 == 0 or len(idx) - nb0 < 30:
        raise SystemExit(f"unexpected selection: {nb0} b0 and {len(idx) - nb0} b1000 volumes")
    other = [x for x in b if not (x < B0_MAX or B1000[0] <= x <= B1000[1] or abs(x - 2000) <= SHELL_TOL)]
    if other:
        raise SystemExit(f"b-values outside the nominal shells: {sorted(set(other))}")
    np.savetxt(f"{out_stem}.bval", b[idx][None, :], fmt="%d")
    np.savetxt(f"{out_stem}.bvec", v[:, idx], fmt="%.6f")
    Path(f"{out_stem}.vols").write_text(",".join(map(str, idx)) + "\n")


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
class Plan:
    def __init__(self, a):
        self.a = a
        self.rows = list(csv.DictReader(open(a.deriv_dir / "inputs" / "preproc_inputs.csv", newline="")))
        self.manifest = {(r["subject"], r["session"]): r
                         for r in csv.DictReader(open(a.meta_dir / "sessions_manifest.csv", newline=""))}
        self.setup = a.brc_dir / "SetUpBRCPipeline.sh"

    def templates_of(self, sub: str, ses: str) -> list[tuple[str, str]]:
        """[(kind, template_id)] for one session: the all-session template, plus the repeat template."""
        m = self.manifest[(sub, ses)]
        out = [("all", f"{sub}/all")]
        if m.get("n_sessions_on_scanner") == "6":
            out.append(("repeat", f"{sub}/repeat-{m['scanner_label']}"))
        return out

    def tdir(self, tid: str) -> Path:
        return self.a.templates_dir / tid

    def t1(self, sub: str, ses: str, name: str) -> Path:
        return self.a.deriv_dir / "brc" / "anat" / f"{sub}_{ses}" / "analysis/anatMRI/T1/processed" / name

    def t1reg_mat(self, tid: str, sub: str, ses: str) -> Path:
        return self.a.out_dir / "_reg" / "T1_to_template" / tid / f"{sub}_{ses}.mat"


def header_script(plan: Plan, key: str) -> str:
    return f"""#!/bin/bash
# Generated by 06_fit.py v{SCRIPT_VERSION} on {utcnow()} - job {key}
source {shlex.quote(str(plan.setup))} > /dev/null 2>&1
set -e -o pipefail
export FSLOUTPUTTYPE=NIFTI_GZ
[ "$(cut -d: -f1 $FSLDIR/etc/fslversion)" = "{EXPECTED_FSL}" ] || {{ echo "FSL is not {EXPECTED_FSL}"; exit 3; }}
PY={shlex.quote(sys.executable)}
SELF={shlex.quote(str(Path(__file__).resolve()))}
set -x
"""


def grid_job(plan: Plan, tid: str) -> dict:
    d = plan.tdir(tid)
    body = f"""cd {d}
flirt -in T_template0 -ref T_template0 -applyisoxfm 2 -interp spline -out T_template0_2mm
$PY $SELF helper brainmask --template {d}/T_template0.nii.gz --out {d}/T_template0_brainmask_1mm.nii.gz
fslmaths T_template0 -mas T_template0_brainmask_1mm T_template0_masked
flirt -in T_template0_brainmask_1mm -ref T_template0 -applyisoxfm 2 -interp nearestneighbour -out T_template0_brainmask_2mm
"""
    return {"stage": "grid", "key": f"grid/{tid}", "body": body,
            "needs": [d / "T_template0.nii.gz"],
            "outputs": [d / "T_template0_2mm.nii.gz", d / "T_template0_brainmask_2mm.nii.gz",
                        d / "T_template0_masked.nii.gz"]}


def t1reg_job(plan: Plan, tid: str, sub: str, ses: str) -> dict:
    mat = plan.t1reg_mat(tid, sub, ses)
    body = f"""mkdir -p {mat.parent}
flirt -in {plan.t1(sub, ses, 'data/T1_unbiased_brain')} -ref {plan.tdir(tid)}/T_template0 -dof 6 -omat {mat} \\
    -out {mat.with_suffix('')}_T1_in_template
"""
    return {"stage": "t1reg", "key": f"t1reg/{tid}/{sub}_{ses}", "body": body,
            "needs": [plan.tdir(tid) / "T_template0.nii.gz", plan.t1(sub, ses, "data/T1_unbiased_brain.nii.gz")],
            "outputs": [mat]}


def fit_job(plan: Plan, row: dict, arm: str) -> dict:
    sub, ses, subj = row["subject"], row["session"], row["brc_subject"]
    n = int(row["main_n_vols"])
    brc = plan.a.deriv_dir / "brc" / arm / subj / "analysis/dMRI"
    W = plan.a.out_dir / arm / subj
    body = f"""rm -rf {W}; mkdir -p {W}
# reverse b0 = last volume; drop it from data, bvals and EDDY's rotated bvecs together
$PY $SELF helper drop-revpe --bvals {brc}/processed/data/bvals \\
    --bvecs {brc}/preproc/eddy/eddy_unwarped_images.eddy_rotated_bvecs --n {n} --revpe-bval {row['revpe_source_bval']} \\
    --out {W}/dwi_main
fslroi {brc}/processed/data/data {W}/dwi_main 0 {n}
# (1) dMRI -> session T1, as BRC diff_reg.sh (nodif = volume 0 = main-run b0), per arm
fslroi {brc}/processed/data/data {W}/nodif 0 1
$BRC_FMRI_SCR/epi_reg_dof.sh --dof=6 --epi={W}/nodif --t1={plan.t1(sub, ses, 'data/T1')} \\
    --t1brain={plan.t1(sub, ses, 'data/T1_unbiased_brain')} --wmseg={plan.t1(sub, ses, 'seg/tissue/sing_chan/T1_WM_mask')} \\
    --out={W}/dmri2T1
"""
    needs = [brc / "processed/data/data.nii.gz", brc / "preproc/eddy/eddy_unwarped_images.eddy_rotated_bvecs",
             plan.t1(sub, ses, "seg/tissue/sing_chan/T1_WM_mask.nii.gz")]
    outputs = [W / "dmri2T1.mat"]
    for kind, tid in plan.templates_of(sub, ses):
        O, T = W / kind, plan.tdir(tid)
        mat = plan.t1reg_mat(tid, sub, ses)
        needs += [mat, T / "T_template0_2mm.nii.gz", T / "T_template0_brainmask_2mm.nii.gz"]
        body += f"""# ---- {kind} template: {tid}
mkdir -p {O}
convert_xfm -omat {O}/dmri2template.mat -concat {mat} {W}/dmri2T1.mat
flirt -in {W}/dwi_main -ref {T}/T_template0_2mm -applyxfm -init {O}/dmri2template.mat -interp spline -out {O}/dwi
fslmaths {O}/dwi -thr 0 {O}/dwi
cp {W}/dwi_main.bval {O}/dwi.bval
$PY $SELF helper rotate --src {W}/dwi_main.bvec --dst {O}/dwi.bvec --mat {O}/dmri2template.mat
$PY $SELF helper select --stem {O}/dwi --out {O}/dwi_b0b1000
fslselectvols -i {O}/dwi -o {O}/dwi_b0b1000 --vols=$(cat {O}/dwi_b0b1000.vols)
dtifit -k {O}/dwi_b0b1000 -m {T}/T_template0_brainmask_2mm -r {O}/dwi_b0b1000.bvec -b {O}/dwi_b0b1000.bval \\
    -o {O}/dti_b0b1000 --save_tensor
dtifit --kurt -k {O}/dwi -m {T}/T_template0_brainmask_2mm -r {O}/dwi.bvec -b {O}/dwi.bval -o {O}/dki_allshells
"""
        outputs += [O / "dwi.nii.gz", O / "dti_b0b1000_FA.nii.gz", O / "dti_b0b1000_MD.nii.gz",
                    O / "dti_b0b1000_tensor.nii.gz", O / "dki_allshells_kurt.nii.gz"]
        if kind == "repeat":
            body += f"""# H4: TOPUP field (Hz) with the same per-arm concatenated affine
flirt -in {brc}/preproc/topup/myfield -ref {T}/T_template0_2mm -applyxfm -init {O}/dmri2template.mat \\
    -interp spline -out {O}/field_hz
"""
            needs.append(brc / "preproc/topup/myfield.nii.gz")
            outputs.append(O / "field_hz.nii.gz")
    return {"stage": "fit", "key": f"fit/{arm}/{subj}", "body": body, "needs": needs, "outputs": outputs,
            "n": n, "row": row, "arm": arm}


def nonpos_session(plan: Plan, row: dict) -> dict:
    """Validity maps per metric, with identical voxel exclusion in both arms."""
    sub, ses, subj = row["subject"], row["session"], row["brc_subject"]
    out = {"key": f"nonpos/{subj}"}
    for kind, tid in plan.templates_of(sub, ses):
        dirs = {arm: plan.a.out_dir / arm / subj / kind for arm in ARMS}
        mask_img = nib.load(str(plan.tdir(tid) / "T_template0_brainmask_2mm.nii.gz"))
        brain = np.asarray(mask_img.dataobj) > 0
        mdir = plan.a.out_dir / "_missing" / subj / kind
        mdir.mkdir(parents=True, exist_ok=True)
        counts = {"brain_voxels": int(brain.sum())}
        for metric, src in (("FA", "dti_b0b1000_FA"), ("MD", "dti_b0b1000_MD"), ("MK", "dki_allshells_kurt")):
            imgs = {arm: nib.load(str(d / f"{src}.nii.gz")) for arm, d in dirs.items()}
            vals = {arm: np.asarray(im.dataobj, dtype=np.float32) for arm, im in imgs.items()}
            miss = np.zeros(brain.shape, bool)
            for v in vals.values():
                miss |= ~np.isfinite(v)
                if metric != "MK":
                    miss |= v <= 0
            miss &= brain
            counts[f"missing_metric_{metric}"] = int(miss.sum())
            nib.save(nib.Nifti1Image(miss.astype(np.uint8), mask_img.affine), str(mdir / f"missing_metric_{metric}.nii.gz"))
            for arm, d in dirs.items():
                img, x = imgs[arm], vals[arm].copy()
                x[miss] = np.nan
                x[~brain] = np.nan
                hdr = img.header.copy()
                hdr.set_data_dtype(np.float32)
                nib.save(nib.Nifti1Image(x, img.affine, hdr), str(d / f"{src}_valid.nii.gz"))
        counts.update(arms_combined=True, generated=utcnow())
        (mdir / "counts.json").write_text(json.dumps(counts, indent=2))
        out[kind] = counts
    return out


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
def verify(plan: Plan, job: dict) -> list[str]:
    errs = [f"missing output {o}" for o in job["outputs"] if not Path(o).exists()]
    if errs or job["stage"] != "fit":
        return errs
    row = job["row"]
    for kind, tid in plan.templates_of(row["subject"], row["session"]):
        O = plan.a.out_dir / job["arm"] / row["brc_subject"] / kind
        g = header(plan.tdir(tid) / "T_template0_2mm.nii.gz")
        for f in ("dwi.nii.gz", "dti_b0b1000_FA.nii.gz", "dti_b0b1000_MD.nii.gz", "dki_allshells_kurt.nii.gz") + \
                (("field_hz.nii.gz",) if kind == "repeat" else ()):
            h = header(O / f)
            if h["shape"][:3] != g["shape"][:3] or not np.allclose(h["affine"], g["affine"], atol=1e-3):
                errs.append(f"{kind}/{f} is not on the 2 mm template grid")
        nv = header(O / "dwi.nii.gz")["shape"]
        if (nv[3] if len(nv) > 3 else 1) != job["n"]:
            errs.append(f"{kind}/dwi.nii.gz has {nv} volumes, expected {job['n']}")
        if np.loadtxt(O / "dwi.bvec", ndmin=2).shape != (3, job["n"]):
            errs.append(f"{kind}/dwi.bvec shape is not 3 x {job['n']}")
    return errs


def status_path(plan: Plan, job: dict) -> Path:
    return plan.a.out_dir / "_jobs" / (job["key"].replace("/", "__") + ".json")


def is_done(plan: Plan, job: dict) -> bool:
    p = status_path(plan, job)
    return p.is_file() and json.loads(p.read_text()).get("state") == "done" and \
        all(Path(o).exists() for o in job["outputs"])


def run_job(plan: Plan, job: dict) -> dict:
    sp = status_path(plan, job)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sh, lg = sp.with_suffix(".sh"), sp.with_suffix(".log")
    sh.write_text(header_script(plan, job["key"]) + job["body"])
    sh.chmod(0o755)
    t0 = time.time()
    sp.write_text(json.dumps({"key": job["key"], "state": "running", "started": utcnow(), "host": platform.node()}))
    with open(lg, "w") as fh:
        rc = subprocess.run(["bash", str(sh)], stdout=fh, stderr=subprocess.STDOUT).returncode
    errs = [f"exit code {rc}"] if rc else verify(plan, job)
    st = {"key": job["key"], "state": "done" if not errs else "failed", "rc": rc, "errors": errs,
          "finished": utcnow(), "runtime_s": round(time.time() - t0), "script": str(sh), "log": str(lg),
          "generated_by": f"06_fit.py v{SCRIPT_VERSION}"}
    sp.write_text(json.dumps(st, indent=2))
    log.log(logging.INFO if not errs else logging.ERROR, "%s %s (%.0f min) %s", st["state"].upper(), job["key"],
            st["runtime_s"] / 60, "; ".join(errs))
    return st


def run_nonpos(plan: Plan, rows: list[dict]) -> int:
    """Apply the missing-value rule to every session whose fits are done in BOTH arms and not yet processed."""
    todo, waiting = [], 0
    for r in rows:
        done = all(is_done(plan, fit_job(plan, r, arm)) for arm in ARMS)
        flag = plan.a.out_dir / "_missing" / r["brc_subject"] / "done.json"
        if not done:
            waiting += 1
        elif not flag.is_file() or plan.a.force:
            todo.append(r)
    log.info("Stage nonpos: %d sessions ready, %d waiting for both arms", len(todo), waiting)
    failed = 0
    if plan.a.dry_run:
        return 0
    with ThreadPoolExecutor(max_workers=max(1, plan.a.jobs // 2)) as ex:
        futs = {ex.submit(nonpos_session, plan, r): r for r in todo}
        for f in as_completed(futs):
            r = futs[f]
            try:
                res = f.result()
                (plan.a.out_dir / "_missing" / r["brc_subject"] / "done.json").write_text(json.dumps(res, indent=2))
                log.info("DONE nonpos/%s %s", r["brc_subject"],
                         {k: tuple(v[f"missing_metric_{m}"] for m in ("FA", "MD", "MK")) for k, v in res.items() if k != "key"})
            except Exception as exc:  # noqa: BLE001 - reported
                failed += 1
                log.error("FAILED nonpos/%s: %s", r["brc_subject"], exc)
    return failed


def execute(plan: Plan, stages: list[str], arms: list[str]) -> int:
    rows = plan.rows
    if plan.a.sessions:
        rows = [r for r in rows if f"{r['subject']}/{r['session']}" in set(plan.a.sessions)]
    tids = sorted({tid for r in rows for _, tid in plan.templates_of(r["subject"], r["session"])})
    stage_jobs = {
        "grid": [grid_job(plan, t) for t in tids],
        "t1reg": [t1reg_job(plan, tid, r["subject"], r["session"]) for r in rows
                  for _, tid in plan.templates_of(r["subject"], r["session"])],
        "fit": [fit_job(plan, r, arm) for arm in arms for r in rows],
    }
    failed = 0
    for stage in ("grid", "t1reg", "fit", "nonpos"):
        if stage == "nonpos":
            if "nonpos" in stages:
                failed += run_nonpos(plan, rows)
            continue
        if stage not in stages:
            continue
        todo, blocked, done = [], [], 0
        for j in stage_jobs[stage]:
            if is_done(plan, j) and not plan.a.force:
                done += 1
            elif [n for n in j["needs"] if not Path(n).exists()]:
                blocked.append(j)
            else:
                todo.append(j)
        log.info("Stage %s: %d jobs, %d done, %d blocked (inputs missing), %d to run", stage,
                 len(stage_jobs[stage]), done, len(blocked), len(todo))
        for j in blocked[:5]:
            log.warning("BLOCKED %s: missing %s", j["key"], [str(n) for n in j["needs"] if not Path(n).exists()][:2])
        if plan.a.dry_run:
            continue
        with ThreadPoolExecutor(max_workers=plan.a.jobs) as ex:
            for f in as_completed([ex.submit(run_job, plan, j) for j in todo]):
                failed += f.result()["state"] == "failed"
    return 1 if failed else 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "helper":
        h = argparse.ArgumentParser(prog="06_fit.py helper")
        sub = h.add_subparsers(dest="cmd", required=True)
        d = sub.add_parser("drop-revpe")
        d.add_argument("--bvals", type=Path, required=True)
        d.add_argument("--bvecs", type=Path, required=True)
        d.add_argument("--n", type=int, required=True)
        d.add_argument("--revpe-bval", type=float, required=True)
        d.add_argument("--out", type=Path, required=True)
        r = sub.add_parser("rotate")
        r.add_argument("--src", type=Path, required=True)
        r.add_argument("--dst", type=Path, required=True)
        r.add_argument("--mat", type=Path, required=True)
        bm = sub.add_parser("brainmask")
        bm.add_argument("--template", type=Path, required=True)
        bm.add_argument("--out", type=Path, required=True)
        s = sub.add_parser("select")
        s.add_argument("--stem", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        x = h.parse_args(argv[1:])
        if x.cmd == "drop-revpe":
            helper_drop_revpe(x.bvals, x.bvecs, x.n, x.revpe_bval, x.out)
        elif x.cmd == "brainmask":
            helper_brainmask(x.template, x.out)
        elif x.cmd == "rotate":
            rotate_bvec_file(x.src, x.dst, x.mat)
        else:
            helper_select_b0_b1000(x.stem, x.out)
        return 0

    p = argparse.ArgumentParser(description="Template-space dMRI fits (FA/MD from b0+b1000, MK from all shells).")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--meta-dir", type=Path, required=True)
    p.add_argument("--templates-dir", type=Path, help="Default: <deriv>/templates")
    p.add_argument("--out-dir", type=Path, help="Default: <deriv>/fits")
    p.add_argument("--stages", nargs="+", choices=("grid", "t1reg", "fit", "nonpos"),
                   default=["grid", "t1reg", "fit", "nonpos"])
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--sessions", nargs="+")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--brc-dir", type=Path, default=DEFAULT_BRC)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.templates_dir = (a.templates_dir or a.deriv_dir / "templates").resolve()
    a.out_dir = (a.out_dir or a.deriv_dir / "fits").resolve()
    (a.out_dir / "_jobs").mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    for h, fmt in ((logging.FileHandler(a.out_dir / "_jobs" / f"06_fit_{dt.datetime.now():%Y%m%dT%H%M%S}.log"),
                    "%(asctime)s %(levelname)-7s %(message)s"),
                   (logging.StreamHandler(sys.stdout), "%(asctime)s %(levelname)-7s %(message)s")):
        h.setFormatter(logging.Formatter(fmt))
        log.addHandler(h)
    log.info("06_fit.py v%s (sha256 %s) pid %d", SCRIPT_VERSION, sha256_of(Path(__file__)), os.getpid())
    log.info("Command: %s", " ".join([sys.executable] + sys.argv))
    return execute(Plan(a), a.stages, a.arms)


if __name__ == "__main__":
    sys.exit(main())
