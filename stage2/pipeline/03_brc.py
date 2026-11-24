#!/usr/bin/env python3
"""
03_brc.py - Job runner for the BRC 1.6.3 pipelines (struc_preproc.sh --qc and
dMRI_preproc.sh --qc) over all sessions and both arms, plus the LR sign test for
the custom-TOPUP session.

Stages (per session, from <deriv-dir>/inputs/preproc_inputs.csv):
  struc    struc_preproc.sh --input <T1w> --path <deriv>/brc/anat --subject <sub_ses> --qc
  d14test custom-TOPUP sessions only , non-denoised arm: BRC's own
           data_copy.sh + basic_preproc.sh build Pos_Neg_b0 exactly as BRC would,
           then BRC's run_topup.sh runs once per x sign of the LR acqparams row.
           Reports TOPUP's final cost and the AP-LR mismatch of the corrected b0s
           (mean |AP - LR| in the common brain mask) for the unblinded check.
  dmri     dMRI_preproc.sh --input <main> --input_2 <reverse b0> --path <deriv>/brc/<arm>
           --subject <sub_ses> --echospacing <sidecar EES> --pe_dir 2 --qc
           Custom-TOPUP sessions : BRC sub-stages with the confirmed custom
           TOPUP (--usetopuppath) and eddy/acqparams.txt overwritten between
           parts 1 and 2; blocked until the LR sign is final in session_exceptions.csv.

Runner rules:
  - Each arm has its own BRC --path; job keys, logs and status files include the
    arm; the denoised arm only takes DESIGNER outputs whose provenance is "ok"
    and whose SHA-256 matches, the non-denoised arm only raw BIDS files.
  - dmri jobs run at most --gpu-jobs at a time (each has one EDDY). On the system used here a local
    wrapper (migtools) gave each EDDY its own GPU instance; adapt this to your own GPU set-up.
  - Completed sessions (status "done" and outputs present) are skipped; failed or
    stale ones are re-run from scratch. Each job has a generated .sh, a .log and a
    status .json under <deriv>/brc/_jobs/.

BLINDING: the runner reads no voxel values itself (completion checks use NIfTI
headers). The LR sign test metrics are computed by FSL on raw b0s (routine QC). BRC's
automatic dtifit outputs are produced but not inspected here.
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
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib

SCRIPT_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS = Path(__file__).resolve().parent / "session_exceptions.csv"
DEFAULT_BRC = Path("/usr/local/BRC_Pipeline/1.6.3")
EXPECTED_BRC = "Ver1.6.3"
EXPECTED_FSL = "6.0.6.4"
ARMS = ("nondenoised", "denoised")

log = logging.getLogger("03_brc")
_status_lock = threading.Lock()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def nifti_shape(path: Path) -> tuple:
    """Header-only read of the image dimensions."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:
        raw = fh.read(348)
    return tuple(int(x) for x in nib.Nifti1Header.from_fileobj(io.BytesIO(raw)).get_data_shape())


def noext(path: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", path)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
@dataclass
class Job:
    stage: str                 # struc | d14test | dmri
    arm: str                   # anat | nondenoised | denoised
    row: dict
    kind: str                  # cpu | gpu
    body: str = ""             # bash commands (after the common header)
    outputs: list = field(default_factory=list)
    blocked: str = ""          # reason the job cannot run yet
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.stage}/{self.arm}/{self.row['brc_subject']}"


class Runner:
    def __init__(self, args):
        self.a = args
        self.brc_root = args.deriv_dir / "brc"
        self.jobs_dir = self.brc_root / "_jobs"
        self.setup = args.brc_dir / "SetUpBRCPipeline.sh"
        self.decisions = list(csv.DictReader(open(args.decisions, newline="", encoding="utf-8")))
        self.manifest = {(r["subject"], r["session"]): r
                         for r in csv.DictReader(open(args.meta_dir / "sessions_manifest.csv", newline=""))}

    # ---- paths -----------------------------------------------------------
    def job_paths(self, job: Job) -> dict:
        base = self.jobs_dir / job.stage / job.arm / job.row["brc_subject"]
        return {"sh": base.with_suffix(".sh"), "log": base.with_suffix(".log"), "status": base.with_suffix(".json")}

    def d14_decision(self, row) -> dict | None:
        for d in self.decisions:
            if d["decision_id"] == "D14" and d["subject"] == row["subject"] and d["session"] == row["session"]:
                return d
        return None

    # ---- writable input copies ---------------------------------------------
    def writable_copy(self, job_arm: str, subj: str, image: str) -> tuple[str, str, str]:
        """Raw BIDS files are read-only, and BRC copies its inputs with imcp (which keeps the mode) and then
        edits the copies in place. So raw inputs are first copied, byte-identical (cmp), into a writable job
        folder; the copy is removed after the job succeeds. Returns (bash lines, image to pass, cleanup line)."""
        if not Path(image).is_relative_to(self.a.bids_dir):
            return "", image, ""
        dst_dir = self.brc_root / "_inputs" / job_arm / subj
        lines = f"rm -rf {dst_dir}; mkdir -p {dst_dir}\n"
        stem = noext(image)
        for ext in (".nii.gz", ".bval", ".bvec"):
            src = Path(stem + ext)
            if src.is_file() or ext == ".nii.gz":
                dst = dst_dir / src.name
                lines += f"install -m 0644 {src} {dst} && cmp {src} {dst}\n"
        return lines, str(dst_dir / Path(image).name), f"rm -rf {dst_dir}\n"

    # ---- script header ----------------------------------------------------
    def header(self, job: Job) -> str:
        return f"""#!/bin/bash
# Generated by 03_brc.py v{SCRIPT_VERSION} on {utcnow()} - job {job.key}
source {shlex.quote(str(self.setup))} > /dev/null 2>&1
set -e -o pipefail
unset CUDA_VISIBLE_DEVICES           # the local GPU wrapper (migtools) assigns a free GPU instance to EDDY
export FSLOUTPUTTYPE=NIFTI_GZ
echo "BRCDIR=$BRCDIR  FSLDIR=$FSLDIR  FSL=$(cat $FSLDIR/etc/fslversion)  host=$(hostname)"
if [ "$(cut -d: -f1 $FSLDIR/etc/fslversion)" != "{EXPECTED_FSL}" ] || [ "$(cat $BRCDIR/version.txt)" != "{EXPECTED_BRC}" ]; then
    echo "ERROR: FSL/BRC version is not the pinned {EXPECTED_FSL} / {EXPECTED_BRC}"; exit 3
fi
set -x
"""

    # ---- stage builders ---------------------------------------------------
    def struc_job(self, row) -> Job:
        m = self.manifest[(row["subject"], row["session"])]
        t1 = self.a.bids_dir / m["t1w_nii"]
        anat = self.brc_root / "anat"
        subj = row["brc_subject"]
        cp, t1_in, clean = self.writable_copy("anat", subj, str(t1))
        body = cp + f"""mkdir -p {anat}
rm -rf {anat}/{subj}
struc_preproc.sh --input {t1_in} --path {anat} --subject {subj} --qc
""" + clean
        data = anat / subj / "analysis" / "anatMRI" / "T1" / "processed" / "data"
        return Job("struc", "anat", row, "cpu", body,
                   outputs=[data / "T1.nii.gz", data / "T1_unbiased.nii.gz", data / "T1_unbiased_brain.nii.gz"])

    def main_input(self, row, arm) -> tuple[str, str]:
        """(image path, reason blocked)."""
        if arm == "nondenoised":
            path = row["main_dwi_nii"]
            if not Path(path).is_relative_to(self.a.bids_dir):
                return path, f"non-denoised input {path} is not a raw BIDS file"
            return path, ""
        out = Path(row["designer_output"])
        prov = Path(noext(str(out)) + "_provenance.json")
        if not prov.is_file():
            return str(out), "DESIGNER output not available yet (02_denoise.py)"
        p = json.loads(prov.read_text())
        if p.get("status") != "ok":
            return str(out), "DESIGNER provenance status is not ok"
        if not out.is_file() or p["output"]["sha256"] != self.designer_sha(out):
            return str(out), "DESIGNER output does not match its provenance SHA-256"
        return str(out), ""

    _sha_cache: dict = {}

    def designer_sha(self, path: Path) -> str:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self._sha_cache:
            self._sha_cache[key] = sha256_of(path)
        return self._sha_cache[key]

    def dmri_outputs(self, arm, subj) -> list:
        d = self.brc_root / arm / subj / "analysis" / "dMRI"
        return [d / "processed/data/data.nii.gz", d / "processed/data/bvals", d / "processed/data/bvecs",
                d / "processed/data/nodif_brain_mask.nii.gz", d / "preproc/topup/myfield.nii.gz",
                d / "preproc/topup/topup_Pos_Neg_b0_fieldcoef.nii.gz",
                d / "preproc/eddy/eddy_unwarped_images.eddy_movement_rms", d / "preproc/qc/qc.json"]

    def dmri_job(self, row, arm) -> Job:
        subj = row["brc_subject"]
        main, blocked = self.main_input(row, arm)
        rev = row["revpe_b0_nii"]
        arm_path = self.brc_root / arm
        job = Job("dmri", arm, row, "gpu", outputs=self.dmri_outputs(arm, subj), blocked=blocked,
                  extra={"main_input": main, "revpe_b0": rev})
        cp, main_in, clean = self.writable_copy(arm, subj, main)
        if row["brc_invocation"] == "dMRI_preproc.sh":
            job.body = cp + f"""mkdir -p {arm_path}
rm -rf {arm_path}/{subj}
dMRI_preproc.sh --input {noext(main_in)} --input_2 {noext(rev)} --path {arm_path} --subject {subj} \\
    --echospacing {row['echospacing_arg']} --pe_dir 2 --qc
""" + clean
            return job
        # Custom TOPUP route : requires the confirmed LR sign.
        d14 = self.d14_decision(row)
        sign = None
        if d14 and d14["status"] == "final":
            m = re.search(r"lr_x_sign=([+-]1)", d14["action"])
            sign = m.group(1) if m else None
        if sign is None:
            job.blocked = job.blocked or ("LR sign not final in session_exceptions.csv (needs action 'lr_x_sign=+1' or "
                                          "'lr_x_sign=-1', status final); run the d14test stage first")
            return job
        job.extra["lr_x_sign"] = sign
        job.body = cp + self.custom_body(row, arm, main_in, rev, sign) + clean
        return job

    def acqparams_text(self, row, sign: str) -> str:
        return f"0 1 0 {row['readout_time_used_s']}\n{int(sign)} 0 0 {row['revpe_readout_time_used_s']}\n"

    def prep_commands(self, W: Path, main: str, rev: str, row) -> str:
        """BRC's own data_copy.sh + basic_preproc.sh into W/prep (unmodified scripts)."""
        return f"""rm -rf {W}; mkdir -p {W}/prep/raw {W}/prep/eddy {W}/prep/topup
touch {W}/prep/log.txt
$BRC_DMRI_SCR/data_copy.sh --dmrirawfolder={W}/prep/raw --eddyfolder={W}/prep/eddy --inputimage={noext(main)} \\
    --inputimage2={noext(rev)} --pedirection=2 --applytopup=yes --donoddi=no --dodki=no --dowmti=no --dofwdti=no \\
    --logfile={W}/prep/log.txt
$BRC_DMRI_SCR/basic_preproc.sh --dmrirawfolder={W}/prep/raw --topupfolder={W}/prep/topup --eddyfolder={W}/prep/eddy \\
    --echospacing={row['echospacing_arg']} --pedir=2 --b0dist=150 --b0maxbval=50 --pifactor=1 --applytopup=yes \\
    --domppca=no --logfile={W}/prep/log.txt
"""

    def custom_body(self, row, arm, main, rev, sign) -> str:
        subj = row["brc_subject"]
        W = self.brc_root / "_custom" / arm / subj
        acq = self.acqparams_text(row, sign).replace("\n", "\\n")
        P = self.brc_root / arm / subj
        dm = P / "analysis/dMRI"
        at = P / "analysis/anatMRI/T1"
        return self.prep_commands(W, main, rev, row) + f"""# Custom TOPUP with the confirmed acqparams (LR x sign {sign}). Same topup command and config as BRC's
# run_topup.sh. BRC's hifib0 step (applytopup least-squares) cannot combine orthogonal PE directions, so
# hifib0 = applytopup --method=jac on the main-run AP b0 alone; then bet -f 0.20 as in run_topup.sh.
mkdir -p {W}/topup
cp {W}/prep/topup/* {W}/topup/
printf '{acq}' > {W}/acqparams_confirmed.txt
cp {W}/acqparams_confirmed.txt {W}/topup/acqparams.txt
cd {W}/topup
topup --imain={W}/topup/Pos_Neg_b0 --datain={W}/topup/acqparams.txt --config=$BRC_GLOBAL_DIR/config/b02b0.cnf.txt \\
    --fout={W}/topup/myfield --out={W}/topup/topup_Pos_Neg_b0 -v
fslroi {W}/topup/Pos_b0 {W}/topup/Pos_b01 0 1
applytopup --imain={W}/topup/Pos_b01 --inindex=1 --datain={W}/topup/acqparams.txt \\
    --topup={W}/topup/topup_Pos_Neg_b0 --method=jac --out={W}/topup/hifib0
imrm {W}/topup/Pos_b0* {W}/topup/Neg_b0*
bet {W}/topup/hifib0 {W}/topup/nodif_brain -m -f 0.20
cd - > /dev/null
# BRC stages exactly as dMRI_preproc.sh would call them (folder set-up mirrors dMRI_preproc.sh)
mkdir -p {self.brc_root / arm}
rm -rf {P}; mkdir -p {P}
mkdir -p {P}/raw/dMRI {dm}/log {dm}/preproc/topup {dm}/preproc/eddy {dm}/preproc/tbss {dm}/preproc/qc {dm}/processed/data
$BRCDIR/Show_version.sh --showdiff="no" --logfile={dm}/log/log.txt
START=$(date -u +%s)
$BRC_DMRI_SCR/dMRI_preproc_part_1.sh --dmrirawfolder={P}/raw/dMRI --eddyfolder={dm}/preproc/eddy \\
    --topupfolder={dm}/preproc/topup --inputimage={noext(main)} --inputimage2={noext(rev)} --pedirection=2 \\
    --applytopup=yes --echospacing={row['echospacing_arg']} --b0dist=150 --b0maxbval=50 --pifactor=1 --hires=no \\
    --donoddi=no --dodki=no --dowmti=no --dofwdti=no --domppca=no --usetopuppath={W}/topup \\
    --logfile={dm}/log/log.txt
# EDDY must model the LR b0 with the confirmed acqparams
cp {W}/acqparams_confirmed.txt {dm}/preproc/eddy/acqparams.txt
cp {W}/acqparams_confirmed.txt {dm}/preproc/topup/acqparams.txt
$BRC_DMRI_SCR/dMRI_preproc_part_2.sh --eddyfolder={dm}/preproc/eddy --topupfolder={dm}/preproc/topup \\
    --applytopup=yes --doqc=yes --qcdir={dm}/preproc/qc --slice2vol=no --slspec=NONE --movebysuscept=no --hires=no \\
    --logfile={dm}/log/log.txt
$BRC_DMRI_SCR/dMRI_preproc_part_3.sh --workingdir={dm} --eddyfolder={dm}/preproc/eddy \\
    --datafolder={dm}/processed/data --combinematched=2 --applytopup=yes --doreg=no --dotbss=no \\
    --tbssfolder={dm}/preproc/tbss --multchant1folder={at}/processed/seg/tissue/multi_chan \\
    --sinchant1folder={at}/processed/seg/tissue/sing_chan --regfolder={dm}/preproc/reg --t1={at}/processed/data/T1 \\
    --t1restore={at}/processed/data/T1_unbiased --t1brain={at}/processed/data/T1_unbiased_brain --dof=6 \\
    --datat1folder={at}/processed/data --regt1folder={at}/preproc/reg --outstr={dm}/processed/data2str \\
    --outstd={dm}/processed/data2std --start=$START --subject={subj} --hires=no --donoddi=no --dodki=no \\
    --dowmti=no --dofwdti=no --b0maxbval=50 --dtimaxshell=1500 --logfile={dm}/log/log.txt
"""

    def d14_job(self, row) -> Job:
        subj = row["brc_subject"]
        main, blocked = self.main_input(row, "nondenoised")
        W = self.brc_root / "_d14" / subj
        cp, main_in, clean = self.writable_copy("d14test", subj, main)
        body = cp + self.prep_commands(W / "work", main_in, row["revpe_b0_nii"], row)
        for name, sign in (("plus", "+1"), ("minus", "-1")):
            C = W / f"cand_{name}"
            acq = self.acqparams_text(row, sign).replace("\n", "\\n")
            body += f"""# Candidate LR x sign {sign}. TOPUP command identical to BRC run_topup.sh; its hifib0 step (applytopup
# least-squares) cannot combine orthogonal PE directions, so each b0 is corrected with the Jacobian method instead.
mkdir -p {C}
cp {W}/work/prep/topup/Pos_Neg_b0.nii.gz {W}/work/prep/topup/extractedb0.txt {C}/
printf '{acq}' > {C}/acqparams.txt
topup --imain={C}/Pos_Neg_b0 --datain={C}/acqparams.txt --config=$BRC_GLOBAL_DIR/config/b02b0.cnf.txt \\
    --fout={C}/myfield --out={C}/topup_Pos_Neg_b0 -v > {C}/topup_stdout.txt 2>&1
applytopup --imain={W}/work/prep/topup/Pos_b0 --inindex=1 --datain={C}/acqparams.txt \\
    --topup={C}/topup_Pos_Neg_b0 --method=jac --out={C}/corr_AP
applytopup --imain={W}/work/prep/topup/Neg_b0 --inindex=2 --datain={C}/acqparams.txt \\
    --topup={C}/topup_Pos_Neg_b0 --method=jac --out={C}/corr_LR
bet {C}/corr_AP {C}/nodif_brain -m -f 0.20
fslmaths {C}/corr_AP -sub {C}/corr_LR -abs {C}/absdiff_AP_LR
"""
        body += f"""fslmaths {W}/cand_plus/nodif_brain_mask -mul {W}/cand_minus/nodif_brain_mask -bin {W}/common_brain_mask
for c in plus minus; do
    echo "D14METRIC $c mad $(fslstats {W}/cand_$c/absdiff_AP_LR -k {W}/common_brain_mask -M)"
    echo "D14METRIC $c mean_AP $(fslstats {W}/cand_$c/corr_AP -k {W}/common_brain_mask -M)"
    echo "D14METRIC $c mean_LR $(fslstats {W}/cand_$c/corr_LR -k {W}/common_brain_mask -M)"
    echo "D14METRIC $c mask_voxels $(fslstats {W}/common_brain_mask -V | cut -d' ' -f1)"
done
""" + clean
        outs = [W / f"cand_{n}" / f for n in ("plus", "minus")
                for f in ("topup_Pos_Neg_b0_fieldcoef.nii.gz", "myfield.nii.gz", "nodif_brain_mask.nii.gz",
                          "corr_AP.nii.gz", "corr_LR.nii.gz")] + [W / "d14_report.md"]
        return Job("d14test", "nondenoised", row, "cpu", body, outputs=outs, blocked=blocked, extra={"W": W})

    # ---- status -----------------------------------------------------------
    def read_status(self, job: Job) -> dict:
        p = self.job_paths(job)["status"]
        return json.loads(p.read_text()) if p.is_file() else {}

    def write_status(self, job: Job, **kw) -> None:
        p = self.job_paths(job)["status"]
        p.parent.mkdir(parents=True, exist_ok=True)
        with _status_lock:
            st = self.read_status(job)
            st.update(kw, key=job.key, stage=job.stage, arm=job.arm, subject=job.row["subject"],
                      session=job.row["session"], updated=utcnow())
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=2, default=str))
            tmp.replace(p)

    def is_running(self, st: dict) -> bool:
        if st.get("state") != "running" or st.get("host") != platform.node():
            return False
        try:
            os.kill(int(st.get("pid", -1)), 0)
            return True
        except (OSError, ValueError):
            return False

    def is_done(self, job: Job) -> bool:
        st = self.read_status(job)
        return st.get("state") == "done" and all(Path(o).exists() for o in job.outputs)

    # ---- running ----------------------------------------------------------
    def verify(self, job: Job) -> list[str]:
        missing = [str(o) for o in job.outputs if not Path(o).exists()]
        errs = [f"missing output {m}" for m in missing]
        if job.stage == "dmri" and not missing:
            data = self.brc_root / job.arm / job.row["brc_subject"] / "analysis/dMRI/processed/data"
            n = nifti_shape(data / "data.nii.gz")
            expected = int(job.row["main_n_vols"]) + 1
            nb = len((data / "bvals").read_text().split())
            if (n[3] if len(n) > 3 else 1) != expected or nb != expected:
                errs.append(f"data.nii.gz has {n} / {nb} b-values; expected {expected} volumes "
                            f"(main run + 1 reverse b0)")
        return errs

    def run(self, job: Job) -> dict:
        paths = self.job_paths(job)
        paths["sh"].parent.mkdir(parents=True, exist_ok=True)
        if job.stage == "d14test" and (job.extra["W"] / "d14_report.md").exists():
            (job.extra["W"] / "d14_report.md").unlink()
        paths["sh"].write_text(self.header(job) + job.body)
        paths["sh"].chmod(0o755)
        inputs = {k: v for k, v in job.extra.items() if k in ("main_input", "revpe_b0")}
        input_sha = {v: sha256_of(Path(v)) for v in inputs.values() if Path(v).is_file()}
        t0 = time.time()
        with open(paths["log"], "w") as fh:
            proc = subprocess.Popen(["bash", str(paths["sh"])], stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            self.write_status(job, state="running", reason=None, pid=proc.pid, host=platform.node(), started=utcnow(),
                              script=str(paths["sh"]), log=str(paths["log"]), inputs=inputs, input_sha256=input_sha,
                              extra={k: v for k, v in job.extra.items() if k not in ("W",)})
            log.info("START %s (pid %d)", job.key, proc.pid)
            rc = proc.wait()
        if job.stage == "d14test" and rc == 0:
            self.write_d14_report(job, paths["log"])
        errs = [f"exit code {rc}"] if rc != 0 else self.verify(job)
        state = "done" if not errs else "failed"
        self.write_status(job, state=state, rc=rc, finished=utcnow(), runtime_s=round(time.time() - t0), errors=errs)
        log.log(logging.INFO if state == "done" else logging.ERROR, "%s %s (%.0f min) %s", state.upper(), job.key,
                (time.time() - t0) / 60, "; ".join(errs))
        return {"key": job.key, "state": state, "errors": errs}

    def write_d14_report(self, job: Job, log_path: Path) -> None:
        W, row = job.extra["W"], job.row
        metrics = {}
        for line in log_path.read_text().splitlines():
            m = re.match(r"^D14METRIC (\w+) (\w+) ([-0-9.eE+]+)", line.strip())
            if m:
                metrics.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
        for name in ("plus", "minus"):
            out = (W / f"cand_{name}" / "topup_stdout.txt").read_text(errors="replace")
            ssd = re.findall(r"(?i)ssd\s*[=:]?\s*([-0-9.]+(?:e[-+]?\d+)?)", out)
            metrics.setdefault(name, {})["final_ssd"] = float(ssd[-1]) if ssd else None
            metrics[name]["acqparams"] = (W / f"cand_{name}" / "acqparams.txt").read_text().strip()
        (W / "d14_metrics.json").write_text(json.dumps(metrics, indent=2))
        rows = []
        for name, sign in (("plus", "+1"), ("minus", "-1")):
            m = metrics.get(name, {})
            rel = (m.get("mad") / m.get("mean_AP")) if m.get("mad") is not None and m.get("mean_AP") else None
            rows.append(f"| {sign} | `{m.get('acqparams', '').replace(chr(10), ' ; ')}` | {m.get('final_ssd')} | "
                        f"{m.get('mad')} | {rel:.4f} |" if rel is not None else
                        f"| {sign} | `{m.get('acqparams', '')}` | {m.get('final_ssd')} | {m.get('mad')} | n/a |")
        (W / "d14_report.md").write_text(f"""# LR sign test: {row['subject']}/{row['session']} (non-denoised arm)

Generated {utcnow()} by 03_brc.py. Pos_Neg_b0 built by BRC's data_copy.sh + basic_preproc.sh; TOPUP with the same
command and config as BRC's run_topup.sh, once per candidate LR x sign. Row 1 (main AP run) is what BRC writes.
Each b0 corrected with applytopup --method=jac (BRC's least-squares hifib0 step cannot combine orthogonal PE
directions); brain mask = bet -f 0.20 (BRC's threshold) on each candidate's corrected AP b0.

| LR x sign | acqparams | TOPUP final SSD | mean abs(AP - LR) in common brain mask | relative to mean AP |
|---|---|---|---|---|
{chr(10).join(rows)}

Common brain mask: intersection of both candidates' `nodif_brain_mask` ({metrics.get('plus', {}).get('mask_voxels')} voxels).
Lower SSD and lower mismatch indicate the correct sign.

## Unblinded visual check (routine QC, raw b0s)
For each candidate folder `{W}/cand_plus` and `{W}/cand_minus`, compare `corr_AP` with `corr_LR` (they should
overlap closely for the correct sign), and inspect `myfield` (Hz).

Then record the result in `pipeline/session_exceptions.csv`, the LR-sign row: `action` = `lr_x_sign=+1` or `lr_x_sign=-1`,
`status` = `final`, `decided_by`, `date`, and a note with the metrics. The dmri stage for this
session runs in both arms once the LR sign is final.
""")

    # ---- orchestration ----------------------------------------------------
    def build(self, stages, arms) -> list[Job]:
        rows = list(csv.DictReader(open(self.a.deriv_dir / "inputs" / "preproc_inputs.csv", newline="")))
        if self.a.sessions:
            rows = [r for r in rows if f"{r['subject']}/{r['session']}" in set(self.a.sessions)]
        jobs = []
        if "d14test" in stages:
            jobs += [self.d14_job(r) for r in rows if r["brc_invocation"] == "substages_custom_topup"]
        if "struc" in stages:
            jobs += [self.struc_job(r) for r in rows]
        if "dmri" in stages:
            for arm in arms:
                # custom-TOPUP sessions first so they are not left to the end
                ordered = sorted(rows, key=lambda r: r["brc_invocation"] == "dMRI_preproc.sh")
                jobs += [self.dmri_job(r, arm) for r in ordered]
        return jobs

    def execute(self, jobs: list[Job]) -> int:
        todo, summary = [], {"done": 0, "blocked": 0, "running_elsewhere": 0}
        for j in jobs:
            st = self.read_status(j)
            if self.is_done(j) and not self.a.force:
                summary["done"] += 1
            elif self.is_running(st):
                summary["running_elsewhere"] += 1
                log.info("SKIP %s: already running (pid %s)", j.key, st.get("pid"))
            elif j.blocked:
                summary["blocked"] += 1
                self.write_status(j, state="blocked", reason=j.blocked)
                log.warning("BLOCKED %s: %s", j.key, j.blocked)
            else:
                todo.append(j)
        log.info("Jobs: %d total, %d already done, %d blocked, %d running elsewhere, %d to run",
                 len(jobs), summary["done"], summary["blocked"], summary["running_elsewhere"], len(todo))
        if self.a.dry_run:
            for j in todo:
                print(f"would run {j.key}")
            return 0
        results = []
        with ThreadPoolExecutor(max_workers=self.a.cpu_jobs, thread_name_prefix="cpu") as cpu, \
                ThreadPoolExecutor(max_workers=self.a.gpu_jobs, thread_name_prefix="gpu") as gpu:
            futs = [(cpu if j.kind == "cpu" else gpu).submit(self.run, j) for j in todo]
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as exc:  # noqa: BLE001 - logged, others continue
                    log.exception("Runner error: %s", exc)
                    results.append({"state": "failed"})
        failed = sum(r["state"] == "failed" for r in results)
        log.info("Finished: %d ran, %d failed", len(results), failed)
        return 1 if failed else 0


def print_status(deriv: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted((deriv / "brc" / "_jobs").rglob("*.json"))]
    counts = {}
    for r in rows:
        k = (r.get("stage"), r.get("arm"))
        counts.setdefault(k, {}).setdefault(r.get("state"), 0)
        counts[k][r.get("state")] += 1
    for (stage, arm), c in sorted(counts.items()):
        print(f"{stage:8s} {arm:12s} " + "  ".join(f"{s}={n}" for s, n in sorted(c.items())))
    for r in rows:
        if r.get("state") in ("failed", "blocked"):
            print(f"  {r['state']:8s} {r['key']}: {'; '.join(r.get('errors', [])) or r.get('reason', '')}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Run BRC struc/dMRI preprocessing and the LR sign test over all sessions.")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--bids-dir", type=Path, required=True)
    p.add_argument("--meta-dir", type=Path, required=True, help="00_download.py --out-dir (sessions_manifest.csv)")
    p.add_argument("--stages", nargs="+", choices=("struc", "d14test", "dmri"), default=["struc", "d14test", "dmri"])
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--sessions", nargs="+", help="Only these sessions (sub-XXXXX/ses-YYYYYYY)")
    p.add_argument("--cpu-jobs", type=int, default=12, help="Concurrent struc/d14test jobs")
    p.add_argument("--gpu-jobs", type=int, default=4, help="Concurrent dMRI jobs (one EDDY each)")
    p.add_argument("--brc-dir", type=Path, default=DEFAULT_BRC)
    p.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    p.add_argument("--force", action="store_true", help="Re-run jobs that are already done")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true", help="Print job status and exit")
    a = p.parse_args(argv)
    for k in ("deriv_dir", "bids_dir", "meta_dir", "brc_dir", "decisions"):
        setattr(a, k, getattr(a, k).expanduser().resolve())
    if a.deriv_dir.is_relative_to(REPO_ROOT):
        p.error("--deriv-dir is inside the code repository")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.status:
        print_status(a.deriv_dir)
        return 0
    logdir = a.deriv_dir / "brc" / "_jobs" / "_runner"
    logdir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    for h, fmt in ((logging.FileHandler(logdir / f"03_brc_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log"),
                    "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"),
                   (logging.StreamHandler(sys.stdout), "%(asctime)s %(levelname)-7s %(message)s")):
        h.setFormatter(logging.Formatter(fmt))
        log.addHandler(h)
    log.info("03_brc.py v%s (sha256 %s) pid %d", SCRIPT_VERSION, sha256_of(Path(__file__)), os.getpid())
    log.info("Command: %s", " ".join([sys.executable] + sys.argv))
    if (a.brc_dir / "version.txt").read_text().strip() != EXPECTED_BRC:
        log.error("BRC at %s is not %s", a.brc_dir, EXPECTED_BRC)
        return 2
    eddy = Path("/usr/local/fsl/6.0.6.x/bin/MIGTOOLS_eddy_cuda10.2")   # EDDY binary on the system used here; logged if present
    if eddy.is_file():
        log.info("EDDY binary: %s sha256 %s", eddy, sha256_of(eddy))
    runner = Runner(a)
    jobs = runner.build(a.stages, a.arms)
    return runner.execute(jobs)


if __name__ == "__main__":
    sys.exit(main())
