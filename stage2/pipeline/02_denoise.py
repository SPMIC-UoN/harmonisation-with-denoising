#!/usr/bin/env python3
"""
02_denoise.py - MPPCA denoising of the main dMRI run of every session with
DESIGNER 2.0.11 (denoised arm), with per-session checks and provenance.

For each row of <deriv-dir>/inputs/preproc_inputs.csv:
  singularity exec <sif> designer -denoise -extent 5,5,5 -adaptive_patch \
      -pf 1 -n_cores N -scratch <dir> -nocleanup <designer_input> <designer_output>
(-pf 1 is a placeholder DESIGNER requires but only uses for -degibbs;
-n_cores, -scratch and -nocleanup only control threads and the temporary
folder.) Then:
  - DESIGNER's log must contain only format conversions (mrconvert/mrinfo):
    no Gibbs, EDDY, Rician, normalisation or other step ran.
  - The output must keep the input's affine and dimensions (incl. volumes);
    float32 is accepted.
  - DESIGNER's exported bval/bvec must equal the raw ones; they are then
    replaced by byte-identical copies of the raw files (BRC needs integers).
  - The noise map (sigma) and the number of signal components (Npars) are
    kept next to the output (routine QC); the rest of the scratch is removed.
  - A provenance JSON records input/output SHA-256, the container SHA-256 and
    version, the command, and the check results.
Re-running skips sessions whose provenance matches; failed sessions are retried.

BLINDING: this script never reads voxel values. Outputs are checked through
NIfTI headers and text files only; images are copied as raw bytes.
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
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SIF_SHA256 = "2f1aea08af7e7b77f9a9ea1702fea16d43f42e4f001136ec15e01278bee31752"
EXPECTED_DESIGNER_VERSION = "2.0.11"
PROTOCOL_ARGS = ["-denoise", "-extent", "5,5,5", "-adaptive_patch"]          # , Table S1
# DESIGNER 2.0.11 refuses to start unless a partial-Fourier factor is given or found in the sidecar (Philips/GE
# sidecars have none), although only -degibbs uses it. A placeholder is passed for every session; on
# sub-16981 ses-OXF1PRI001 the output was byte-identical with no -pf, -pf 0.75 and -pf 1.
PF_PLACEHOLDER = ["-pf", "1"]
ALLOWED_LOG_COMMANDS = {"mrconvert", "mrinfo"}                             # format conversion only
BVAL_TOL = 1.0          # s/mm^2
BVEC_TOL = 1e-3

log = logging.getLogger("02_denoise")


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_nifti_header(path: Path) -> dict:
    """Parse ONLY the NIfTI header: exactly sizeof_hdr bytes are read, no voxel data."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:
        first = fh.read(4)
        size = int.from_bytes(first, "little")
        if size not in (348, 540):
            size = int.from_bytes(first, "big")
        if size not in (348, 540):
            raise ValueError(f"{path.name}: not a NIfTI-1/2 header")
        raw = first + fh.read(size - 4)
    klass = nib.Nifti1Header if size == 348 else nib.Nifti2Header
    hdr = klass.from_fileobj(io.BytesIO(raw))
    return {"shape": tuple(int(x) for x in hdr.get_data_shape()), "affine": hdr.get_best_affine(),
            "dtype": str(hdr.get_data_dtype()), "zooms": tuple(float(z) for z in hdr.get_zooms())}


def read_bvals(path: Path) -> np.ndarray:
    return np.array([float(x) for x in path.read_text().split()])


def read_bvecs(path: Path) -> np.ndarray:
    rows = [[float(x) for x in ln.split()] for ln in path.read_text().splitlines() if ln.strip()]
    return np.array(rows)


def gzip_copy(src: Path, dst: Path) -> None:
    """Compress raw file bytes (no decoding of voxel values)."""
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 22)
    tmp.replace(dst)


def stem(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return str(path)[: -len(ext)]
    return str(path)


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------
def container_info(sif: Path, cache: Path) -> dict:
    """SHA-256 (cached by size+mtime) and designer2 version of the container."""
    st = sif.stat()
    key = {"path": str(sif), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("key") == key:
            return cached["info"]
    log.info("Hashing container %s (%.1f GB)", sif, st.st_size / 1e9)
    info = {"path": str(sif), "sha256": sha256_of(sif), "size": st.st_size}
    res = subprocess.run(["singularity", "exec", str(sif), "pip", "show", "designer2"],
                         capture_output=True, text=True)
    m = re.search(r"^Version:\s*(\S+)", res.stdout, re.M)
    info["designer2_version"] = m.group(1) if m else "unknown"
    res = subprocess.run(["singularity", "inspect", str(sif)], capture_output=True, text=True)
    info["labels"] = dict(ln.split(": ", 1) for ln in res.stdout.splitlines() if ": " in ln)
    res = subprocess.run(["singularity", "--version"], capture_output=True, text=True)
    info["singularity"] = res.stdout.strip()
    cache.write_text(json.dumps({"key": key, "info": info}, indent=2))
    return info


# --------------------------------------------------------------------------
# One session
# --------------------------------------------------------------------------
def outputs_for(row: dict) -> dict:
    out = Path(row["designer_output"])
    s = stem(out)
    base = s.rsplit("_desc-", 1)[0]
    return {"nii": out, "bval": Path(s + ".bval"), "bvec": Path(s + ".bvec"),
            "sigma": Path(base + "_desc-mppca_noisemap.nii.gz"),
            "npars": Path(base + "_desc-mppca_npars.nii.gz"),
            "prov": Path(s + "_provenance.json"), "logdir": out.parent / "designer_logs",
            "status": out.parent / "status.json"}


def up_to_date(row: dict, o: dict, sif_sha: str, in_sha: str) -> bool:
    if not all(o[k].is_file() for k in ("nii", "bval", "bvec", "sigma", "prov")):
        return False
    prov = json.loads(o["prov"].read_text())
    return (prov.get("status") == "ok" and prov.get("input", {}).get("sha256") == in_sha
            and prov.get("container", {}).get("sha256") == sif_sha
            and prov.get("designer_args") == PROTOCOL_ARGS)


def check_log(scratch: Path) -> tuple[list[str], list[str]]:
    """Commands DESIGNER ran (log.txt) and any that are not pure format conversions."""
    lines = [ln.strip() for ln in (scratch / "log.txt").read_text().splitlines() if ln.strip()]
    tools = [ln.split()[0] for ln in lines]
    bad = [ln for ln, t in zip(lines, tools) if Path(t).name not in ALLOWED_LOG_COMMANDS]
    return lines, bad


def denoise_session(row: dict, args, info: dict) -> dict:
    key = f"{row['subject']}/{row['session']}"
    src = Path(row["designer_input"])
    o = outputs_for(row)
    o["nii"].parent.mkdir(parents=True, exist_ok=True)
    in_sha = sha256_of(src)
    if not args.force and up_to_date(row, o, info["sha256"], in_sha):
        return {"session": key, "status": "skipped (up to date)"}

    status = {"session": key, "status": "running", "started": utcnow(), "host": platform.node()}
    o["status"].write_text(json.dumps(status, indent=2))
    scratch_root = args.scratch_root / f"{row['subject']}_{row['session']}"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True)
    for p in (o["nii"], o["bval"], o["bvec"]):
        if p.exists():
            p.unlink()

    binds = [f"{Path(row['main_dwi_nii']).parents[3]}:{Path(row['main_dwi_nii']).parents[3]}:ro",
             f"{args.deriv_dir}:{args.deriv_dir}"]
    if not scratch_root.is_relative_to(args.deriv_dir):
        binds.append(f"{args.scratch_root}:{args.scratch_root}")
    cmd = (["singularity", "exec", "--cleanenv"] + [x for b in binds for x in ("-B", b)]
           + [str(args.sif), "designer"] + PROTOCOL_ARGS + PF_PLACEHOLDER
           + ["-n_cores", str(args.cores), "-scratch", str(scratch_root) + "/", "-nocleanup",
              str(src), str(o["nii"])])
    session_log = o["logdir"] / "02_denoise_stdout.txt"
    o["logdir"].mkdir(exist_ok=True)
    log.info("%s: RUN %s", key, " ".join(cmd))
    t0 = time.time()
    with open(session_log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    runtime = time.time() - t0
    errors, checks = [], {}
    if rc != 0:
        errors.append(f"designer exited with {rc} (see {session_log})")
    scratches = sorted(scratch_root.glob("designer-tmp-*"))
    scratch = scratches[-1] if scratches else None
    if scratch is None:
        errors.append("DESIGNER scratch folder not found")

    if not errors:
        # 1. Only format conversions ran; command as specified.
        lines, bad = check_log(scratch)
        checks["log_commands"] = lines
        if bad:
            errors.append(f"DESIGNER ran non-conversion commands: {bad}")
        command = (scratch / "command.txt").read_text().strip()
        checks["designer_command"] = command
        for a in PROTOCOL_ARGS:
            if a not in command.split():
                errors.append(f"designer command lacks {a}: {command}")
        forbidden = [a for a in ("-degibbs", "-rician", "-eddy", "-normalize", "-b1correct", "-mask",
                                 "-pe_dir", "-phase", "-patch2self", "-shrinkage", "-algorithm") if a in command.split()]
        if forbidden:
            errors.append(f"designer command contains {forbidden}")
        for f in ("log.txt", "command.txt", "cwd.txt"):
            if (scratch / f).is_file():
                shutil.copy2(scratch / f, o["logdir"] / f"designer_{f}")

        # 2. Geometry (headers only).
        hi, ho = read_nifti_header(src), read_nifti_header(o["nii"])
        checks["input_header"] = {"shape": hi["shape"], "dtype": hi["dtype"]}
        checks["output_header"] = {"shape": ho["shape"], "dtype": ho["dtype"]}
        checks["affine_equal"] = bool(np.allclose(hi["affine"], ho["affine"], atol=1e-4))
        checks["shape_equal"] = hi["shape"] == ho["shape"]
        if not checks["affine_equal"]:
            errors.append(f"output affine differs from input:\n{hi['affine']}\n{ho['affine']}")
        if not checks["shape_equal"]:
            errors.append(f"output shape {ho['shape']} != input {hi['shape']}")

        # 3. Gradients: DESIGNER's export must equal the raw files; then use raw copies.
        rb, rv = read_bvals(Path(row["main_dwi_bval"])), read_bvecs(Path(row["main_dwi_bvec"]))
        if o["bval"].is_file() and o["bvec"].is_file():
            db, dv = read_bvals(o["bval"]), read_bvecs(o["bvec"])
            checks["bval_max_abs_diff"] = float(np.max(np.abs(db - rb))) if db.shape == rb.shape else None
            checks["bvec_max_abs_diff"] = float(np.max(np.abs(dv - rv))) if dv.shape == rv.shape else None
            if checks["bval_max_abs_diff"] is None or checks["bval_max_abs_diff"] > BVAL_TOL:
                errors.append(f"DESIGNER bvals differ from raw (max diff {checks['bval_max_abs_diff']})")
            if checks["bvec_max_abs_diff"] is None or checks["bvec_max_abs_diff"] > BVEC_TOL:
                errors.append(f"DESIGNER bvecs differ from raw (max diff {checks['bvec_max_abs_diff']})")
        else:
            errors.append("DESIGNER did not write bval/bvec next to the output")
        if not errors:
            shutil.copyfile(row["main_dwi_bval"], o["bval"])
            shutil.copyfile(row["main_dwi_bvec"], o["bvec"])
            checks["gradients"] = "raw bval/bvec copied byte-for-byte after numeric check"

        # 4. Keep noise map and Npars (routine QC), check their grid, drop the scratch.
        for name, dst in (("sigma.nii", o["sigma"]), ("Npars.nii", o["npars"])):
            if (scratch / name).is_file():
                gzip_copy(scratch / name, dst)
                h = read_nifti_header(dst)
                same = bool(np.allclose(h["affine"], hi["affine"], atol=1e-4)) and h["shape"][:3] == hi["shape"][:3]
                checks[f"{name}_on_input_grid"] = same
                if not same:
                    errors.append(f"{name} is not on the input grid (shape {h['shape']})")
            elif name == "sigma.nii":
                errors.append("DESIGNER noise map sigma.nii not found in scratch")
    if scratch_root.exists() and (not errors or not args.keep_failed_scratch):
        shutil.rmtree(scratch_root)

    prov = {
        "status": "ok" if not errors else "failed", "errors": errors,
        "input": {"path": str(src), "sha256": in_sha},
        "output": {"path": str(o["nii"]), "sha256": sha256_of(o["nii"]) if o["nii"].is_file() else None},
        "noisemap": str(o["sigma"]), "npars": str(o["npars"]),
        "designer_args": PROTOCOL_ARGS, "technical_args": {"pf_placeholder": " ".join(PF_PLACEHOLDER), "n_cores": args.cores, "scratch": "per session",
                                                           "nocleanup": "scratch kept only to retrieve sigma/Npars/logs"},
        "command": cmd, "container": info, "checks": checks, "runtime_s": round(runtime, 1),
        "started": status["started"], "finished": utcnow(), "host": platform.node(),
        "generated_by": {"name": "02_denoise.py", "version": SCRIPT_VERSION},
    }
    o["prov"].write_text(json.dumps(prov, indent=2, default=str))
    status.update(status=prov["status"], finished=prov["finished"], errors=errors, runtime_s=prov["runtime_s"])
    o["status"].write_text(json.dumps(status, indent=2))
    return {"session": key, "status": prov["status"], "errors": errors, "runtime_s": prov["runtime_s"]}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="MPPCA denoising (DESIGNER 2.0.11) of every session's main dMRI run.")
    p.add_argument("--deriv-dir", type=Path, required=True, help="Derivatives root (contains inputs/preproc_inputs.csv)")
    p.add_argument("--sif", type=Path, required=True, help="DESIGNER 2.0.11 Singularity image")
    p.add_argument("--sessions", nargs="+", help="Only these sessions, as sub-XXXXX/ses-YYYYYYY")
    p.add_argument("--jobs", type=int, default=4, help="Sessions denoised in parallel")
    p.add_argument("--cores", type=int, default=16, help="DESIGNER -n_cores per session")
    p.add_argument("--scratch-root", type=Path, help="DESIGNER scratch (default: <deriv-dir>/designer/_scratch)")
    p.add_argument("--force", action="store_true", help="Re-run sessions even if up to date")
    p.add_argument("--keep-failed-scratch", action="store_true", help="Keep scratch of failed sessions")
    p.add_argument("--allow-other-sif", action="store_true", help="Do not require the pinned container hash")
    args = p.parse_args(argv)
    args.deriv_dir = args.deriv_dir.expanduser().resolve()
    args.scratch_root = (args.scratch_root or args.deriv_dir / "designer" / "_scratch").expanduser().resolve()
    if args.deriv_dir.is_relative_to(REPO_ROOT):
        p.error(f"--deriv-dir is inside the code repository {REPO_ROOT}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    ddir = args.deriv_dir / "designer"
    (ddir / "_logs").mkdir(parents=True, exist_ok=True)
    log_path = ddir / "_logs" / f"02_denoise_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    log.setLevel(logging.DEBUG)
    for h, lvl, fmt in ((logging.FileHandler(log_path), logging.DEBUG, "%(asctime)s %(levelname)-7s %(message)s"),
                        (logging.StreamHandler(sys.stdout), logging.INFO, "%(levelname)-7s %(message)s")):
        h.setLevel(lvl)
        h.setFormatter(logging.Formatter(fmt))
        log.addHandler(h)
    script = Path(__file__).resolve()
    log.info("02_denoise.py v%s (sha256 %s)", SCRIPT_VERSION, sha256_of(script))
    log.info("Command: %s", " ".join([sys.executable] + sys.argv))

    rows = list(csv.DictReader(open(args.deriv_dir / "inputs" / "preproc_inputs.csv", newline="")))
    if args.sessions:
        wanted = set(args.sessions)
        rows = [r for r in rows if f"{r['subject']}/{r['session']}" in wanted]
        missing = wanted - {f"{r['subject']}/{r['session']}" for r in rows}
        if missing:
            log.error("Sessions not in preproc_inputs.csv: %s", sorted(missing))
            return 2
    info = container_info(args.sif, ddir / "_logs" / "container_sha256.json")
    log.info("Container %s: sha256 %s, designer2 %s", info["path"], info["sha256"], info["designer2_version"])
    if not args.allow_other_sif and (info["sha256"] != EXPECTED_SIF_SHA256
                                     or info["designer2_version"] != EXPECTED_DESIGNER_VERSION):
        log.error("Container is not the pinned DESIGNER %s (sha256 %s)", EXPECTED_DESIGNER_VERSION, EXPECTED_SIF_SHA256)
        return 2

    log.info("Denoising %d sessions, %d in parallel, %d cores each", len(rows), args.jobs, args.cores)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(denoise_session, r, args, info): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 - reported per session
                log.exception("%s/%s crashed", r["subject"], r["session"])
                res = {"session": f"{r['subject']}/{r['session']}", "status": "failed", "errors": [str(exc)]}
            results.append(res)
            lvl = logging.INFO if res["status"] != "failed" else logging.ERROR
            log.log(lvl, "[%d/%d] %s: %s %s", i, len(rows), res["session"], res["status"],
                    "; ".join(res.get("errors", [])))
    failed = [r for r in results if r["status"] == "failed"]
    summary = ddir / "_logs" / "02_denoise_summary.csv"
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["session", "status", "runtime_s", "errors"], extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda r: r["session"]):
            w.writerow({**r, "errors": "; ".join(r.get("errors", []))})
    log.info("Done: %d ok/skipped, %d failed. Summary: %s", len(results) - len(failed), len(failed), summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
