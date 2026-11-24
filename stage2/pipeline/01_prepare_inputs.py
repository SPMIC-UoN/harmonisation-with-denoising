#!/usr/bin/env python3
"""
01_prepare_inputs.py - Turn the session manifest (00_download.py) and the
session exceptions (session_exceptions.csv) into the per-session inputs for DESIGNER and
the BRC pipelines, and stage the reverse-PE b0 images.

Outputs (under --deriv-dir/inputs):
  preproc_inputs.csv          one row per session: every argument the
                              denoising and BRC steps need, the readout time
                              BRC will use, the main-run PE direction, and the
                              volumes to drop from BRC's output.
  ge_readout_for_authors.csv  GE echo spacing / readout metadata.
  01_report.md                checks and summary.
  sub-*/ses-*/                staged reverse-PE b0 (+ bval/bvec/provenance),
                              and for custom-TOPUP sessions the resampled
                              reverse b0 and the custom acqparams.

Modes:
  --plan-only  metadata only: sidecars, bval/bvec and NIfTI headers.
  default      also stages images with FSL 6.0.6.4 command-line tools
               (fslroi, flirt -applyxfm -usesqform).

BLINDING: this script never reads voxel values. NIfTI files are opened only to
parse the fixed-size header; images are copied/resampled by FSL tools and the
results are checked through their headers only.


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
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS = Path(__file__).resolve().parent / "session_exceptions.csv"
EXPECTED_FSL = "6.0.6.4"
EXPECTED_BRC = "Ver1.6.3"
B0_MAX_BVAL = 50               # BRC b0maxbval: volumes with b < 50 are b0s
BRC_PE_DIR = 2                 # AP/PA
ARMS = ("nondenoised", "denoised")
DESIGNER_ARGS = "-denoise -extent 5,5,5 -adaptive_patch"
REVPE_VOLUME = 0               # volume 0 of the reverse run
# Session-level actions this script knows how to apply (session_exceptions.csv); the confirmed left-right sign
# (lr_x_sign=+1 / -1, recorded after the sign test of 03_brc.py) is read by 03_brc.py.
KNOWN_SESSION_ACTIONS = {"custom_topup_lr", "use_acquired_revpe", "include_as_is", "verify_lr_sign"}
GE_AUTHOR_ROWS = {"Premier48_SPMICQMC": "GE Premier 48ch", "Premier21_OCMR": "GE Premier 21ch"}

log = logging.getLogger("01_prepare_inputs")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def write_csv(path: Path, columns, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    tmp.replace(path)


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
    slope, inter = hdr.get_slope_inter()
    return {"shape": tuple(int(x) for x in hdr.get_data_shape()),
            "zooms": tuple(float(x) for x in hdr.get_zooms()),
            "affine": hdr.get_best_affine(),
            "dtype": str(hdr.get_data_dtype()),
            "scl": (slope, inter)}


def read_bvals(path: Path) -> list[float]:
    return [float(x) for x in path.read_text().split()]


def read_bvecs(path: Path) -> list[list[str]]:
    return [ln.split() for ln in path.read_text().splitlines() if ln.strip()]


def brc_readout(echospacing: str, n_pe: int) -> Decimal:
    """ro_time exactly as BRC basic_preproc.sh computes it with --p_im 1:
    bc 'scale=6; es / 1 * (n_pe - 1)', i.e. es truncated to 6 decimals."""
    es = Decimal(echospacing).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    return es * (n_pe - 1)


def decimal_str(x: float) -> str:
    """Plain decimal notation (bc cannot parse '3.3e-04')."""
    return format(Decimal(repr(float(x))), "f")


def fsl_pe_vector(ped: str, affine) -> tuple[int, int, int]:
    """BIDS PhaseEncodingDirection -> FSL acqparams vector. FSL works in radiological
    voxel coordinates, so voxel x is flipped for neurological (det > 0) images."""
    axis = "ijk".index(ped[0])
    sign = -1 if ped.endswith("-") else 1
    if axis == 0 and np.linalg.det(np.asarray(affine)[:3, :3]) > 0:
        sign = -sign
    vec = [0, 0, 0]
    vec[axis] = sign
    return tuple(vec)


def ped_on_grid(ped: str, src_affine, dst_affine) -> str:
    """Express a PE direction given on one voxel grid on another grid (via world space).
    The resampled reverse b0 of a custom-TOPUP session lives on the main-run grid."""
    axis = "ijk".index(ped[0])
    world = np.asarray(src_affine)[:3, :3][:, axis] * (-1 if ped.endswith("-") else 1)
    vox = np.linalg.solve(np.asarray(dst_affine)[:3, :3], world)
    k = int(np.argmax(np.abs(vox)))
    return "ijk"[k] + ("-" if vox[k] < 0 else "")


def fmt_vec(vec) -> str:
    return " ".join(str(int(v)) for v in vec)


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------
def load_decisions(path: Path) -> dict:
    rows = read_csv(path)
    required = {"decision_id", "scope", "subject", "session", "scanner_label", "action", "status"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    by_session, by_scanner, pending = defaultdict(list), defaultdict(list), []
    for r in rows:
        if r["status"] == "pending":
            pending.append(r)
        if r["scope"] == "session":
            if not r["subject"] or not r["session"]:
                raise SystemExit(f"{path}: session-scoped {r['decision_id']} lacks subject/session")
            for action in r["action"].split(";"):
                if action and action not in KNOWN_SESSION_ACTIONS and not action.startswith("lr_x_sign="):
                    raise SystemExit(f"{path}: {r['decision_id']} has unknown session action {action!r}; "
                                     f"this script only implements {sorted(KNOWN_SESSION_ACTIONS)}")
            by_session[(r["subject"], r["session"])].append(r)
        elif r["scope"] == "scanner":
            by_scanner[r["scanner_label"]].append(r)
    return {"rows": rows, "by_session": by_session, "by_scanner": by_scanner, "pending": pending,
            "sha256": sha256_of(path), "path": str(path)}


# --------------------------------------------------------------------------
# Planning (metadata only)
# --------------------------------------------------------------------------
PREP_COLUMNS = (
    "subject", "session", "scanner_label", "brc_subject",
    "main_dwi_nii", "main_dwi_bval", "main_dwi_bvec", "main_dwi_json", "main_n_vols",
    "main_pe_direction", "main_pe_fsl", "field_sign_vs_true",
    "pe_matrix", "echospacing_arg", "readout_time_used_s", "readout_time_sidecar_s",
    "revpe_source_nii", "revpe_source_json", "revpe_source_volume", "revpe_source_bval",
    "revpe_pe_direction", "revpe_readout_time_used_s",
    "revpe_b0_nii", "revpe_b0_bval", "revpe_b0_bvec",
    "designer_input", "designer_output", "designer_args",
    "brc_pe_dir", "brc_invocation", "custom_topup_acqparams",
    "brc_data_main_vols", "brc_data_drop_vols",
    "decisions", "flags", "notes",
)


def derivative_name(bids_name: str, desc: str) -> str:
    """sub-X_ses-Y_dir-PA_dwi.nii.gz -> sub-X_ses-Y_dir-PA_desc-<desc>_dwi.nii.gz"""
    stem = bids_name.replace(".nii.gz", "").replace(".nii", "")
    base, suffix = stem.rsplit("_", 1)
    return f"{base}_desc-{desc}_{suffix}"


def plan_session(row: dict, bids: Path, deriv: Path, decisions: dict, errors: list) -> dict:
    sub, ses, label = row["subject"], row["session"], row["scanner_label"]
    key = f"{sub}/{ses}"
    notes, applied = [], set()   # session_exceptions.csv rows that apply to this session
    actions = set()
    for d in decisions["by_session"].get((sub, ses), []):
        actions.update(a for a in d["action"].split(";") if a)
        applied.add(d["decision_id"])
    for d in decisions["by_scanner"].get(label, []):
        applied.add(d["decision_id"])
    custom = "custom_topup_lr" in actions

    def need(col):
        if not row.get(col):
            errors.append(f"{key}: manifest column {col} is empty")
            return None
        path = bids / row[col]
        if not path.is_file():
            errors.append(f"{key}: missing file {path}")
            return None
        return path

    m_nii, m_bval, m_bvec, m_json = (need(c) for c in ("dwi_nii", "dwi_bval", "dwi_bvec", "dwi_json"))
    r_nii, r_bval, r_bvec, r_json = (need(c) for c in ("revpe_nii", "revpe_bval", "revpe_bvec", "revpe_json"))
    if None in (m_nii, m_bval, m_bvec, m_json, r_nii, r_bval, r_bvec, r_json):
        return {}

    m_meta, r_meta = read_json(m_json), read_json(r_json)
    m_hdr, r_hdr = read_nifti_header(m_nii), read_nifti_header(r_nii)
    m_b, r_b = read_bvals(m_bval), read_bvals(r_bval)
    n_main = m_hdr["shape"][3] if len(m_hdr["shape"]) > 3 else 1
    if len(m_b) != n_main:
        errors.append(f"{key}: main run has {n_main} volumes but {len(m_b)} b-values")
    if any(b != int(b) for b in m_b + r_b):
        errors.append(f"{key}: non-integer b-values (BRC's bash tests need integers)")
    if m_b and m_b[0] >= B0_MAX_BVAL:
        errors.append(f"{key}: main-run volume 0 is not a b0 (b={m_b[0]}); BRC would pick a different b0")
    if len(r_b) <= REVPE_VOLUME or r_b[REVPE_VOLUME] >= B0_MAX_BVAL:
        errors.append(f"{key}: reverse-run volume {REVPE_VOLUME} is not a b0")

    m_ped, r_ped = m_meta.get("PhaseEncodingDirection", ""), r_meta.get("PhaseEncodingDirection", "")
    if m_ped not in ("j", "j-"):
        errors.append(f"{key}: main-run PED {m_ped!r} is not along j (BRC --pe_dir 2 expects AP/PA)")
        return {}
    if not r_ped:
        errors.append(f"{key}: reverse-run PhaseEncodingDirection missing")
        return {}
    if custom:
        if r_ped[0] == m_ped[0]:
            errors.append(f"{key}: custom_topup_lr decided, but reverse PED {r_ped} is on the main axis")
    elif r_ped[0] != m_ped[0] or r_ped.endswith("-") == m_ped.endswith("-"):
        errors.append(f"{key}: reverse PED {r_ped} is not the reverse of main PED {m_ped} and no decision "
                      "covers it")

    es = m_meta.get("EffectiveEchoSpacing")
    trt = m_meta.get("TotalReadoutTime")
    if not isinstance(es, (int, float)) or not isinstance(trt, (int, float)):
        errors.append(f"{key}: EffectiveEchoSpacing/TotalReadoutTime missing from main sidecar")
        return {}
    es_arg = decimal_str(es)
    n_pe = m_hdr["shape"][1]
    ro_used = brc_readout(es_arg, n_pe)
    if abs(float(ro_used) - trt) > 0.005 * trt:
        errors.append(f"{key}: BRC readout {ro_used} differs from sidecar TotalReadoutTime {trt} by >0.5%")

    main_fsl = fsl_pe_vector(m_ped, m_hdr["affine"])
    frame = 1 if main_fsl == (0, 1, 0) else -1          # BRC writes 0 1 0 for the main run
    r_es = r_meta.get("EffectiveEchoSpacing")
    r_axis = "ijk".index(r_ped[0])
    r_ro = brc_readout(decimal_str(r_es), r_hdr["shape"][r_axis]) if isinstance(r_es, (int, float)) else None
    if r_ro is None:
        errors.append(f"{key}: reverse-run EffectiveEchoSpacing missing")

    in_dir = deriv / "inputs" / sub / ses
    rev_b0 = in_dir / (derivative_name(r_nii.name, "revpeb0") + ".nii.gz")
    brc_rev = rev_b0
    acq_path = ""
    if custom:
        brc_rev = in_dir / (derivative_name(r_nii.name, "revpeb0resampled") + ".nii.gz")
        acq_path = in_dir / f"{sub}_{ses}_desc-customtopup_acqparams.txt"
        on_main = ped_on_grid(r_ped, r_hdr["affine"], m_hdr["affine"])
        notes.append(f"custom TOPUP: reverse PED {r_ped} ({on_main} on the main grid); acqparams row = "
                     f"BRC frame {frame:+d} x FSL {fmt_vec(fsl_pe_vector(on_main, m_hdr['affine']))}; "
                     "sign to verify")
    if "use_acquired_revpe" in actions:
        notes.append("acquired reverse b0 used despite dataset README")
    if "include_as_is" in actions:
        notes.append("included as-is")

    designer_out = deriv / "designer" / sub / ses / "dwi" / (derivative_name(m_nii.name, "mppca") + ".nii.gz")
    stem_b0 = str(brc_rev)[:-len(".nii.gz")]
    return {
        "subject": sub, "session": ses, "scanner_label": label, "brc_subject": f"{sub}_{ses}",
        "main_dwi_nii": str(m_nii), "main_dwi_bval": str(m_bval), "main_dwi_bvec": str(m_bvec),
        "main_dwi_json": str(m_json), "main_n_vols": n_main,
        "main_pe_direction": m_ped, "main_pe_fsl": fmt_vec(main_fsl), "field_sign_vs_true": f"{frame:+d}",
        "pe_matrix": n_pe, "echospacing_arg": es_arg, "readout_time_used_s": str(ro_used),
        "readout_time_sidecar_s": repr(float(trt)),
        "revpe_source_nii": str(r_nii), "revpe_source_json": str(r_json), "revpe_source_volume": REVPE_VOLUME,
        "revpe_source_bval": int(r_b[REVPE_VOLUME]) if r_b else "",
        "revpe_pe_direction": r_ped, "revpe_readout_time_used_s": str(r_ro) if r_ro is not None else "",
        "revpe_b0_nii": str(brc_rev), "revpe_b0_bval": stem_b0 + ".bval", "revpe_b0_bvec": stem_b0 + ".bvec",
        "designer_input": str(m_nii), "designer_output": str(designer_out), "designer_args": DESIGNER_ARGS,
        "brc_pe_dir": BRC_PE_DIR, "brc_invocation": "substages_custom_topup" if custom else "dMRI_preproc.sh",
        "custom_topup_acqparams": str(acq_path),
        "brc_data_main_vols": f"0-{n_main - 1}", "brc_data_drop_vols": str(n_main),
        "decisions": ";".join(sorted(applied)), "flags": row.get("flags", ""), "notes": "; ".join(notes),
        # private fields used for staging and reporting
        "_rev_b0": rev_b0, "_custom": custom, "_m_hdr": m_hdr, "_r_hdr": r_hdr, "_r_bvec": r_bvec,
        "_frame": frame, "_r_ped": r_ped, "_m_meta": m_meta, "_r_meta": r_meta, "_ro": ro_used, "_r_ro": r_ro,
    }


# --------------------------------------------------------------------------
# Staging (FSL command-line tools only)
# --------------------------------------------------------------------------
class Fsl:
    def __init__(self, fsldir: Path):
        self.fsldir = fsldir
        self.env = dict(os.environ, FSLDIR=str(fsldir), FSLOUTPUTTYPE="NIFTI_GZ")
        self.env["PATH"] = f"{fsldir}/share/fsl/bin:{fsldir}/bin:" + self.env.get("PATH", "")
        self.version = (fsldir / "etc" / "fslversion").read_text().split(":")[0].strip()

    def run(self, tool: str, *args) -> str:
        cmd = [str(self.fsldir / "bin" / tool)] + [str(a) for a in args]
        log.info("RUN %s", " ".join(cmd))
        res = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
        if res.stdout.strip():
            log.debug("stdout: %s", res.stdout.strip())
        if res.returncode != 0:
            raise RuntimeError(f"{tool} failed ({res.returncode}): {res.stderr.strip()}")
        return " ".join(cmd)


def provenance_ok(out: Path, sources: dict, command_key: str) -> bool:
    prov = out.with_name(out.name.replace(".nii.gz", ".json"))
    if not (out.is_file() and prov.is_file()):
        return False
    meta = read_json(prov)
    return meta.get("SourcesSHA256") == sources and meta.get("CommandKey") == command_key


def write_provenance(out: Path, sources: dict, command_key: str, commands: list, fsl: Fsl, description: str):
    prov = out.with_name(out.name.replace(".nii.gz", ".json"))
    prov.write_text(json.dumps({
        "Description": description, "Sources": list(sources), "SourcesSHA256": sources,
        "CommandKey": command_key, "Commands": commands, "FSLVersion": fsl.version,
        "GeneratedBy": {"Name": "01_prepare_inputs.py", "Version": SCRIPT_VERSION}, "Created": utcnow(),
    }, indent=2))


def lossless_dtype(src: str, dst: str) -> bool:
    """FSL 6 writes uint16 input as int32; widening integer types (or ints to float32/64) is lossless."""
    s, d = np.dtype(src), np.dtype(dst)
    if s == d:
        return True
    if s.kind in "ui" and d.kind == "i":
        return d.itemsize > s.itemsize
    if s.kind in "ui" and d.kind == "f":
        return s.itemsize <= 2 and d.itemsize >= 4
    return False


def same_geometry(a: dict, b: dict, spatial_only: bool = True) -> bool:
    n = 3 if spatial_only else 4
    return a["shape"][:n] == b["shape"][:n] and np.allclose(a["affine"], b["affine"], atol=1e-4)


def stage_session(p: dict, fsl: Fsl, errors: list) -> list[str]:
    key = f"{p['subject']}/{p['session']}"
    done = []
    src = Path(p["revpe_source_nii"])
    rev_b0 = p["_rev_b0"]
    rev_b0.parent.mkdir(parents=True, exist_ok=True)
    sources = {str(src): sha256_of(src)}
    ckey = f"fslroi vol {REVPE_VOLUME}"
    if provenance_ok(rev_b0, sources, ckey):
        done.append("revpe b0 up to date")
    else:
        cmd = fsl.run("fslroi", src, rev_b0, REVPE_VOLUME, 1)
        write_provenance(rev_b0, sources, ckey, [cmd], fsl,
                         f"Volume {REVPE_VOLUME} (b0) of the reverse-PE run, for TOPUP only. Not denoised.")
        done.append("revpe b0 extracted")
    out_hdr = read_nifti_header(rev_b0)
    r_hdr = p["_r_hdr"]
    if not (same_geometry(out_hdr, r_hdr) and (len(out_hdr["shape"]) == 3 or out_hdr["shape"][3] == 1)):
        errors.append(f"{key}: staged reverse b0 differs from its source in shape or affine")
    if not lossless_dtype(r_hdr["dtype"], out_hdr["dtype"]) or out_hdr["scl"] != r_hdr["scl"] and not \
            all(np.isnan(v) or v in (0, 1) for v in out_hdr["scl"] + r_hdr["scl"] if v is not None):
        errors.append(f"{key}: staged reverse b0 datatype/scaling changed lossily "
                      f"({r_hdr['dtype']} -> {out_hdr['dtype']}, scl {r_hdr['scl']} -> {out_hdr['scl']})")
    elif out_hdr["dtype"] != r_hdr["dtype"]:
        done.append(f"datatype {r_hdr['dtype']} -> {out_hdr['dtype']} (lossless, FSL)")
    write_gradients(rev_b0, p)

    if p["_custom"]:
        resampled = Path(p["revpe_b0_nii"])
        main = Path(p["main_dwi_nii"])
        sources_c = {str(src): sources[str(src)], str(main): sha256_of(main)}
        ckey_c = "fslroi main vol0 as ref; flirt -applyxfm -usesqform -interp spline"
        if provenance_ok(resampled, sources_c, ckey_c):
            done.append("resampled reverse b0 up to date")
        else:
            with tempfile.TemporaryDirectory(dir=resampled.parent) as tmp:
                ref = Path(tmp) / "ref_grid.nii.gz"
                cmds = [fsl.run("fslroi", main, ref, 0, 1),
                        fsl.run("flirt", "-in", rev_b0, "-ref", ref, "-applyxfm", "-usesqform",
                                "-interp", "spline", "-out", resampled)]
            write_provenance(resampled, sources_c, ckey_c, cmds, fsl,
                             "Reverse-PE b0 (LR) resampled onto the main-run grid via header affines only.")
            done.append("reverse b0 resampled onto main grid")
        res_hdr = read_nifti_header(resampled)
        if not same_geometry(res_hdr, p["_m_hdr"]):
            errors.append(f"{key}: resampled reverse b0 is not on the main-run grid")
        write_gradients(resampled, p)
        acq = Path(p["custom_topup_acqparams"])
        acq.write_text(custom_acqparams(p))
        done.append(f"custom acqparams written ({acq.name})")
    return done


def write_gradients(nii: Path, p: dict) -> None:
    """bval/bvec for the single staged b0: the source run's values for that volume."""
    stem = str(nii)[:-len(".nii.gz")]
    Path(stem + ".bval").write_text(f"{p['revpe_source_bval']}\n")
    rows = read_bvecs(p["_r_bvec"])
    Path(stem + ".bvec").write_text("".join(f"{r[REVPE_VOLUME]}\n" for r in rows))


def custom_acqparams(p: dict) -> str:
    """Row 1: exactly what BRC writes for the main run. Row 2: reverse b0 in FSL voxel
    coordinates, multiplied by the BRC frame sign so both rows are consistent."""
    ped_main_grid = ped_on_grid(p["_r_ped"], p["_r_hdr"]["affine"], p["_m_hdr"]["affine"])
    rev = tuple(p["_frame"] * v for v in fsl_pe_vector(ped_main_grid, p["_m_hdr"]["affine"]))
    return f"0 1 0 {p['_ro']}\n{fmt_vec(rev)} {p['_r_ro']}\n"


# --------------------------------------------------------------------------
# GE readout table for the dataset authors
# --------------------------------------------------------------------------
GE_COLUMNS = ("subject", "session", "scanner", "site", "SeriesDescription", "ProtocolName",
              "EffectiveEchoSpacing_ms", "TotalReadoutTime_ms", "authors_table_EES_ms", "authors_table_TRT_ms",
              "EES_diff_pct", "TRT_diff_pct", "reverse_EffectiveEchoSpacing_ms", "reverse_TotalReadoutTime_ms",
              "PixelBandwidth_Hz_per_px", "AcquisitionMatrixPE", "ReconMatrixPE",
              "ParallelReductionFactorInPlane", "MultibandAccelerationFactor", "gradient_mode",
              "PhaseEncodingPolarityGE", "SliceThickness_mm", "EchoTime_ms", "ScanOptions",
              "SoftwareVersions", "ConversionSoftwareVersion")


def authors_table(bids: Path) -> dict:
    path = bids / "protocol_accelerations.csv"
    if not path.is_file():
        return {}
    out, header = {}, None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                break                                   # dMRI block ends at the first blank row
            if row[0].strip() == "dMRI":
                header = [h.strip() for h in row]
                continue
            if header:
                out[row[0].strip()] = dict(zip(header, (c.strip() for c in row)))
    return out


def ge_rows(plans: list[dict], bids: Path, manifest: dict) -> list[dict]:
    table = authors_table(bids)
    rows = []
    for p in plans:
        if p["scanner_label"] not in GE_AUTHOR_ROWS:
            continue
        m, r = p["_m_meta"], p["_r_meta"]
        ref = table.get(GE_AUTHOR_ROWS[p["scanner_label"]], {})

        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        ees, trt = m["EffectiveEchoSpacing"] * 1000, m["TotalReadoutTime"] * 1000
        a_ees, a_trt = num(ref.get("EES (ms)")), num(ref.get("TRT (ms)"))
        rows.append({
            "subject": p["subject"], "session": p["session"], "scanner": p["scanner_label"],
            "site": manifest[(p["subject"], p["session"])].get("site", ""),
            "SeriesDescription": m.get("SeriesDescription", ""), "ProtocolName": m.get("ProtocolName", ""),
            "EffectiveEchoSpacing_ms": f"{ees:.6g}", "TotalReadoutTime_ms": f"{trt:.6g}",
            "authors_table_EES_ms": ref.get("EES (ms)", ""), "authors_table_TRT_ms": ref.get("TRT (ms)", ""),
            "EES_diff_pct": f"{100 * (ees - a_ees) / a_ees:+.1f}" if a_ees else "",
            "TRT_diff_pct": f"{100 * (trt - a_trt) / a_trt:+.1f}" if a_trt else "",
            "reverse_EffectiveEchoSpacing_ms": f"{r.get('EffectiveEchoSpacing', 0) * 1000:.6g}",
            "reverse_TotalReadoutTime_ms": f"{r.get('TotalReadoutTime', 0) * 1000:.6g}",
            "PixelBandwidth_Hz_per_px": m.get("PixelBandwidth", ""),
            "AcquisitionMatrixPE": m.get("AcquisitionMatrixPE", ""), "ReconMatrixPE": m.get("ReconMatrixPE", ""),
            "ParallelReductionFactorInPlane": m.get("ParallelReductionFactorInPlane", ""),
            "MultibandAccelerationFactor": m.get("MultibandAccelerationFactor", ""),
            "gradient_mode": m.get("GradientMode", "not recorded in sidecar"),
            "PhaseEncodingPolarityGE": m.get("PhaseEncodingPolarityGE", ""),
            "SliceThickness_mm": m.get("SliceThickness", ""),
            "EchoTime_ms": f"{m['EchoTime'] * 1000:.4g}" if isinstance(m.get("EchoTime"), (int, float)) else "",
            "ScanOptions": m.get("ScanOptions", ""), "SoftwareVersions": m.get("SoftwareVersions", ""),
            "ConversionSoftwareVersion": m.get("ConversionSoftwareVersion", ""),
        })
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def md_table(header, rows) -> str:
    esc = lambda x: str(x).replace("|", "\\|")  # noqa: E731
    return "\n".join(["| " + " | ".join(map(esc, header)) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def render_report(plans, errors, staged, decisions, prov, plan_only) -> str:
    L = ["# 01_prepare_inputs - report\n",
         f"Generated {utcnow()} by `01_prepare_inputs.py` v{SCRIPT_VERSION} "
         f"({'plan only: metadata' if plan_only else 'plan + staging'}). "
         f"Decisions file sha256 `{decisions['sha256'][:12]}`. FSL {prov.get('fsl_version', 'n/a')}.\n",
         "Blinding: only sidecars, bval/bvec and NIfTI headers were read; staged images were produced by FSL "
         "tools and checked through their headers.\n",
         f"## Result: {'FAILED' if errors else 'OK'} - {len(plans)} sessions planned, {len(errors)} errors\n"]
    if errors:
        L.append("\n".join(f"- {e}" for e in errors) + "\n")
    by = defaultdict(list)
    for p in plans:
        by[p["scanner_label"]].append(p)
    rows = []
    for lab, ps in sorted(by.items()):
        ro = [float(p["_ro"]) for p in ps]
        sc = [float(p["readout_time_sidecar_s"]) for p in ps]
        rows.append([lab, len(ps), ", ".join(sorted({p["main_pe_direction"] for p in ps})),
                     ", ".join(sorted({p["field_sign_vs_true"] for p in ps})),
                     f"{min(ro) * 1000:.4f}-{max(ro) * 1000:.4f}",
                     f"{max(100 * (s - r) / s for s, r in zip(sc, ro)):.3f}",
                     ", ".join(sorted({str(p["main_n_vols"]) for p in ps})),
                     ", ".join(sorted({str(p["revpe_source_bval"]) for p in ps}))])
    L.append("## Per scanner\n")
    L.append(md_table(["scanner", "sessions", "main PED (sidecar)", "BRC field sign vs true",
                       "readout used by BRC (ms)", "max % below sidecar (bc truncation)",
                       "main volumes", "reverse b0 bval"], rows) + "\n")
    special = [[f"{p['subject']}/{p['session']}", p["brc_invocation"], p["notes"]]
               for p in plans if p["notes"] or p["brc_invocation"] != "dMRI_preproc.sh"]
    L.append("## Sessions with special handling\n")
    L.append((md_table(["session", "BRC invocation", "notes"], special) if special else "None.") + "\n")
    custom = [p for p in plans if p["_custom"]]
    for p in custom:
        L.append(f"### Custom TOPUP acqparams: {p['subject']}/{p['session']}\n\n```\n{custom_acqparams(p)}```\n"
                 f"Row 1 = main run (BRC convention); row 2 = reverse b0 ({p['_r_ped']}). "
                 "**An unblinded author checks the x sign on the TOPUP-corrected b0 before EDDY.**\n")
    if staged:
        L.append("## Staging\n")
        L.append(md_table(["session", "actions"], [[k, "; ".join(v)] for k, v in sorted(staged.items())]) + "\n")
    L.append("## Pending decisions\n")
    L.append((md_table(["id", "scope", "topic", "note"],
                       [[d["decision_id"], d["scope"], d.get("topic", ""), d.get("note", "")]
                        for d in decisions["pending"]]) if decisions["pending"] else "None.") + "\n")
    L.append("## How later steps use `preproc_inputs.csv`\n")
    L.append("- Denoised arm: `designer " + DESIGNER_ARGS + " <designer_input> <designer_output>`; then "
             "copy the raw bval/bvec next to the output after checking they are unchanged.\n"
             "- BRC, per arm: `dMRI_preproc.sh --input <main without extension> --input_2 <revpe_b0_nii without "
             "extension> --path <deriv>/brc/<arm> --subject <brc_subject> --echospacing <echospacing_arg> "
             "--pe_dir 2 --qc`, where main = `main_dwi_nii` (non-denoised arm) or `designer_output` (denoised arm). "
             "For `brc_invocation == substages_custom_topup`, use the route with `custom_topup_acqparams`.\n"
             "- Before any model fit, drop volume `brc_data_drop_vols` (the reverse b0) from BRC's `data.nii.gz`; "
             "keep `brc_data_main_vols`.\n"
             "- Signed field (if ever needed) = `field_sign_vs_true` x BRC field. H4 uses |Hz|. GE rescaling, "
             "if needed: Hz_true = Hz_est x `readout_time_used_s` / readout_true.\n")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
PATH_OPTIONS = ("bids_dir", "meta_dir", "deriv_dir", "log_dir", "decisions", "fsldir", "brc_dir")


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, help="JSON file with option values (keys = long option names); "
                                                 "command-line options override it")
    known, _ = pre.parse_known_args(argv)
    p = argparse.ArgumentParser(
        parents=[pre], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepare per-session DESIGNER/BRC inputs from the 00_download manifest and session_exceptions.csv.")
    p.add_argument("--bids-dir", type=Path, help="Raw BIDS root (read-only)")
    p.add_argument("--meta-dir", type=Path, help="00_download.py --out-dir (contains sessions_manifest.csv)")
    p.add_argument("--deriv-dir", type=Path, help="Derivatives root; outputs go to <deriv-dir>/inputs")
    p.add_argument("--log-dir", type=Path, help="Log directory (default: <deriv-dir>/logs)")
    p.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS, help="session_exceptions.csv")
    p.add_argument("--fsldir", type=Path, default=Path(os.environ.get("FSLDIR", "/usr/local/fsl/6.0.6.x")),
                   help=f"FSL installation (must be {EXPECTED_FSL} when staging)")
    p.add_argument("--brc-dir", type=Path, default=Path("/usr/local/BRC_Pipeline/1.6.3"),
                   help="BRC pipeline installation (version is checked, not run)")
    p.add_argument("--plan-only", action="store_true", help="Metadata only: do not stage any image")
    p.add_argument("--allow-inside-repo", action="store_true", help="Allow output paths inside the repository")
    if known.config:
        cfg = json.loads(known.config.read_text())
        dests = {a.dest for a in p._actions}
        norm = {}
        for k, v in cfg.items():
            key = k.lstrip("-").replace("-", "_")
            if key not in dests or key == "config":
                p.error(f"unknown key in {known.config}: {k}")
            if key in PATH_OPTIONS and v is not None:
                v = Path(os.path.expanduser(v))
                v = v if v.is_absolute() else (known.config.parent / v)
            norm[key] = v
        p.set_defaults(**norm)
    args = p.parse_args(argv)
    for opt in ("bids_dir", "meta_dir", "deriv_dir"):
        if not getattr(args, opt):
            p.error(f"--{opt.replace('_', '-')} is required (on the command line or in --config)")
    for opt in PATH_OPTIONS:
        if getattr(args, opt):
            setattr(args, opt, getattr(args, opt).expanduser().resolve())
    args.log_dir = args.log_dir or args.deriv_dir / "logs"
    if args.deriv_dir.is_relative_to(args.bids_dir):
        p.error("--deriv-dir must be outside --bids-dir (raw data stays clean)")
    inside = [str(getattr(args, k)) for k in ("deriv_dir", "log_dir") if getattr(args, k).is_relative_to(REPO_ROOT)]
    if inside and not args.allow_inside_repo:
        p.error(f"output paths inside the code repository {REPO_ROOT}: {inside}")
    return args


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"01_prepare_inputs_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)
    return path


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = args.deriv_dir / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(args.log_dir)
    script = Path(__file__).resolve()
    prov = {"script": str(script), "script_version": SCRIPT_VERSION, "script_sha256": sha256_of(script),
            "command": [sys.executable] + sys.argv, "started_utc": utcnow(), "python": sys.version.split()[0],
            "nibabel": nib.__version__, "numpy": np.__version__, "platform": platform.platform(),
            "args": {k: str(v) for k, v in vars(args).items()}, "log_file": str(log_path)}
    log.info("01_prepare_inputs.py v%s (sha256 %s)", SCRIPT_VERSION, prov["script_sha256"])
    log.info("Command: %s", " ".join(prov["command"]))

    manifest_path = args.meta_dir / "sessions_manifest.csv"
    manifest_rows = read_csv(manifest_path)
    prov["manifest"] = {"path": str(manifest_path), "sha256": sha256_of(manifest_path), "rows": len(manifest_rows)}
    decisions = load_decisions(args.decisions)
    prov["decisions"] = {"path": decisions["path"], "sha256": decisions["sha256"]}
    log.info("Manifest: %d sessions; decisions: %d rows (%d pending)", len(manifest_rows),
             len(decisions["rows"]), len(decisions["pending"]))
    for d in decisions["pending"]:
        log.warning("Pending decision %s (%s): %s", d["decision_id"], d.get("topic", ""), d.get("note", ""))

    brc_version = (args.brc_dir / "version.txt").read_text().strip() if (args.brc_dir / "version.txt").is_file() \
        else "not found"
    prov["brc_version"] = brc_version
    if brc_version != EXPECTED_BRC:
        log.warning("BRC at %s reports %s (expected %s); acqparams conventions assume BRC 1.6.3",
                    args.brc_dir, brc_version, EXPECTED_BRC)

    errors: list[str] = []
    manifest = {(r["subject"], r["session"]): r for r in manifest_rows}
    known_sessions = set(manifest)
    for key in decisions["by_session"]:
        if key not in known_sessions:
            errors.append(f"decision for {key[0]}/{key[1]} but that session is not in the manifest")
    for r in manifest_rows:
        if not r.get("scanner_label"):
            errors.append(f"{r['subject']}/{r['session']}: no scanner label in the manifest")

    plans = []
    for r in sorted(manifest_rows, key=lambda r: (r["subject"], r["session"])):
        p = plan_session(r, args.bids_dir, args.deriv_dir, decisions, errors)
        if p:
            plans.append(p)

    staged = {}
    if not args.plan_only and not errors:
        fsl = Fsl(args.fsldir)
        prov["fsl_version"] = fsl.version
        prov["fsldir"] = str(args.fsldir)
        if fsl.version != EXPECTED_FSL:
            log.error("FSL at %s is %s; staging requires the pinned %s", args.fsldir, fsl.version, EXPECTED_FSL)
            return 2
        log.info("Staging reverse-PE b0 images with FSL %s", fsl.version)
        for p in plans:
            try:
                staged[f"{p['subject']}/{p['session']}"] = stage_session(p, fsl, errors)
            except Exception as exc:  # noqa: BLE001 - collected and reported
                errors.append(f"{p['subject']}/{p['session']}: staging failed: {exc}")
                log.exception("Staging failed for %s/%s", p["subject"], p["session"])
    elif errors:
        log.error("Planning found %d errors; nothing staged", len(errors))

    write_csv(out_dir / "preproc_inputs.csv", PREP_COLUMNS, plans)
    ge = ge_rows(plans, args.bids_dir, manifest)
    write_csv(out_dir / "ge_readout_for_authors.csv", GE_COLUMNS, ge)
    (out_dir / "01_report.md").write_text(
        render_report(plans, errors, staged, decisions, prov, args.plan_only), encoding="utf-8")
    prov.update(finished_utc=utcnow(), n_sessions=len(plans), n_errors=len(errors), plan_only=args.plan_only,
                custom_topup_sessions=[f"{p['subject']}/{p['session']}" for p in plans if p["_custom"]])
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    for e in errors:
        log.error(e)
    log.info("Wrote %s (%d sessions), ge_readout_for_authors.csv (%d rows), 01_report.md",
             out_dir / "preproc_inputs.csv", len(plans), len(ge))
    kinds = Counter(p["brc_invocation"] for p in plans)
    log.info("BRC invocation: %s", dict(kinds))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
