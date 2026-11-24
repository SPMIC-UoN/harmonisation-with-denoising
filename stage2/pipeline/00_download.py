#!/usr/bin/env python3
"""
00_download.py - Download and organise the Phase B subset of ON-Harmony
(OpenNeuro ds004712) and produce a session manifest and a metadata-only QC
report.

Scope: download, organisation and metadata QC only. No preprocessing,
denoising or analysis happens here.

BLINDING: this script never reads voxel intensities. NIfTI files are opened
only to parse the fixed-size header (exactly sizeof_hdr = 348/540 bytes are
read); no image array is ever constructed. Integrity checksums are computed
over raw file bytes without decoding them. No figures or thumbnails are made.

Method: the OpenNeuro GraphQL API provides the file listing of a fixed
snapshot (default 2.0.1). Each file is fetched anonymously from the
OpenNeuro S3 bucket via a URL pinned to the S3 object version of that
snapshot, and verified against its git-annex key (SHA256) or git blob id
(SHA1). Only `requests` and `nibabel` are required.


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
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import requests
from nibabel.orientations import aff2axcodes

SCRIPT_VERSION = "1.0.0"
DATASET_ID = "ds004712"
DEFAULT_SNAPSHOT = "2.0.1"
GRAPHQL_URL = "https://openneuro.org/crn/graphql"
REPO_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("00_download")

# --------------------------------------------------------------------------
# What to download
# --------------------------------------------------------------------------
# Top-level BIDS files, copied unmodified (participants.tsv still lists the
# Phase A subjects; it is not edited).
TOP_LEVEL_BIDS = ("dataset_description.json", "participants.tsv", "participants.json",
                  "README", "CHANGES", ".bidsignore")
# Small provenance tables shipped by the authors (listed in .bidsignore).
# all_idps.csv (derived imaging phenotypes) and data_dictionary.pdf (which
# documents them) are derivatives and are deliberately NOT downloaded.
TOP_LEVEL_PROVENANCE = ("protocol_accelerations.csv", "software_versions.csv",
                        "scan_delays.csv", "radiographers.csv")
SESSION_FILE_PATTERNS = (
    re.compile(r"^sub-[^/]+/ses-[^/]+/anat/[^/]+_T1w\.(nii\.gz|json)$"),
    re.compile(r"^sub-[^/]+/ses-[^/]+/dwi/[^/]+_dwi\.(nii\.gz|json|bval|bvec)$"),
    re.compile(r"^sub-[^/]+/(ses-[^/]+/)?[^/]+_(sessions|scans)\.(tsv|json)$"),
)
# fmap/ EPI pairs are included only if their sidecar says they are intended
# for dMRI. (In snapshot 2.0.1 they exist only for Philips and target rs-fMRI.)
FMAP_JSON = re.compile(r"^(?P<sub>sub-[^/]+)/(?P<ses>ses-[^/]+)/fmap/(?P<name>[^/]+_epi)\.json$")
SESSION_DIR = re.compile(r"^(?P<sub>sub-[^/]+)/(?P<ses>ses-[^/]+)/")
SESSION_CODE = re.compile(r"^ses-(?P<code>[A-Z]{3}\d[A-Z]{3})(?P<num>\d{3})$")
NII_EXTS = ("nii.gz", "nii")


# --------------------------------------------------------------------------
# Scanner identification (from JSON sidecar metadata only)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScannerRule:
    label: str
    manufacturer: str          # regex on Manufacturer
    model: str                 # regex on ManufacturersModelName
    institution: str           # regex on InstitutionName
    coil: str | None           # regex on ReceiveCoilName / CoilString (None: not recorded by vendor)
    nominal_coil: str          # from the dataset documentation
    site: str                  # from the dataset documentation

    def matches(self, meta: dict) -> bool:
        def ok(pattern, value):
            return value is not None and re.search(pattern, str(value), re.I) is not None
        if not (ok(self.manufacturer, meta.get("Manufacturer"))
                and ok(self.model, meta.get("ManufacturersModelName"))
                and ok(self.institution, meta.get("InstitutionName"))):
            return False
        return self.coil is None or ok(self.coil, coil_of(meta)[1])


# A sidecar is assigned to a scanner only if it matches exactly one rule, and
# a session only if all of its sidecars agree. Otherwise it is flagged.
SCANNERS = (
    ScannerRule("Prisma32_FMRIB", r"^Siemens", r"Prisma", r"John Radcliffe", r"^Head_32$", "32ch",
                "WIN-FMRIB, John Radcliffe Hospital, Oxford"),
    ScannerRule("Prisma64_OHBA", r"^Siemens", r"Prisma", r"Warneford", r"^HeadNeck_64$", "64ch",
                "WIN-OHBA, Warneford Hospital, Oxford"),
    ScannerRule("Achieva32_SPMICUP", r"^Philips", r"Achieva", r"SPMMRC Nottingham", None, "32ch",
                "SPMIC-UP, University Park, Nottingham"),
    ScannerRule("Ingenia32_SPMICUP", r"^Philips", r"Ingenia", r"Nottingham University", None, "32ch",
                "SPMIC-UP, University Park, Nottingham"),
    ScannerRule("Premier48_SPMICQMC", r"^GE", r"Premier", r"Nottingham Medical School", r"^48", "48ch",
                "SPMIC-QMC, Queen's Medical Centre, Nottingham"),
    ScannerRule("Premier21_OCMR", r"^GE", r"Premier", r"OCMR", r"^21", "21ch",
                "OCMR, John Radcliffe Hospital, Oxford"),
)
SCANNER_BY_LABEL = {r.label: r for r in SCANNERS}
# Session-name codes, used ONLY to cross-check the metadata-based assignment.
SESSION_CODE_HINT = {"OXF1PRI": "Prisma32_FMRIB", "OXF2PRI": "Prisma64_OHBA",
                     "NOT1ACH": "Achieva32_SPMICUP", "NOT2ING": "Ingenia32_SPMICUP",
                     "NOT4GEP": "Premier48_SPMICQMC", "OXF4GEP": "Premier21_OCMR"}
IDENTITY_KEYS = ("Manufacturer", "ManufacturersModelName", "InstitutionName",
                 "StationName", "DeviceSerialNumber")

# Study design expectations checked in the QC report.
EXPECTED_N_SUBJECTS = 10
EXPECTED_N_REPEAT_SUBJECTS = 5
EXPECTED_N_REPEATS = 6
NO_REPEAT_SCANNER = "Achieva32_SPMICUP"
EXPECTED_N_SESSIONS = EXPECTED_N_SUBJECTS * len(SCANNERS) + EXPECTED_N_REPEAT_SUBJECTS * (EXPECTED_N_REPEATS - 1)
EXPECTED_VOXEL_MM = 2.0
NOMINAL_SHELLS = (0, 1000, 2000)
SHELL_TOLERANCE = 100          # s/mm^2
CONTINUOUS_REL_TOL = 0.05      # TE/TR/TI deviating >5% from the scanner median are flagged
READOUT_REL_TOL = 0.01         # dwi vs revPE TotalReadoutTime

# Deviations documented in the README of snapshot 2.0.1 ("Minor deviations",
# Phase B). These come from documentation, not metadata; the QC report states
# whether the metadata agrees with each one.
README_NOTES = {
    ("sub-16793", "ses-OXF4GEP001"): [
        ("readme_revpe_acquired_LR", "README: dMRI reversed-PE b0 acquired with LR phase encoding instead of PA."),
        ("readme_t1w_matrix_212x256x256", "README: T1w acquired with a 212x256x256 matrix instead of 256x256x212."),
    ],
    ("sub-16745", "ses-OXF4GEP001"): [
        ("readme_dwi_slice_2p4mm", "README: dMRI acquired with 2.4 mm slice thickness instead of 2.0 mm."),
    ],
    ("sub-16794", "ses-NOT4GEP001"): [
        ("readme_no_revpe_acquired", "README: no dMRI reversed-PE b0 was acquired; the authors synthesised one "
                                     "with Synb0-DisCo for their own processing."),
    ],
    ("sub-16794", "ses-OXF4GEP001"): [
        ("readme_t1w_matrix_212x256x256", "README: T1w acquired with a 212x256x256 matrix instead of 256x256x212."),
    ],
}

FLAG_DESCRIPTIONS = {
    "file_not_downloaded": "A file selected for download is not present locally (download failed or was skipped).",
    "missing_t1w": "No T1w NIfTI in anat/.",
    "missing_t1w_json": "T1w NIfTI has no JSON sidecar.",
    "multiple_t1w": "More than one T1w image; the first (sorted) is listed in the manifest.",
    "missing_dwi": "No dMRI run in dwi/.",
    "missing_dwi_json": "Main dMRI run has no JSON sidecar.",
    "missing_dwi_bval": "Main dMRI run has no .bval.",
    "missing_dwi_bvec": "Main dMRI run has no .bvec.",
    "missing_revpe": "No reverse phase-encode acquisition for the dMRI (neither in dwi/ nor in fmap/).",
    "missing_revpe_json": "Reverse-PE run has no JSON sidecar.",
    "multiple_revpe": "More than one candidate reverse-PE run; the first (sorted) is listed in the manifest.",
    "dwi_run_roles_ambiguous": "Two dMRI runs have the same number of volumes; main vs reverse-PE chosen by name.",
    "revpe_from_fmap": "Reverse-PE data taken from fmap/ (sidecar IntendedFor targets dwi/).",
    "header_unreadable": "A NIfTI header could not be parsed.",
    "bval_nvols_mismatch": "Number of b-values differs from the number of volumes in the NIfTI header.",
    "bvec_shape_mismatch": "bvec is not 3 x n_volumes.",
    "dwi_unexpected_bvals": "b-values present that are not within 100 s/mm^2 of 0, 1000 or 2000.",
    "dwi_voxsize_not_2mm": "dMRI voxel size differs from 2 mm isotropic by more than 0.05 mm.",
    "missing_pe_direction": "PhaseEncodingDirection missing from a dMRI/reverse-PE sidecar.",
    "pe_dir_label_mismatch_dwi": "Main dMRI: PhaseEncodingDirection combined with the header affine points "
                                 "the other way to the dir- entity in the file name.",
    "pe_dir_label_mismatch_revpe": "Reverse-PE: PhaseEncodingDirection combined with the header affine points "
                                   "the other way to the dir- entity in the file name.",
    "revpe_pe_axis_differs": "Reverse-PE run is phase-encoded along a different axis from the main dMRI.",
    "revpe_pe_not_reversed": "Reverse-PE run has the same PhaseEncodingDirection polarity as the main dMRI.",
    "revpe_geometry_mismatch": "Reverse-PE and main dMRI differ in matrix, voxel size or affine.",
    "revpe_readout_time_mismatch": "TotalReadoutTime of reverse-PE and main dMRI differ by more than 1%.",
    "scanner_unassigned": "Sidecar metadata matches none of the 6 scanner definitions.",
    "scanner_ambiguous": "Sidecars in this session match different or multiple scanner definitions.",
    "scanner_identity_inconsistent": "Identity fields (manufacturer/model/institution/station/serial) differ "
                                     "between sidecars within this session.",
    "scanner_differs_from_session_code": "Scanner assigned from metadata disagrees with the scanner code in the "
                                         "session name.",
    "nonmodal_*": "nonmodal_<run>_<field>: value differs from the most common value (or by >5% from the median "
                  "for TE/TR/TI) among all sessions of the same scanner.",
    "readme_*": "Deviation documented in the dataset README (not derived from metadata); see section 9.",
}


# --------------------------------------------------------------------------
# Remote listing, integrity and downloading
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    file_id: str          # git-annex key (annexed files) or git blob SHA1
    annexed: bool
    urls: tuple

    def checksum(self):
        """(algorithm, hexdigest) from the annex key or git blob id, or None."""
        if self.annexed:
            m = re.match(r"^(SHA256|SHA512|SHA1|MD5)E?-s(\d+)--([0-9a-fA-F]+)", self.file_id)
            if m:
                return m.group(1).lower(), m.group(3).lower()
            return None
        if re.fullmatch(r"[0-9a-f]{40}", self.file_id):
            return "gitblob", self.file_id
        return None


def safe_relpath(path: str) -> str:
    parts = path.split("/")
    if path.startswith("/") or "\\" in path or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe path in remote listing: {path!r}")
    return path


def graphql(query: str, timeout: int = 600, retries: int = 4) -> dict:
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(GRAPHQL_URL, json={"query": query}, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            if payload.get("errors"):
                raise RuntimeError(payload["errors"])
            return payload
        except Exception as exc:  # noqa: BLE001 - retried and re-raised
            last = exc
            log.warning("GraphQL attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"OpenNeuro GraphQL query failed: {last}")


def get_listing(snapshot: str, cache_dir: Path, refresh: bool) -> dict:
    """Snapshot file listing; cached, since a snapshot is immutable."""
    cache = cache_dir / f"listing_{DATASET_ID}_{snapshot}.json"
    if cache.exists() and not refresh:
        log.info("Using cached listing %s", cache)
        listing = json.loads(cache.read_text())
    else:
        log.info("Fetching file listing of %s snapshot %s from %s", DATASET_ID, snapshot, GRAPHQL_URL)
        q = ('{ snapshot(datasetId: %s, tag: %s) { tag hexsha created '
             'files(recursive: true) { id filename size directory annexed urls } } }'
             % (json.dumps(DATASET_ID), json.dumps(snapshot)))
        payload = graphql(q)
        snap = payload["data"]["snapshot"]
        if snap is None:
            raise RuntimeError(f"snapshot {snapshot} of {DATASET_ID} not found")
        listing = {
            "dataset": DATASET_ID, "tag": snap["tag"], "hexsha": snap["hexsha"],
            "created": snap.get("created"),
            "openneuro_api_version": payload.get("extensions", {}).get("openneuro", {}).get("version"),
            "fetched_utc": utcnow(),
            "files": [f for f in snap["files"] if not f["directory"]],
        }
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(listing))
        tmp.replace(cache)
    files = [RemoteFile(safe_relpath(f["filename"]), int(f["size"]), f["id"], bool(f["annexed"]),
                        tuple(f.get("urls") or ())) for f in listing["files"]]
    log.info("Snapshot %s (git %s, created %s): %d files, %.1f GB",
             listing["tag"], listing["hexsha"], listing.get("created"), len(files),
             sum(f.size for f in files) / 1e9)
    return {**{k: v for k, v in listing.items() if k != "files"}, "files": files}


def latest_snapshot_tag() -> str | None:
    try:
        q = '{ dataset(id: %s) { latestSnapshot { tag } } }' % json.dumps(DATASET_ID)
        return graphql(q, timeout=60, retries=2)["data"]["dataset"]["latestSnapshot"]["tag"]
    except Exception as exc:  # noqa: BLE001 - informational only
        log.warning("Could not query latest snapshot: %s", exc)
        return None


def verify_file(path: Path, rf: RemoteFile) -> tuple[bool, str]:
    size = path.stat().st_size
    if size != rf.size:
        return False, f"size {size} != {rf.size}"
    spec = rf.checksum()
    if spec is None:
        return True, "size-only"
    algo, expected = spec
    if algo == "gitblob":
        h = hashlib.sha1()
        h.update(b"blob %d\0" % size)
    else:
        h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        return False, f"{algo} mismatch ({got} != {expected})"
    return True, "git-blob-sha1" if algo == "gitblob" else algo


def make_writable(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | 0o200)
    except OSError:
        pass


class VerifiedState:
    """Records files already verified, so re-runs skip re-hashing them."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def is_current(self, rf: RemoteFile, dest: Path) -> bool:
        entry = self.data.get(rf.path)
        if not entry or entry["id"] != rf.file_id:
            return False
        st = dest.stat()
        return entry["size"] == st.st_size == rf.size and entry["mtime_ns"] == st.st_mtime_ns

    def record(self, rf: RemoteFile, dest: Path, method: str) -> None:
        st = dest.stat()
        with self.lock:
            self.data[rf.path] = {"id": rf.file_id, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                  "method": method, "verified_utc": utcnow()}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=0))
            tmp.replace(self.path)


class Downloader:
    def __init__(self, staging_dir: Path, retries: int, timeout: int, readonly: bool):
        self.staging_dir = staging_dir
        self.retries = retries
        self.timeout = timeout
        self.readonly = readonly
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers["User-Agent"] = f"on-harmony-00_download/{SCRIPT_VERSION}"
            self._local.s = s
        return self._local.s

    def fetch(self, rf: RemoteFile, dest: Path, readonly: bool | None = None) -> str:
        """Download rf to dest (resuming a partial file), verify, then move into place."""
        if not rf.urls:
            raise RuntimeError(f"{rf.path}: no download URL in listing")
        part = self.staging_dir / (rf.path + ".part")
        last = None
        for attempt in range(1, self.retries + 1):
            url = rf.urls[(attempt - 1) % len(rf.urls)]
            try:
                self._stream(url, rf, part)
                ok, how = verify_file(part, rf)
                if not ok:
                    part.unlink()
                    raise IOError(f"integrity check failed: {how}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    make_writable(dest)
                    dest.unlink()
                shutil.move(str(part), str(dest))
                if self.readonly if readonly is None else readonly:
                    dest.chmod(0o444)
                return how
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last = exc
                log.warning("%s: attempt %d/%d failed: %s", rf.path, attempt, self.retries, exc)
                if attempt < self.retries:
                    time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"{rf.path}: failed after {self.retries} attempts: {last}")

    def _stream(self, url: str, rf: RemoteFile, part: Path) -> None:
        part.parent.mkdir(parents=True, exist_ok=True)
        have = part.stat().st_size if part.exists() else 0
        if have > rf.size:
            part.unlink()
            have = 0
        if have == rf.size and have > 0:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        log.debug("GET %s (resume from %d bytes)", url, have)
        with self._session().get(url, headers=headers, stream=True, timeout=(30, self.timeout)) as r:
            if r.status_code == 416:
                part.unlink(missing_ok=True)
                raise IOError("HTTP 416 on resume; partial file discarded")
            r.raise_for_status()
            mode = "ab" if have and r.status_code == 206 else "wb"
            with open(part, mode) as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        got = part.stat().st_size
        if got > rf.size:
            part.unlink()
        if got != rf.size:
            raise IOError(f"received {got} bytes, expected {rf.size}")


def ensure_file(rf: RemoteFile, bids_dir: Path, dl: Downloader, state: VerifiedState, reverify: bool) -> str:
    dest = bids_dir / rf.path
    if dest.exists():
        if not reverify and state.is_current(rf, dest):
            return "skipped (verified earlier)"
        ok, how = verify_file(dest, rf)
        if ok:
            state.record(rf, dest, how)
            return f"verified existing ({how})"
        log.warning("%s: existing file fails integrity check (%s); re-downloading", rf.path, how)
    how = dl.fetch(rf, dest)
    state.record(rf, dest, how)
    return f"downloaded ({how})"


def fetch_cached(rf: RemoteFile, cache_dir: Path, dl: Downloader) -> Path:
    """Fetch a small text file (metadata) into the cache for planning."""
    dest = cache_dir / "meta" / rf.path
    if dest.exists() and verify_file(dest, rf)[0]:
        return dest
    dl.fetch(rf, dest, readonly=False)
    return dest


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
@dataclass
class Plan:
    subjects: list
    sessions: dict                      # subject -> sorted session labels (from the listing)
    selected: list                      # RemoteFile
    fmap_included: list = field(default_factory=list)
    fmap_excluded: Counter = field(default_factory=Counter)
    excluded_subjects: list = field(default_factory=list)


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def log_structure(files: list[RemoteFile], participants: list[dict]) -> None:
    phase_of = {r["participant_id"]: r.get("phase", "?") for r in participants}
    sessions = defaultdict(set)
    datatypes = Counter()
    for rf in files:
        m = SESSION_DIR.match(rf.path)
        if m:
            sessions[m["sub"]].add(m["ses"])
            parts = rf.path.split("/")
            if len(parts) > 3:
                datatypes[parts[2]] += 1
    by_phase = defaultdict(list)
    for sub in sorted(sessions):
        by_phase[phase_of.get(sub, "not in participants.tsv")].append(sub)
    for phase, subs in sorted(by_phase.items()):
        codes = Counter(SESSION_CODE.match(s)["code"] if SESSION_CODE.match(s) else s
                        for sub in subs for s in sessions[sub])
        log.info("Phase %s: %d subjects, %d sessions; session codes %s", phase, len(subs),
                 sum(len(sessions[s]) for s in subs), dict(sorted(codes.items())))
    log.info("Files per datatype (all subjects): %s", dict(sorted(datatypes.items())))


def build_plan(files: list[RemoteFile], participants: list[dict], phase: str, subjects_filter,
               cache_dir: Path, dl: Downloader) -> Plan:
    by_path = {rf.path: rf for rf in files}
    subjects = sorted(r["participant_id"] for r in participants if r.get("phase") == phase)
    if subjects_filter:
        unknown = set(subjects_filter) - set(subjects)
        if unknown:
            raise SystemExit(f"--subjects not in phase {phase}: {sorted(unknown)}")
        subjects = sorted(subjects_filter)
    if not subjects:
        raise SystemExit(f"No subjects with phase == {phase!r} in participants.tsv")
    wanted = set(subjects)
    all_subjects = {rf.path.split("/")[0] for rf in files if rf.path.startswith("sub-")}
    plan = Plan(subjects=subjects, sessions={}, selected=[],
                excluded_subjects=sorted(all_subjects - wanted))

    for name in TOP_LEVEL_BIDS + TOP_LEVEL_PROVENANCE:
        if name in by_path:
            plan.selected.append(by_path[name])
        else:
            (log.warning if name in TOP_LEVEL_BIDS else log.info)("Top-level file %s not in snapshot", name)

    sessions = defaultdict(set)
    fmap_jsons = []
    for rf in files:
        sub = rf.path.split("/")[0]
        if sub not in wanted:
            continue
        m = SESSION_DIR.match(rf.path)
        if m:
            sessions[sub].add(m["ses"])
        if any(p.match(rf.path) for p in SESSION_FILE_PATTERNS):
            plan.selected.append(rf)
        elif FMAP_JSON.match(rf.path):
            fmap_jsons.append(rf)
    plan.sessions = {s: sorted(sessions[s]) for s in subjects}

    # Inspect fmap sidecars (text) to decide whether they belong to the dMRI.
    with ThreadPoolExecutor(max_workers=8) as ex:
        local = dict(zip(fmap_jsons, ex.map(lambda r: fetch_cached(r, cache_dir, dl), fmap_jsons)))
    for rf, path in local.items():
        meta = json.loads(path.read_text())
        intended = meta.get("IntendedFor") or []
        intended = [intended] if isinstance(intended, str) else intended
        targets = sorted({re.sub(r"^bids::", "", t).split("/")[-2] if "/" in t else "?" for t in intended})
        if any("/dwi/" in t or t.startswith("dwi/") for t in intended):
            stem = rf.path[:-len(".json")]
            for ext in NII_EXTS + ("json",):
                if f"{stem}.{ext}" in by_path:
                    plan.selected.append(by_path[f"{stem}.{ext}"])
                    plan.fmap_included.append(f"{stem}.{ext}")
        else:
            plan.fmap_excluded[f"IntendedFor datatype(s) {targets or ['none']}"] += 1
    if plan.fmap_excluded:
        log.info("fmap/ EPI sidecars not intended for dMRI (excluded): %s", dict(plan.fmap_excluded))
    if plan.fmap_included:
        log.info("fmap/ files intended for dMRI (included): %d", len(plan.fmap_included))

    plan.selected = sorted({rf.path: rf for rf in plan.selected}.values(), key=lambda r: r.path)
    return plan


# --------------------------------------------------------------------------
# Header / sidecar reading (metadata only; never voxel data)
# --------------------------------------------------------------------------
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
    shape = tuple(int(x) for x in hdr.get_data_shape())
    zooms = tuple(float(x) for x in hdr.get_zooms())
    return {"shape": shape, "zooms": zooms, "affine": hdr.get_best_affine()}


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def coil_of(meta: dict) -> tuple[str, str | None]:
    for key in ("ReceiveCoilName", "CoilString"):
        if meta.get(key) not in (None, ""):
            return key, str(meta[key])
    return "", None


def fmt_num(x):
    if x is None or x == "":
        return ""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:.6g}" if isinstance(x, float) else str(x)
    return str(x)


def keyed(meta: dict, keys) -> str:
    """'value (SidecarKey)' for the first key present; '' if none."""
    for k in keys:
        if meta.get(k) not in (None, ""):
            return f"{fmt_num(meta[k])} ({k})"
    return ""


def partial_fourier(meta: dict) -> str:
    keys = ("PartialFourier", "PartialFourierEnabled", "PartialFourierDirection",
            "PhaseEncodingStepsNoPartialFourier")
    return "; ".join(f"{k}={fmt_num(meta[k])}" for k in keys if meta.get(k) not in (None, ""))


OPPOSITE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def pe_target(ped: str | None, affine) -> str:
    """World-space letter the PE direction points to (e.g. 'P' for an A->P acquisition)."""
    if not ped or ped[0] not in "ijk":
        return ""
    letter = aff2axcodes(affine)["ijk".index(ped[0])]
    if letter is None:
        return ""
    return OPPOSITE[letter] if ped.endswith("-") else letter


def shell_summary(bvals: list[float]) -> tuple[Counter, list[float]]:
    counts, unexpected = Counter(), []
    for b in bvals:
        nominal = min(NOMINAL_SHELLS, key=lambda s: abs(s - b))
        if abs(nominal - b) <= SHELL_TOLERANCE:
            counts[nominal] += 1
        else:
            counts[int(round(b / 100.0) * 100)] += 1
            unexpected.append(b)
    return counts, unexpected


def describe_run(kind: str, files: dict, bids_dir: Path, flags: set) -> dict:
    """Metadata record for one run (t1w / dwi / revpe). files: ext -> relpath (local only)."""
    rec = {"run": kind, "nii": files.get("nii", ""), "json": files.get("json", ""),
           "bval": files.get("bval", ""), "bvec": files.get("bvec", "")}
    m = re.search(r"_dir-([A-Za-z]+)_", rec["nii"] or rec["json"])
    rec["dir_entity"] = m.group(1) if m else ""
    meta = read_json(bids_dir / rec["json"]) if rec["json"] else {}
    rec["_meta"] = meta
    hdr = None
    if rec["nii"]:
        try:
            hdr = read_nifti_header(bids_dir / rec["nii"])
        except Exception as exc:  # noqa: BLE001 - recorded as a flag
            log.error("Header of %s unreadable: %s", rec["nii"], exc)
            flags.add("header_unreadable")
    rec["_hdr"] = hdr
    if hdr:
        rec["dims"] = "x".join(str(d) for d in hdr["shape"][:3])
        rec["nvols"] = hdr["shape"][3] if len(hdr["shape"]) > 3 else 1
        rec["voxel_mm"] = "x".join(f"{z:.2f}" for z in hdr["zooms"][:3])
        rec["orientation"] = "".join(c or "?" for c in aff2axcodes(hdr["affine"]))
    if rec["bval"]:
        bvals = [float(x) for x in (bids_dir / rec["bval"]).read_text().split()]
        counts, unexpected = shell_summary(bvals)
        rec["n_bval"] = len(bvals)
        rec["shells"] = ";".join(f"{b}:{n}" for b, n in sorted(counts.items()))
        rec["n_b0"] = counts.get(0, 0)
        rec["n_dw"] = len(bvals) - counts.get(0, 0)
        rec["_unexpected_b"] = unexpected
        if hdr and len(bvals) != rec["nvols"]:
            flags.add("bval_nvols_mismatch")
    if rec["bvec"]:
        rows = [ln.split() for ln in (bids_dir / rec["bvec"]).read_text().splitlines() if ln.strip()]
        rec["bvec_shape"] = f"{len(rows)}x{','.join(sorted({str(len(r)) for r in rows}))}"
        if hdr and (len(rows) != 3 or any(len(r) != rec["nvols"] for r in rows)):
            flags.add("bvec_shape_mismatch")
    te = meta.get("EchoTime")
    rec["TE_ms"] = round(te * 1000, 3) if isinstance(te, (int, float)) else ""
    rec["TR_s"] = meta.get("RepetitionTime", "")
    rec["TI_s"] = meta.get("InversionTime", "")
    rec["flip_deg"] = meta.get("FlipAngle", "")
    rec["PED"] = meta.get("PhaseEncodingDirection", "")
    rec["pe_target"] = pe_target(rec["PED"], hdr["affine"]) if hdr else ""
    rec["MB"] = keyed(meta, ("MultibandAccelerationFactor", "ParallelReductionOutOfPlane"))
    rec["PI"] = keyed(meta, ("ParallelReductionFactorInPlane",))
    rec["PF"] = partial_fourier(meta)
    rec["EffectiveEchoSpacing_s"] = meta.get("EffectiveEchoSpacing", "")
    rec["TotalReadoutTime_s"] = meta.get("TotalReadoutTime", "")
    rec["SliceThickness_mm"] = meta.get("SliceThickness", "")
    for k in ("SeriesDescription", "SoftwareVersions") + IDENTITY_KEYS:
        rec[k] = meta.get(k, "")
    rec["coil"] = coil_of(meta)[1] or ""
    return rec


# --------------------------------------------------------------------------
# Sessions, manifest and QC
# --------------------------------------------------------------------------
@dataclass
class Session:
    subject: str
    session: str
    phase: str
    code: str = ""
    number: str = ""
    runs: dict = field(default_factory=dict)      # kind -> record
    revpe_source: str = ""
    label: str = ""
    identity: dict = field(default_factory=dict)
    flags: set = field(default_factory=set)
    notes: list = field(default_factory=list)
    repeat_index: int | None = None
    n_on_scanner: int | None = None
    is_repeat_subject: bool = False


def group_runs(paths: list[str], datatype: str) -> dict:
    """stem -> {ext: relpath} for files in <datatype>/."""
    runs = defaultdict(dict)
    for p in paths:
        if f"/{datatype}/" not in p:
            continue
        m = re.match(r"^(.*?)\.(nii\.gz|nii|json|bval|bvec)$", p)
        if m:
            runs[m.group(1)]["nii" if m.group(2).startswith("nii") else m.group(2)] = p
    return runs


def assemble_session(sub: str, ses: str, phase: str, selected: list[str], bids_dir: Path) -> Session:
    s = Session(sub, ses, phase)
    m = SESSION_CODE.match(ses)
    if m:
        s.code, s.number = m["code"], m["num"]
    local = []
    for p in selected:
        if (bids_dir / p).is_file():
            local.append(p)
        else:
            s.flags.add("file_not_downloaded")
            s.notes.append(f"not downloaded: {p}")

    # T1w
    t1 = {k: v for k, v in group_runs(local, "anat").items() if k.endswith("_T1w")}
    t1_with_nii = sorted(k for k, v in t1.items() if "nii" in v)
    if not t1_with_nii:
        s.flags.add("missing_t1w")
    else:
        if len(t1_with_nii) > 1:
            s.flags.add("multiple_t1w")
        files = t1[t1_with_nii[0]]
        if "json" not in files:
            s.flags.add("missing_t1w_json")
        s.runs["t1w"] = describe_run("t1w", files, bids_dir, s.flags)

    # dMRI: main run = most volumes; other dwi runs are reverse-PE candidates.
    dwi = {k: v for k, v in group_runs(local, "dwi").items() if "nii" in v}
    described = {k: describe_run("dwi", v, bids_dir, s.flags) for k, v in sorted(dwi.items())}
    if not described:
        s.flags.add("missing_dwi")
    else:
        def nvols(k):
            rec = described[k]
            return rec.get("n_bval") or rec.get("nvols") or 0
        order = sorted(described, key=lambda k: (-nvols(k), k))
        if len(order) > 1 and nvols(order[0]) == nvols(order[1]):
            s.flags.add("dwi_run_roles_ambiguous")
        s.runs["dwi"] = described[order[0]]
        for ext, flag in (("json", "missing_dwi_json"), ("bval", "missing_dwi_bval"), ("bvec", "missing_dwi_bvec")):
            if not s.runs["dwi"][ext]:
                s.flags.add(flag)
        rev = order[1:]
        if len(rev) > 1:
            s.flags.add("multiple_revpe")
        if rev:
            s.runs["revpe"] = dict(described[rev[0]], run="revpe")
            s.revpe_source = "dwi"
    if "revpe" not in s.runs:
        fmap = {k: v for k, v in group_runs(local, "fmap").items() if "nii" in v}
        main_ped = s.runs.get("dwi", {}).get("PED", "")
        cands = [describe_run("revpe", v, bids_dir, s.flags) for k, v in sorted(fmap.items())]
        cands = [c for c in cands if c["PED"] and main_ped and c["PED"][0] == main_ped[0]
                 and c["PED"].endswith("-") != main_ped.endswith("-")] or cands
        if cands:
            s.runs["revpe"] = cands[0]
            s.revpe_source = "fmap"
            s.flags.add("revpe_from_fmap")
            if len(cands) > 1:
                s.flags.add("multiple_revpe")
    if "revpe" not in s.runs:
        s.flags.add("missing_revpe")
    elif not s.runs["revpe"]["json"]:
        s.flags.add("missing_revpe_json")

    # Per-session consistency checks (metadata only).
    d, r = s.runs.get("dwi"), s.runs.get("revpe")
    if d:
        if d.get("_unexpected_b"):
            s.flags.add("dwi_unexpected_bvals")
        if d["_hdr"] and any(abs(z - EXPECTED_VOXEL_MM) > 0.05 for z in d["_hdr"]["zooms"][:3]):
            s.flags.add("dwi_voxsize_not_2mm")
    for kind, rec in (("dwi", d), ("revpe", r)):
        if not rec:
            continue
        if rec["json"] and not rec["PED"]:
            s.flags.add("missing_pe_direction")
        ent = rec["dir_entity"]
        if len(ent) == 2 and rec["pe_target"] and ent[1].upper() != rec["pe_target"]:
            s.flags.add(f"pe_dir_label_mismatch_{kind}")
    if d and r and d["PED"] and r["PED"]:
        if d["PED"][0] != r["PED"][0]:
            s.flags.add("revpe_pe_axis_differs")
        elif d["PED"].endswith("-") == r["PED"].endswith("-"):
            s.flags.add("revpe_pe_not_reversed")
    if d and r and d["_hdr"] and r["_hdr"]:
        dh, rh = d["_hdr"], r["_hdr"]
        same = (dh["shape"][:3] == rh["shape"][:3]
                and all(abs(a - b) < 1e-3 for a, b in zip(dh["zooms"][:3], rh["zooms"][:3]))
                and float(abs(dh["affine"] - rh["affine"]).max()) < 1e-2)
        if not same:
            s.flags.add("revpe_geometry_mismatch")
    if d and r and isinstance(d["TotalReadoutTime_s"], (int, float)) and isinstance(r["TotalReadoutTime_s"], (int, float)):
        if abs(d["TotalReadoutTime_s"] - r["TotalReadoutTime_s"]) > READOUT_REL_TOL * d["TotalReadoutTime_s"]:
            s.flags.add("revpe_readout_time_mismatch")

    # Scanner assignment: every sidecar must match exactly one rule, and all must agree.
    metas = {k: rec["_meta"] for k, rec in s.runs.items() if rec["_meta"]}
    labels = {}
    for kind, meta in metas.items():
        hits = [rule.label for rule in SCANNERS if rule.matches(meta)]
        labels[kind] = hits[0] if len(hits) == 1 else None
    identities = {tuple((k, str(meta.get(k, ""))) for k in IDENTITY_KEYS) + (("coil", coil_of(meta)[1] or ""),)
                  for meta in metas.values()}
    if len(identities) > 1:
        s.flags.add("scanner_identity_inconsistent")
    distinct = set(labels.values())
    if not labels or distinct == {None}:
        s.flags.add("scanner_unassigned")
    elif len(distinct) > 1:
        s.flags.add("scanner_ambiguous")
        s.notes.append("per-sidecar scanner match: " + ", ".join(f"{k}={v}" for k, v in labels.items()))
    else:
        s.label = distinct.pop()
    ref = metas.get("dwi") or metas.get("t1w") or next(iter(metas.values()), {})
    coil_key, coil_val = coil_of(ref)
    s.identity = {
        "manufacturer": ref.get("Manufacturer", ""), "model": ref.get("ManufacturersModelName", ""),
        "receive_coil": coil_val or "", "receive_coil_source": f"sidecar:{coil_key}" if coil_key else "not_in_sidecar",
        "institution": ref.get("InstitutionName", ""), "station_name": ref.get("StationName", ""),
        "device_serial_number": ref.get("DeviceSerialNumber", ""), "software_versions": ref.get("SoftwareVersions", ""),
    }
    hint = SESSION_CODE_HINT.get(s.code)
    if s.label and hint != s.label:
        s.flags.add("scanner_differs_from_session_code")
    for code, text in README_NOTES.get((sub, ses), []):
        s.flags.add(code)
        s.notes.append(text)
    return s


# T1w matrix size is not compared: the released T1w images are cropped per session.
DISCRETE_FIELDS = {"dwi": ("dims", "nvols", "voxel_mm", "orientation", "shells", "PED", "MB", "PI", "PF"),
                   "revpe": ("dims", "nvols", "voxel_mm", "orientation", "shells", "PED"),
                   "t1w": ("voxel_mm", "orientation")}
CONTINUOUS_FIELDS = {"dwi": ("TE_ms", "TR_s"), "revpe": ("TE_ms", "TR_s"), "t1w": ("TE_ms", "TR_s", "TI_s")}


def dataset_level(sessions: list[Session]) -> None:
    """Repeat indices and flags relative to all sessions of the same scanner."""
    groups = defaultdict(list)
    for s in sessions:
        if s.label:
            groups[(s.subject, s.label)].append(s)
    repeat_subjects = {sub for (sub, _), g in groups.items() if len(g) > 1}
    for g in groups.values():
        g.sort(key=lambda s: s.session)
        for i, s in enumerate(g, 1):
            s.repeat_index, s.n_on_scanner = i, len(g)
    for s in sessions:
        s.is_repeat_subject = s.subject in repeat_subjects

    by_scanner = defaultdict(list)
    for s in sessions:
        if s.label:
            by_scanner[s.label].append(s)
    for members in by_scanner.values():
        if len(members) < 3:
            continue
        for kind in ("dwi", "revpe", "t1w"):
            recs = [(s, s.runs[kind]) for s in members if kind in s.runs]
            for f in DISCRETE_FIELDS[kind]:
                vals = Counter(str(rec.get(f, "")) for _, rec in recs)
                mode = vals.most_common(1)[0][0] if vals else None
                for s, rec in recs:
                    if str(rec.get(f, "")) != mode:
                        s.flags.add(f"nonmodal_{kind}_{f.lower()}")
            for f in CONTINUOUS_FIELDS[kind]:
                nums = [rec[f] for _, rec in recs if isinstance(rec.get(f), (int, float))]
                if not nums:
                    continue
                med = statistics.median(nums)
                for s, rec in recs:
                    v = rec.get(f)
                    if isinstance(v, (int, float)) and med and abs(v - med) > CONTINUOUS_REL_TOL * abs(med):
                        s.flags.add(f"nonmodal_{kind}_{f.lower()}")


MANIFEST_COLUMNS = (
    "subject", "session", "phase", "scanner_label", "session_scanner_code",
    "manufacturer", "model", "receive_coil", "receive_coil_source", "receive_coil_nominal",
    "institution", "site", "station_name", "device_serial_number", "software_versions",
    "repeat_index", "n_sessions_on_scanner", "is_repeat_subject",
    "t1w_nii", "t1w_json",
    "dwi_nii", "dwi_bval", "dwi_bvec", "dwi_json", "dwi_phase_encoding_direction", "dwi_total_readout_time",
    "revpe_source", "revpe_nii", "revpe_bval", "revpe_bvec", "revpe_json",
    "revpe_phase_encoding_direction", "revpe_total_readout_time",
    "flags",
)


def manifest_row(s: Session) -> dict:
    rule = SCANNER_BY_LABEL.get(s.label)
    t1, d, r = (s.runs.get(k, {}) for k in ("t1w", "dwi", "revpe"))
    return {
        "subject": s.subject, "session": s.session, "phase": s.phase, "scanner_label": s.label,
        "session_scanner_code": s.code, **{k: s.identity.get(k, "") for k in (
            "manufacturer", "model", "receive_coil", "receive_coil_source")},
        "receive_coil_nominal": rule.nominal_coil if rule else "",
        "institution": s.identity.get("institution", ""), "site": rule.site if rule else "",
        "station_name": s.identity.get("station_name", ""),
        "device_serial_number": s.identity.get("device_serial_number", ""),
        "software_versions": s.identity.get("software_versions", ""),
        "repeat_index": s.repeat_index or "", "n_sessions_on_scanner": s.n_on_scanner or "",
        "is_repeat_subject": str(s.is_repeat_subject),
        "t1w_nii": t1.get("nii", ""), "t1w_json": t1.get("json", ""),
        "dwi_nii": d.get("nii", ""), "dwi_bval": d.get("bval", ""), "dwi_bvec": d.get("bvec", ""),
        "dwi_json": d.get("json", ""), "dwi_phase_encoding_direction": d.get("PED", ""),
        "dwi_total_readout_time": d.get("TotalReadoutTime_s", ""),
        "revpe_source": s.revpe_source, "revpe_nii": r.get("nii", ""), "revpe_bval": r.get("bval", ""),
        "revpe_bvec": r.get("bvec", ""), "revpe_json": r.get("json", ""),
        "revpe_phase_encoding_direction": r.get("PED", ""),
        "revpe_total_readout_time": r.get("TotalReadoutTime_s", ""),
        "flags": ";".join(sorted(s.flags)),
    }


RUN_COLUMNS = ("subject", "session", "scanner_label", "run", "nii", "json", "bval", "bvec", "dir_entity",
               "dims", "nvols", "voxel_mm", "orientation", "n_bval", "bvec_shape", "shells", "n_b0", "n_dw",
               "TE_ms", "TR_s", "TI_s", "flip_deg", "PED", "pe_target", "MB", "PI", "PF",
               "EffectiveEchoSpacing_s", "TotalReadoutTime_s", "SliceThickness_mm",
               "SeriesDescription", "SoftwareVersions") + IDENTITY_KEYS + ("coil",)


def write_csv(path: Path, columns, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    tmp.replace(path)


def summarise_values(values: list, continuous: bool) -> tuple[str, int]:
    present = [v for v in values if v not in ("", None)]
    missing = len(values) - len(present)
    if not present:
        return "not in sidecar/header", missing
    if continuous and all(isinstance(v, (int, float)) for v in present):
        lo, hi, med = min(present), max(present), statistics.median(present)
        text = fmt_num(med) if hi - lo <= 1e-9 else f"{fmt_num(med)} [{fmt_num(lo)}-{fmt_num(hi)}]"
    else:
        counts = Counter(str(v) for v in present)
        (mode, k), = counts.most_common(1)
        text = mode if len(counts) == 1 else f"{mode} ({k}/{len(present)}); other: " + ", ".join(
            f"{v} ({n})" for v, n in counts.most_common()[1:])
    if missing:
        text += f" [missing in {missing}]"
    return text, missing


def dims_range(values: list) -> str:
    dims = [tuple(int(x) for x in v.split("x")) for v in values if v]
    if not dims:
        return "not in header"
    return "cropped per session: " + " x ".join(f"{min(d[i] for d in dims)}-{max(d[i] for d in dims)}"
                                                 for i in range(3))


PROTOCOL_FIELDS = {
    "dwi": ("dims", "voxel_mm", "nvols", "shells", "n_b0", "n_dw", "TE_ms", "TR_s", "PED", "MB", "PI", "PF",
            "EffectiveEchoSpacing_s", "TotalReadoutTime_s", "SliceThickness_mm", "SeriesDescription"),
    "revpe": ("dims", "voxel_mm", "nvols", "shells", "TE_ms", "TR_s", "PED", "TotalReadoutTime_s",
              "SeriesDescription"),
    "t1w": ("dims", "voxel_mm", "orientation", "TE_ms", "TR_s", "TI_s", "flip_deg", "SeriesDescription"),
}
CONTINUOUS_SUMMARY = {"TE_ms", "TR_s", "TI_s", "flip_deg", "EffectiveEchoSpacing_s", "TotalReadoutTime_s",
                      "SliceThickness_mm"}


def md_table(header: list, rows: list) -> str:
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def build_outputs(plan: Plan, participants: list[dict], bids_dir: Path, out_dir: Path, provenance: dict,
                  download_summary: dict) -> None:
    log.info("Building manifest and metadata-only QC report")
    phase_of = {r["participant_id"]: r.get("phase", "") for r in participants}
    selected_by_session = defaultdict(list)
    for rf in plan.selected:
        m = SESSION_DIR.match(rf.path)
        if m:
            selected_by_session[(m["sub"], m["ses"])].append(rf.path)
    sessions = [assemble_session(sub, ses, phase_of.get(sub, ""), selected_by_session[(sub, ses)], bids_dir)
                for sub in plan.subjects for ses in plan.sessions[sub]]
    dataset_level(sessions)
    sessions.sort(key=lambda s: (s.subject, s.label or "~", s.session))

    manifest = out_dir / "sessions_manifest.csv"
    write_csv(manifest, MANIFEST_COLUMNS, [manifest_row(s) for s in sessions])
    runs = [{**{k: v for k, v in rec.items() if not k.startswith("_")},
             "subject": s.subject, "session": s.session, "scanner_label": s.label}
            for s in sessions for rec in s.runs.values()]
    write_csv(out_dir / "qc_runs.csv", RUN_COLUMNS, runs)

    # Protocol summary per scanner
    labels = [r.label for r in SCANNERS]
    proto_rows = []
    proto_md = {}
    for kind, fields in PROTOCOL_FIELDS.items():
        table = []
        for f in fields:
            cells = []
            for lab in labels:
                vals = [s.runs[kind].get(f, "") for s in sessions if s.label == lab and kind in s.runs]
                if kind == "t1w" and f == "dims":
                    text, missing = dims_range(vals), sum(1 for v in vals if not v)
                else:
                    text, missing = summarise_values(vals, f in CONTINUOUS_SUMMARY)
                cells.append(text)
                proto_rows.append({"scanner_label": lab, "run": kind, "field": f, "n_sessions": len(vals),
                                   "n_missing": missing, "summary": text})
            table.append([f] + cells)
        n_row = ["n_sessions"] + [str(sum(1 for s in sessions if s.label == lab and kind in s.runs)) for lab in labels]
        proto_md[kind] = md_table(["field"] + labels, [n_row] + table)
    write_csv(out_dir / "qc_protocol_by_scanner.csv",
              ("scanner_label", "run", "field", "n_sessions", "n_missing", "summary"), proto_rows)

    report = render_report(plan, sessions, provenance, download_summary, proto_md)
    (out_dir / "qc_report.md").write_text(report, encoding="utf-8")
    log.info("Wrote %s (%d sessions), qc_runs.csv, qc_protocol_by_scanner.csv, qc_report.md",
             manifest, len(sessions))


def render_report(plan: Plan, sessions: list[Session], prov: dict, dl: dict, proto_md: dict) -> str:
    labels = [r.label for r in SCANNERS]
    checks = []

    def check(name, expected, observed, ok):
        checks.append([name, expected, observed, "PASS" if ok else "FAIL"])

    subjects = plan.subjects
    check("Phase B subjects", EXPECTED_N_SUBJECTS, len(subjects), len(subjects) == EXPECTED_N_SUBJECTS)
    cover = defaultdict(Counter)
    for s in sessions:
        cover[s.subject][s.label or "UNASSIGNED"] += 1
    observed_labels = sorted({s.label for s in sessions if s.label})
    check("Scanners observed", ", ".join(labels), ", ".join(observed_labels), observed_labels == sorted(labels))
    incomplete = [sub for sub in subjects if any(cover[sub][lab] < 1 for lab in labels)]
    check("Every subject scanned on all 6 scanners", "all", f"{len(subjects) - len(incomplete)}/{len(subjects)}"
          + (f" (incomplete: {', '.join(incomplete)})" if incomplete else ""), not incomplete)
    repeat = {sub: [lab for lab in labels if cover[sub][lab] > 1] for sub in subjects}
    repeat = {k: v for k, v in repeat.items() if v}
    check("Subjects with within-scanner repeats", EXPECTED_N_REPEAT_SUBJECTS, len(repeat),
          len(repeat) == EXPECTED_N_REPEAT_SUBJECTS)
    bad = {sub: v for sub, v in repeat.items() if len(v) != 1 or cover[sub][v[0]] != EXPECTED_N_REPEATS}
    check(f"Each repeat subject: exactly 1 repeat scanner with {EXPECTED_N_REPEATS} sessions", "5/5",
          f"{len(repeat) - len(bad)}/{len(repeat)}" + (f" (not: {bad})" if bad else ""), not bad and bool(repeat))
    on_achieva = [sub for sub, v in repeat.items() if NO_REPEAT_SCANNER in v]
    check("No within-scanner repeats on the Achieva", "none", ", ".join(on_achieva) or "none", not on_achieva)
    rep_scanners = [lab for v in repeat.values() for lab in v]
    check("Repeat scanners distinct across subjects", "distinct",
          ", ".join(sorted(rep_scanners)), len(rep_scanners) == len(set(rep_scanners)))
    check("Total Phase B sessions", EXPECTED_N_SESSIONS, len(sessions), len(sessions) == EXPECTED_N_SESSIONS)
    unassigned = [f"{s.subject}/{s.session}" for s in sessions if not s.label]
    check("All sessions assigned to exactly one scanner from metadata", 0, len(unassigned), not unassigned)
    serials = defaultdict(set)
    for s in sessions:
        if s.label:
            serials[s.label].add(s.identity.get("device_serial_number", ""))
    multi = {k: sorted(v) for k, v in serials.items() if len(v) != 1}
    all_serials = [next(iter(v)) for v in serials.values() if len(v) == 1]
    check("One DeviceSerialNumber per scanner, distinct across scanners", "yes",
          "yes" if not multi and len(all_serials) == len(set(all_serials)) else f"no {multi}",
          not multi and len(all_serials) == len(set(all_serials)))
    mismatch = [f"{s.subject}/{s.session}" for s in sessions if "scanner_differs_from_session_code" in s.flags]
    check("Metadata scanner agrees with session-name code", 0, len(mismatch), not mismatch)
    required = ("file_not_downloaded", "missing_t1w", "missing_t1w_json", "missing_dwi", "missing_dwi_json",
                "missing_dwi_bval", "missing_dwi_bvec", "missing_revpe", "missing_revpe_json")
    missing = [s for s in sessions if s.flags & set(required)]
    check("Sessions with T1w+json, DWI+bval+bvec+json, reverse-PE+json", len(sessions),
          len(sessions) - len(missing), not missing)
    check("Selected files downloaded and verified", dl.get("n_selected", "?"), dl.get("n_ok", "?"),
          dl.get("n_failed", 1) == 0)

    L = []
    L.append("# ON-Harmony Phase B - metadata-only QC report\n")
    L.append(f"Generated {utcnow()} by `pipeline/00_download.py` v{SCRIPT_VERSION} "
             f"(sha256 `{prov.get('script_sha256', '')[:12]}`).\n")
    L.append(f"Dataset {DATASET_ID} snapshot **{prov.get('snapshot')}** (git `{prov.get('snapshot_hexsha')}`, "
             f"created {prov.get('snapshot_created')}); DOI {prov.get('dataset_doi', '')}.\n")
    L.append("**Blinding:** only JSON sidecars, bval/bvec text files and NIfTI headers (dimensions, voxel size, "
             "affine) were read. No voxel intensities were loaded, no image statistics computed, "
             "no images displayed.\n")

    L.append("## 1. Summary of checks\n")
    L.append(md_table(["check", "expected", "observed", "result"], checks) + "\n")

    L.append("## 2. Download integrity\n")
    L.append(f"- Files selected: {dl.get('n_selected')} ({dl.get('bytes_selected', 0) / 1e9:.2f} GB)\n"
             f"- Verified present: {dl.get('n_ok')}; failed: {dl.get('n_failed')}\n"
             f"- Verification methods: {dl.get('methods', {})}\n"
             f"- Phase A subjects not downloaded: {', '.join(plan.excluded_subjects) or 'none'}\n"
             f"- fmap/ EPI files not downloaded (not intended for dMRI): {dict(plan.fmap_excluded) or 'none'}\n")
    if dl.get("failed"):
        L.append("Failed files:\n\n" + "\n".join(f"- `{p}`: {e}" for p, e in dl["failed"]) + "\n")

    L.append("## 3. Subjects x scanners (number of sessions)\n")
    cols = labels + (["UNASSIGNED"] if unassigned else [])
    L.append(md_table(["subject"] + cols, [[sub] + [cover[sub][c] for c in cols] for sub in subjects]) + "\n")

    L.append("## 4. Scanner assignment from sidecar metadata\n")
    L.append("A session is assigned only if every sidecar (T1w, dMRI, reverse-PE) matches exactly one rule "
             "and all agree. Session names are used only as a cross-check.\n")
    L.append(md_table(["label", "Manufacturer", "ManufacturersModelName", "InstitutionName", "coil (ReceiveCoilName/CoilString)",
                       "nominal coil / site (documentation)"],
                      [[r.label, r.manufacturer, r.model, r.institution, r.coil or "(not recorded by vendor)",
                        f"{r.nominal_coil}; {r.site}"] for r in SCANNERS]) + "\n")
    ident_rows = []
    for lab in labels:
        members = [s for s in sessions if s.label == lab]
        for key in ("manufacturer", "model", "receive_coil", "institution", "station_name", "device_serial_number",
                    "software_versions"):
            vals = Counter(s.identity.get(key, "") for s in members)
            ident_rows.append([lab, key, "; ".join(f"{v or '(empty)'} (n={n})" for v, n in vals.most_common())])
    L.append("Observed identity values per scanner:\n\n" + md_table(["scanner", "field", "values"], ident_rows) + "\n")
    if unassigned:
        L.append("Sessions NOT assigned (flagged, not guessed): " + ", ".join(unassigned) + "\n")

    L.append("## 5. Within-scanner repeats\n")
    for sub, labs in sorted(repeat.items()):
        for lab in labs:
            members = sorted((s for s in sessions if s.subject == sub and s.label == lab), key=lambda s: s.session)
            idents = {tuple(s.identity.get(k, "") for k in ("manufacturer", "model", "receive_coil", "institution",
                                                             "station_name", "device_serial_number"))
                      for s in members}
            first = members[0]
            L.append(f"### {sub} - {lab}\n")
            L.append(md_table(["session", "repeat_index", "model", "coil", "institution", "station", "serial",
                               "software"],
                              [[s.session, s.repeat_index, s.identity.get("model"), s.identity.get("receive_coil")
                                or "(n/a)", s.identity.get("institution"), s.identity.get("station_name"),
                                s.identity.get("device_serial_number"), s.identity.get("software_versions")]
                               for s in members]) + "\n")
            same = "identical" if len(idents) == 1 else "NOT identical"
            L.append(f"- Scanner identity (model, coil, institution, station, serial) across all "
                     f"{len(members)} sessions: **{same}**.\n")
            L.append(f"- {first.session} (numbered like this subject's sessions on the other scanners) was assigned "
                     f"from its own sidecars to {lab}, like the other sessions.\n")
            if len(members) == EXPECTED_N_REPEATS:
                L.append(f"- {len(members)} sessions in total on this scanner: the travelling-heads session is "
                         f"**one of the {EXPECTED_N_REPEATS} repeats, not additional to them**.\n")
            else:
                L.append(f"- **{len(members)} sessions on this scanner (expected {EXPECTED_N_REPEATS})**: check "
                         f"whether the travelling-heads session is additional to the repeats.\n")

    L.append("## 6. Missing files\n")
    miss_rows = [[f"{s.subject}/{s.session}", ";".join(sorted(s.flags & set(required))),
                  "; ".join(n for n in s.notes if n.startswith("not downloaded"))] for s in missing]
    L.append((md_table(["session", "flags", "details"], miss_rows) if miss_rows else "None.") + "\n")

    L.append("## 7. Acquisition parameters per scanner (headers / sidecars / bval only)\n")
    L.append("Values: modal value, or median [min-max] for continuous fields; `other:` lists deviating values "
             "with counts. MB = MultibandAccelerationFactor (Siemens/GE) or ParallelReductionOutOfPlane (Philips); "
             "PI = ParallelReductionFactorInPlane; PF = partial-Fourier fields as recorded (GE records none). "
             "`shells` = b-value:count, b rounded to the nearest nominal shell (0/1000/2000) within 100 s/mm^2. "
             "PED = BIDS PhaseEncodingDirection.\n")
    for kind, title in (("dwi", "Main dMRI run"), ("revpe", "Reverse phase-encode run"), ("t1w", "T1w")):
        L.append(f"### {title}\n\n{proto_md[kind]}\n")
    ndw = Counter(str(s.runs["dwi"].get("n_dw")) for s in sessions if "dwi" in s.runs)
    L.append("### Comparison with the expected protocol\n")
    L.append(f"- Expected: 2 mm isotropic, b = 0/1000/2000 s/mm^2, 105 directions, reverse-PE b0.\n"
             f"- Observed number of diffusion-weighted (b>0) volumes in the main run: {dict(ndw)}; "
             f"b0 and total volume counts per scanner are in the table above. If '105' refers to the total "
             f"volume count it matches only scanners with 5 b0 volumes.\n")

    L.append("## 8. Flags per session\n")
    flag_rows = [[f"{s.subject}/{s.session}", s.label or "(unassigned)", ";".join(sorted(s.flags))]
                 for s in sessions if s.flags]
    L.append((md_table(["session", "scanner", "flags"], flag_rows) if flag_rows else "No flags.") + "\n")
    counts = Counter(f for s in sessions for f in s.flags)
    L.append("Flag counts: " + (", ".join(f"`{k}` x{v}" for k, v in sorted(counts.items())) or "none") + "\n")
    L.append("\n### Flag glossary\n")
    L.append(md_table(["flag", "meaning"], [[k, v] for k, v in FLAG_DESCRIPTIONS.items()]) + "\n")
    L.append("Note: for Philips data, dcm2niix cannot read phase-encode polarity from DICOM, so the sign of "
             "PhaseEncodingDirection in Philips sidecars does not come from the scanner. Confirm polarity "
             "(e.g. with the dir- label and the acquisition protocol) before building TOPUP acqparams.\n")

    L.append("## 9. Deviations documented in the dataset README vs metadata\n")
    rows = []
    for (sub, ses), notes in sorted(README_NOTES.items()):
        s = next((x for x in sessions if x.subject == sub and x.session == ses), None)
        for code, text in notes:
            rows.append([f"{sub}/{ses}", text, readme_agreement(code, s)])
    L.append(md_table(["session", "README statement", "what the metadata shows"], rows) + "\n")

    L.append("## 10. Other notes per session\n")
    note_rows = [[f"{s.subject}/{s.session}", n] for s in sessions for n in s.notes
                 if not n.startswith("README:") and not n.startswith("not downloaded")]
    L.append((md_table(["session", "note"], note_rows) if note_rows else "None.") + "\n")
    return "\n".join(L)


def readme_agreement(code: str, s: Session | None) -> str:
    if s is None:
        return "session not present in the download"
    d, r, t1 = s.runs.get("dwi", {}), s.runs.get("revpe", {}), s.runs.get("t1w", {})
    if code == "readme_revpe_acquired_LR":
        return (f"reverse-PE PED={r.get('PED') or '?'} (main dMRI PED={d.get('PED') or '?'}), "
                f"file named dir-{r.get('dir_entity') or '?'}; "
                + ("consistent with README (PE along i = left-right)." if str(r.get("PED", "")).startswith("i")
                   else "NOT consistent with README."))
    if code == "readme_dwi_slice_2p4mm":
        return (f"dMRI voxel size {d.get('voxel_mm') or '?'} mm, SliceThickness {d.get('SliceThickness_mm') or '?'}; "
                + ("consistent with README." if "2.40" in str(d.get("voxel_mm", "")) else "check."))
    if code == "readme_no_revpe_acquired":
        if not r:
            return "no reverse-PE run present; consistent with README."
        img = r.get("_meta", {}).get("ImageType", "")
        return (f"CONTRADICTS README: a reverse-PE run is present ({r.get('nii')}; PED={r.get('PED')}, "
                f"nvols={r.get('nvols')}, ImageType={img}, SeriesDescription={r.get('SeriesDescription')}, "
                f"AcquisitionTime={r.get('_meta', {}).get('AcquisitionTime', '?')}). Whether it is a genuine "
                f"acquisition cannot be decided from metadata; resolve during visual QC.")
    if code == "readme_t1w_matrix_212x256x256":
        return (f"cannot be verified from the header: released T1w images are cropped per session "
                f"(dims {t1.get('dims') or '?'}). Header orientation {t1.get('orientation') or '?'}, "
                f"SeriesDescription '{t1.get('SeriesDescription') or '?'}'.")
    return ""


# --------------------------------------------------------------------------
# CLI, logging, provenance
# --------------------------------------------------------------------------
def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PATH_OPTIONS = ("bids_dir", "out_dir", "log_dir", "staging_dir")


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, help="JSON file with option values (keys = long option names); "
                                                 "command-line options override it")
    known, _ = pre.parse_known_args(argv)
    p = argparse.ArgumentParser(
        parents=[pre], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download the Phase B subset of ON-Harmony (OpenNeuro ds004712: T1w, dMRI, reverse-PE, "
                    "top-level BIDS files) and write a session manifest and a metadata-only QC report.")
    p.add_argument("--bids-dir", type=Path, help="BIDS root to download into (treated as read-only raw data)")
    p.add_argument("--out-dir", type=Path, help="Manifest, QC report, provenance, cache and state")
    p.add_argument("--log-dir", type=Path, help="Log directory (default: <out-dir>/logs)")
    p.add_argument("--staging-dir", type=Path, help="Partial downloads (default: <out-dir>/staging)")
    p.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="OpenNeuro snapshot tag to pin")
    p.add_argument("--phase", default="B", help="Value of the participants.tsv 'phase' column to select")
    p.add_argument("--subjects", nargs="+", help="Restrict to these subjects (testing only)")
    p.add_argument("--jobs", type=int, default=4, help="Parallel downloads")
    p.add_argument("--retries", type=int, default=5, help="Attempts per file")
    p.add_argument("--timeout", type=int, default=300, help="Read timeout per request (s)")
    p.add_argument("--dry-run", action="store_true", help="List what would be downloaded and the space needed; "
                                                          "only small text metadata is fetched (into the cache)")
    p.add_argument("--skip-download", action="store_true", help="Only rebuild manifest/QC from local files")
    p.add_argument("--reverify", action="store_true", help="Re-hash local files even if verified before")
    p.add_argument("--refresh-listing", action="store_true", help="Re-fetch the cached snapshot listing")
    p.add_argument("--no-readonly", action="store_true", help="Do not chmod downloaded files to read-only")
    p.add_argument("--ignore-disk-space", action="store_true", help="Proceed even if free space looks too small")
    p.add_argument("--allow-inside-repo", action="store_true",
                   help="Allow output paths inside the code repository (must then be gitignored)")
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
    if not args.bids_dir or not args.out_dir:
        p.error("--bids-dir and --out-dir are required (on the command line or in --config)")
    args.bids_dir = args.bids_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.log_dir = (args.log_dir or args.out_dir / "logs").expanduser().resolve()
    args.staging_dir = (args.staging_dir or args.out_dir / "staging").expanduser().resolve()
    if args.bids_dir == args.out_dir or args.out_dir.is_relative_to(args.bids_dir) \
            or args.log_dir.is_relative_to(args.bids_dir) or args.staging_dir.is_relative_to(args.bids_dir):
        p.error("--out-dir, --log-dir and --staging-dir must be outside --bids-dir (raw data stays clean)")
    inside = [str(getattr(args, k)) for k in PATH_OPTIONS if getattr(args, k).is_relative_to(REPO_ROOT)]
    if inside and not args.allow_inside_repo:
        p.error(f"output paths inside the code repository {REPO_ROOT}: {inside}. "
                "Use paths outside it, or --allow-inside-repo with gitignored paths.")
    return args


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"00_download_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)
    return path


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def main(argv=None) -> int:
    args = parse_args(argv)
    for d in (args.out_dir, args.staging_dir, args.out_dir / "cache", args.out_dir / "state"):
        d.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(args.log_dir)
    script = Path(__file__).resolve()
    prov = {
        "script": str(script), "script_version": SCRIPT_VERSION, "script_sha256": sha256_of(script),
        "command": [sys.executable] + sys.argv, "cwd": os.getcwd(), "started_utc": utcnow(),
        "python": sys.version.split()[0], "requests": requests.__version__, "nibabel": nib.__version__,
        "platform": platform.platform(), "dataset": DATASET_ID, "snapshot": args.snapshot,
        "args": {k: str(v) for k, v in vars(args).items()}, "log_file": str(log_path),
    }
    log.info("00_download.py v%s (sha256 %s)", SCRIPT_VERSION, prov["script_sha256"])
    log.info("Command: %s", " ".join(prov["command"]))
    log.info("Python %s | requests %s | nibabel %s | %s", prov["python"], prov["requests"], prov["nibabel"],
             prov["platform"])
    log.info("BIDS dir: %s | out dir: %s | log: %s", args.bids_dir, args.out_dir, log_path)

    try:
        listing = get_listing(args.snapshot, args.out_dir / "cache", args.refresh_listing)
        prov.update(snapshot_hexsha=listing["hexsha"], snapshot_created=listing.get("created"),
                    openneuro_api_version=listing.get("openneuro_api_version"))
        latest = latest_snapshot_tag()
        prov["latest_snapshot_at_run"] = latest
        if latest and latest != args.snapshot:
            log.warning("Pinned snapshot %s is not the latest (%s); continuing with %s", args.snapshot, latest,
                        args.snapshot)
        files = listing["files"]
        by_path = {rf.path: rf for rf in files}
        dl = Downloader(args.staging_dir, args.retries, args.timeout, readonly=not args.no_readonly)
        cache = args.out_dir / "cache"
        participants = read_tsv(fetch_cached(by_path["participants.tsv"], cache, dl))
        desc = json.loads(fetch_cached(by_path["dataset_description.json"], cache, dl).read_text())
        prov["dataset_doi"] = desc.get("DatasetDOI", "")
        log_structure(files, participants)
        plan = build_plan(files, participants, args.phase, args.subjects, cache, dl)
    except Exception:
        log.exception("Planning failed")
        return 2

    state = VerifiedState(args.out_dir / "state" / "verified.json")
    total = sum(rf.size for rf in plan.selected)
    present = [rf for rf in plan.selected if (args.bids_dir / rf.path).is_file()
               and (args.bids_dir / rf.path).stat().st_size == rf.size]
    partial = sum((args.staging_dir / (rf.path + ".part")).stat().st_size for rf in plan.selected
                  if (args.staging_dir / (rf.path + ".part")).exists())
    remaining = total - sum(rf.size for rf in present) - partial
    free = free_bytes(args.bids_dir)
    n_sessions = sum(len(v) for v in plan.sessions.values())
    log.info("Selection: phase %s, %d subjects, %d sessions, %d files, %.2f GB",
             args.phase, len(plan.subjects), n_sessions, len(plan.selected), total / 1e9)
    kinds = Counter()
    for rf in plan.selected:
        parts = rf.path.split("/")
        kinds[parts[2] if len(parts) > 3 else "top-level"] += rf.size
    log.info("Size by datatype: %s", {k: f"{v / 1e9:.2f} GB" for k, v in sorted(kinds.items())})
    log.info("Already present (size matches): %d files; still to download: %.2f GB; free at target: %.1f GB",
             len(present), remaining / 1e9, free / 1e9)
    plan_path = args.out_dir / ("download_plan_dryrun.tsv" if args.dry_run else "download_plan.tsv")
    with open(plan_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "size_bytes", "checksum_algo", "checksum", "annexed", "present_locally", "url"])
        present_set = {rf.path for rf in present}
        for rf in plan.selected:
            algo, digest = rf.checksum() or ("size-only", "")
            w.writerow([rf.path, rf.size, algo, digest, rf.annexed, rf.path in present_set,
                        rf.urls[0] if rf.urls else ""])
    log.info("Download plan written to %s", plan_path)

    if args.dry_run:
        for rf in plan.selected:
            status = "present" if rf.path in present_set else "to download"
            print(f"{status:12s} {rf.size / 1e6:10.2f} MB  {rf.path}")
        log.info("Dry run: nothing written to %s", args.bids_dir)
        return 0

    if remaining * 1.05 > free and not args.ignore_disk_space:
        log.error("Not enough free space: need ~%.2f GB, have %.2f GB (use --ignore-disk-space to override)",
                  remaining * 1.05 / 1e9, free / 1e9)
        return 3

    summary = {"n_selected": len(plan.selected), "bytes_selected": total, "failed": [], "methods": Counter()}
    if not args.skip_download:
        args.bids_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading/verifying %d files with %d workers", len(plan.selected), args.jobs)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.jobs, thread_name_prefix="dl") as ex:
            futures = {ex.submit(ensure_file, rf, args.bids_dir, dl, state, args.reverify): rf
                       for rf in plan.selected}
            for i, fut in enumerate(as_completed(futures), 1):
                rf = futures[fut]
                try:
                    status = fut.result()
                    log.log(logging.INFO if status.startswith("downloaded") else logging.DEBUG,
                            "[%d/%d] %s: %s", i, len(futures), rf.path, status)
                except Exception as exc:  # noqa: BLE001 - collected and reported
                    summary["failed"].append((rf.path, str(exc)))
                    log.error("[%d/%d] %s: %s", i, len(futures), rf.path, exc)
        log.info("Download phase finished in %.0f s", time.time() - t0)
    else:
        log.info("--skip-download: using local files as they are")
    for rf in plan.selected:
        entry = state.data.get(rf.path)
        dest = args.bids_dir / rf.path
        if dest.exists() and entry and state.is_current(rf, dest):
            summary["methods"][entry["method"]] += 1
    summary["n_ok"] = sum(summary["methods"].values())
    summary["n_failed"] = len(plan.selected) - summary["n_ok"]
    if args.skip_download and summary["n_failed"]:
        log.warning("%d selected files are missing or not verified (run without --skip-download)",
                    summary["n_failed"])
    summary["methods"] = dict(summary["methods"])

    prov.update(finished_utc=utcnow(), n_files_selected=len(plan.selected), bytes_selected=total,
                n_files_verified=summary["n_ok"], n_files_failed=summary["n_failed"],
                subjects=plan.subjects, bids_dir=str(args.bids_dir))
    try:
        build_outputs(plan, participants, args.bids_dir, args.out_dir, prov, summary)
    except Exception:
        log.exception("Building manifest/QC failed")
        return 2
    (args.out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    log.info("Provenance written to %s", args.out_dir / "provenance.json")
    if summary["n_failed"]:
        log.error("%d files failed or are unverified; re-run to retry", summary["n_failed"])
        return 1
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
