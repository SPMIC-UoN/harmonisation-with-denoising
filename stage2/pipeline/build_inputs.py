#!/usr/bin/env python3
"""
build_inputs.py - Build the confirmatory model input tables (H1-H4) from the template-space fits
(Stage 1 Sections 3.3-3.4).

Reads per-session `*_valid` maps (NaN where missing), template WM checkerboards / ROI label images (07) and, for
H3a, the MNI maps (08). Writes long-format tables to <out>/ and every structural count to <out>/counts.json.
BLINDING: prints and stores counts only; no values, means or arm comparisons are printed or summarised here.

  metrics  FA, MD = dti_b0b1000_{FA,MD}_valid ; MK = dki_allshells_kurt_valid   (06_fit.py: missing in both arms
           where, in either arm, FA <= 0 or MD <= 0); MK missing in both arms where, in either arm, MK <= 0 or MK > 3
  H1a/b    'all' template; per subject: repeats averaged per scanner, CoV across the 6 scanners per arm
           (voxel: WM sparse checkerboard, 1 in 8; ROI: means over ROI voxels valid in all sessions); z pooled over subjects,
           units and both arms per metric
  H2a/b    'all' template of each repeat subject; Delta Bias = Bias_den - Bias_nonden; / SD
  H3a      MNI WM sparse checkerboard, *_MNI2mm_valid; H3b: 'all'-template ROI means; repeats averaged;
           y - mu_r per arm, both arms / one pooled SD; one table per metric with both arms
  H4a/b    repeat template; CoV across the 6 repeats per arm; Delta = z_den - z_non (z pooled over both arms);
           off_hz = mean |Hz| over repeats and arms (registered model); also off_z (off_hz z-scored within pair)
           for the exploratory H4 fits
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

import inputs as I

SCRIPT_VERSION = "1.0.0"
ARMS = I.ARMS
METRICS = {"FA": ("dti_b0b1000_FA_valid", "missing_metric_FA"),
           "MD": ("dti_b0b1000_MD_valid", "missing_metric_MD"),
           "MK": ("dki_allshells_kurt_valid", "missing_metric_MK")}
MNI_FILES = {"FA": "FA_MNI2mm_valid", "MD": "MD_MNI2mm_valid", "MK": "MK_MNI2mm_valid"}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Data:
    def __init__(self, deriv: Path, manifest: Path):
        self.deriv = deriv
        self.pattern = "lattice"                                # voxel sampling (--checkerboard)
        rows = list(csv.DictReader(open(manifest)))
        self.sessions = defaultdict(list)                       # subject -> [(session, scanner)]
        for r in rows:
            self.sessions[r["subject"]].append((r["session"], r["scanner_label"]))
        for s in self.sessions:
            self.sessions[s].sort()
        self.repeat_scanner = {}
        for s, ses in self.sessions.items():
            n = defaultdict(int)
            for _, sc in ses:
                n[sc] += 1
            reps = [sc for sc, k in n.items() if k > 1]
            if len(reps) > 1:
                raise RuntimeError(f"{s}: more than one repeat scanner {reps}")
            if reps:
                if n[reps[0]] != 6:
                    raise RuntimeError(f"{s}: {n[reps[0]]} repeats on {reps[0]}, expected 6")
                self.repeat_scanner[s] = reps[0]
        self._ref = {}

    def img(self, p: Path, ref_key=None) -> np.ndarray:
        im = nib.load(str(p))
        if ref_key is not None:                               # all images of one space share shape and affine
            ref = self._ref.setdefault(ref_key, (im.shape[:3], im.affine))
            if im.shape[:3] != ref[0] or not np.allclose(im.affine, ref[1], atol=1e-4):
                raise RuntimeError(f"{p}: grid differs from the reference grid of {ref_key}")
        return np.asarray(im.dataobj, dtype=np.float64)

    def template(self, sub: str, kind: str) -> Path:
        return self.deriv / "templates" / sub / ("all" if kind == "all" else f"repeat-{self.repeat_scanner[sub]}")

    def fit(self, arm: str, sub: str, ses: str, kind: str, name: str) -> Path:
        return self.deriv / "fits" / arm / f"{sub}_{ses}" / kind / f"{name}.nii.gz"

    def stack(self, sub, sessions, kind, name, sel, ref_key):
        """(n_sessions, n_sel) per arm -> dict arm -> array."""
        return {arm: np.stack([self.img(self.fit(arm, sub, ses, kind, name), ref_key)[sel] for ses in sessions])
                for arm in ARMS}

    def missing(self, sub, sessions, kind, mname, sel, ref_key):
        return np.stack([self.img(self.deriv / "fits" / "_missing" / f"{sub}_{ses}" / kind / f"{mname}.nii.gz",
                                  ref_key)[sel] > 0 for ses in sessions])


def check_nan_pattern(vals: dict, miss: np.ndarray, where: str, counts: dict) -> np.ndarray:
    """Structural check: NaNs identical in both arms and equal to the _missing masks. Returns keep."""
    nan_a, nan_b = ~np.isfinite(vals[ARMS[0]]), ~np.isfinite(vals[ARMS[1]])
    counts.setdefault("d41_checks", {})[where] = dict(
        nan_arms_differ=int((nan_a != nan_b).sum()), nan_vs_missing_mask_differ=int((nan_a != miss).sum()),
        missing_session_units=int(miss.sum()))
    if (nan_a != nan_b).any() or (nan_a != miss).any():
        raise RuntimeError(f"{where}: NaN pattern differs between arms or from the missing masks")
    return ~miss.any(axis=0)


def write_table(path: Path, header: list, cols: list) -> int:
    n = len(cols[0])
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for i in range(n):
            w.writerow([c[i] if not isinstance(c[i], (float, np.floating)) else repr(float(c[i])) for c in cols])
    return n


# --------------------------------------------------------------------------------------------- template level
def subject_units(D: Data, sub: str, kind: str, sessions: list, metric: str, counts: dict, tag: str):
    """Lattice and ROI values for one subject/template/metric. Returns dict with per-arm session arrays."""
    tdir = D.template(sub, kind)
    rk = f"{sub}/{kind}"
    wm = D.img(tdir / "T_wm_2mm.nii.gz", rk) > 0
    lat = D.img(tdir / f"T_wm_checkerboard-{D.pattern}_2mm.nii.gz", rk) > 0
    lab = D.img(tdir / "T_xtract_rois_2mm.nii.gz", rk).astype(int)
    if (lat & ~wm).any() or ((lab > 0) & ~wm).any():
        raise RuntimeError(f"{tdir}: checkerboard/ROI voxels outside the WM mask")
    name, mname = METRICS[metric]
    fkind = "all" if kind == "all" else "repeat"
    ses = [s for s, _ in sessions]
    vals = D.stack(sub, ses, fkind, name, wm, rk)                 # all WM voxels (ROIs use WM voxels)
    miss = D.missing(sub, ses, fkind, mname, wm, rk)
    keep = check_nan_pattern(vals, miss, f"{tag}/{sub}/{kind}/{metric}", counts)
    keep_d41 = keep
    lat_w = lat[wm]
    lab_w = lab[wm]
    c = counts.setdefault("units", {}).setdefault(f"{tag}/{sub}/{kind}/{metric}", {})
    if metric == "MK":                                            # 0 < MK <= 3 in both arms, every session
        a_, b_, ta, tb = I.range_validity(vals[ARMS[0]], vals[ARMS[1]])
        vals = {ARMS[0]: a_, ARMS[1]: b_}
        ta_u, tb_u = ta.any(axis=0) & keep_d41, tb.any(axis=0) & keep_d41
        keep = keep_d41 & ~(ta_u | tb_u)
        rids = roi_ids(tdir / "T_xtract_rois_counts.csv")
        in_roi = lab_w > 0
        n_d41 = np.bincount(lab_w[keep_d41], minlength=lab_w.max() + 1)
        n_d56 = np.bincount(lab_w[keep], minlength=lab_w.max() + 1)
        c["d56"] = dict(checkerboard_excluded=I.trigger_split(ta_u[lat_w], tb_u[lat_w]),
                        roi_voxels_excluded=I.trigger_split(ta_u[in_roi], tb_u[in_roi]),
                        rois_emptied=int(sum(n_d41[k] > 0 and n_d56[k] == 0 for k in rids if k < n_d41.size)))
    c.update(n_sessions=len(ses), wm_voxels=int(wm.sum()), checkerboard_voxels=int(lat_w.sum()),
             checkerboard_dropped_d41=int((lat_w & ~keep_d41).sum()), checkerboard_kept=int((lat_w & keep).sum()))
    return dict(vals=vals, keep=keep, lat=lat_w, lab=lab_w, flat_idx=np.flatnonzero(wm), wm=wm)


def roi_ids(atlas_counts: Path) -> np.ndarray:
    return np.array([int(r["label"]) for r in csv.DictReader(open(atlas_counts))])


# --------------------------------------------------------------------------------------------- H1
def build_h1(D: Data, out: Path, counts: dict):
    subs = sorted(D.sessions)
    for metric in METRICS:
        rows_a, rows_b = defaultdict(list), defaultdict(list)
        for sub in subs:
            sess = D.sessions[sub]
            u = subject_units(D, sub, "all", sess, metric, counts, "H1")
            scanners = [sc for _, sc in sess]
            rids = roi_ids(D.template(sub, "all") / "T_xtract_rois_counts.csv")
            cov_a, cov_b = {}, {}
            for arm in ARMS:
                v = u["vals"][arm]
                per_sc, labs = I.average_by_group(v[:, u["lat"] & u["keep"]], scanners)     #  / Section 3.3
                if len(labs) != 6:
                    raise RuntimeError(f"{sub}: {len(labs)} scanners")
                cov_a[arm] = I.cov(per_sc)
                rm, n_kept, n_drop = I.roi_means(v, u["lab"], u["keep"], rids)
                cov_b[arm] = I.cov(I.average_by_group(rm, scanners)[0])
            ca = I.missing_in_either_arm(cov_a[ARMS[0]], cov_a[ARMS[1]])
            cb = I.missing_in_either_arm(cov_b[ARMS[0]], cov_b[ARMS[1]])
            vox = u["flat_idx"][u["lat"] & u["keep"]]
            c = counts["units"][f"H1/{sub}/all/{metric}"]
            c.update(cov_nonpos_mean_voxels=int(np.isnan(ca[0]).sum()), rois=int(rids.size),
                     rois_empty_or_dropped=int((n_kept == 0).sum()),
                     rois_cov_undefined=int(np.isnan(cb[0]).sum() - (n_kept == 0).sum()),
                     roi_voxels_dropped_d41=int(n_drop.sum()))
            for j, arm in enumerate(ARMS):
                ok = np.isfinite(ca[j])
                rows_a["subject"] += [sub] * int(ok.sum()); rows_a["unit"] += vox[ok].tolist()
                rows_a["arm"] += [j] * int(ok.sum()); rows_a["cov"] += ca[j][ok].tolist()
                ok = np.isfinite(cb[j])
                rows_b["subject"] += [sub] * int(ok.sum()); rows_b["unit"] += rids[ok].tolist()
                rows_b["arm"] += [j] * int(ok.sum()); rows_b["cov"] += cb[j][ok].tolist()
        for lvl, rows in (("a", rows_a), ("b", rows_b)):
            cv = np.array(rows["cov"])
            arm = np.array(rows["arm"])
            z_non, z_den = I.zscore_pooled(cv[arm == 0], cv[arm == 1])      # one mean/SD over both arms
            z = np.empty_like(cv); z[arm == 0] = z_non; z[arm == 1] = z_den
            n = write_table(out / f"h1{lvl}_{metric}.csv.gz", ["subject", "unit", "den", "cov", "y"],
                            [rows["subject"], rows["unit"], rows["arm"], cv, z])
            counts.setdefault("tables", {})[f"h1{lvl}_{metric}"] = dict(
                rows=n, rows_per_arm=[int((arm == 0).sum()), int((arm == 1).sum())],
                subjects=len(set(rows["subject"])))
        print(f"{utcnow()} H1 {metric}: tables written", flush=True)


# --------------------------------------------------------------------------------------------- H2
def build_h2(D: Data, out: Path, counts: dict):
    for sub, rsc in sorted(D.repeat_scanner.items()):
        sess = D.sessions[sub]
        is_rep = np.array([sc == rsc for _, sc in sess])
        rids = roi_ids(D.template(sub, "all") / "T_xtract_rois_counts.csv")
        for metric in METRICS:
            u = subject_units(D, sub, "all", sess, metric, counts, "H2")
            res = {}
            for lvl in ("a", "b"):
                if lvl == "a":
                    v = {arm: u["vals"][arm][:, u["lat"] & u["keep"]] for arm in ARMS}
                    units = u["flat_idx"][u["lat"] & u["keep"]]
                else:
                    v = {arm: I.roi_means(u["vals"][arm], u["lab"], u["keep"], rids)[0] for arm in ARMS}
                    units = rids
                db = I.delta_bias(v[ARMS[0]][is_rep], v[ARMS[0]][~is_rep], v[ARMS[1]][is_rep], v[ARMS[1]][~is_rep])
                ok = np.isfinite(db)
                y, sd = I.scale_by_sd_uncentred(db[ok])
                n = write_table(out / f"h2{lvl}_{sub}_{metric}.csv.gz", ["unit", "delta_bias", "y"],
                                [units[ok].tolist(), db[ok], y])
                counts.setdefault("tables", {})[f"h2{lvl}_{sub}_{metric}"] = dict(
                    rows=n, repeat_scanner=rsc, n_within=int(is_rep.sum()), n_across=int((~is_rep).sum()),
                    units_missing=int((~ok).sum()))
        print(f"{utcnow()} H2 {sub}: tables written", flush=True)


# --------------------------------------------------------------------------------------------- H3
def h3_write(out: Path, name: str, rows: dict, counts: dict):
    yv = np.array(rows["y_raw"]); arm = np.array(rows["arm"])
    unit_codes = {k: i for i, k in enumerate(sorted(set(rows["unit"])))}
    r = np.array([unit_codes[k] for k in rows["unit"]])
    centred = I.h3_centre(yv, r, arm)                                   # mu_r per condition
    y, s = I.h3_shared_scale(centred)                                   # one constant for both arms
    n = write_table(out / f"{name}.csv.gz", ["subject", "scanner", "unit", "r", "den", "y_raw", "y"],
                    [rows["subject"], rows["scanner"], rows["unit"], r.tolist(), rows["arm"], yv, y])
    cells = defaultdict(int)
    for sub, sc, a in zip(rows["subject"], rows["scanner"], rows["arm"]):
        cells[(sub, sc, a)] += 1
    per_sub = defaultdict(set)
    for sub, sc, a in cells:
        per_sub[sub].add(sc)
    counts.setdefault("tables", {})[name] = dict(
        rows=n, units=len(unit_codes), rows_per_arm=[int((arm == 0).sum()), int((arm == 1).sum())],
        subjects=len(per_sub), scanners_per_subject=sorted({len(v) for v in per_sub.values()}),
        cells=len(cells), cell_rows_min=min(cells.values()), cell_rows_max=max(cells.values()))


def build_h3(D: Data, out: Path, counts: dict, levels=("a", "b")):
    """H3 uses one set of units for ALL subjects : mu_r is the mean over all sessions of
    all subjects, so a voxel (H3a) is used only if valid in every session of every subject, and an ROI (H3b) only if
    it has kept voxels in every subject. Two passes: per-subject values, then the common unit set."""
    subs = sorted(D.sessions)
    for metric in METRICS:
        per = {lvl: {} for lvl in levels}                   # lvl -> sub -> (per-arm (n_sc, n_units), scanners, keep)
        trig_a = {}                                         # H3a MK: sub -> arm -> unit-level MK-range trigger (any session)
        for sub in subs:
            sess = D.sessions[sub]
            ses = [s for s, _ in sess]
            scanners = [sc for _, sc in sess]
            if "a" in levels:
                lat = D.img(D.deriv / "mmorf" / f"MNI152_2mm_wm_checkerboard-{D.pattern}.nii.gz", "MNI") > 0
                vals = {arm: np.stack([D.img(D.deriv / "mmorf" / arm / f"{sub}_{s}" / f"{MNI_FILES[metric]}.nii.gz",
                                             "MNI")[lat] for s in ses]) for arm in ARMS}
                nan_a, nan_b = ~np.isfinite(vals[ARMS[0]]), ~np.isfinite(vals[ARMS[1]])
                if (nan_a != nan_b).any():
                    raise RuntimeError(f"H3a {sub} {metric}: NaN pattern differs between arms")
                keep = ~nan_a.any(axis=0)
                counts.setdefault("units", {})[f"H3a/{sub}/{metric}"] = dict(
                    checkerboard=int(lat.sum()), dropped_subject=int((~keep).sum()), kept_subject=int(keep.sum()))
                per["a"][sub] = ({arm: I.average_by_group(vals[arm], scanners)[0] for arm in ARMS},
                                 I.average_by_group(vals[ARMS[0]], scanners)[1], keep, np.flatnonzero(lat))
                if metric == "MK":                                # MK-range trigger maps (either arm's warp), counts only
                    tr = {x: np.zeros(int(lat.sum()), bool) for x in ARMS}
                    for x in ARMS:
                        for s in ses:
                            for arm in ARMS:
                                tr[x] |= D.img(D.deriv / "mmorf" / arm / f"{sub}_{s}" / f"trigMK_{x}_MNI2mm.nii.gz",
                                               "MNI")[lat] > 0
                    trig_a[sub] = tr
            if "b" in levels:
                u = subject_units(D, sub, "all", sess, metric, counts, "H3b")
                rids = roi_ids(D.template(sub, "all") / "T_xtract_rois_counts.csv")
                v, keep = {}, None
                for arm in ARMS:
                    rm, n_kept, _ = I.roi_means(u["vals"][arm], u["lab"], u["keep"], rids)
                    v[arm], labs = I.average_by_group(rm, scanners)
                    keep = n_kept > 0
                per["b"][sub] = (v, labs, keep, rids)
        for lvl in levels:
            ids = [per[lvl][s][3] for s in subs]
            if any(not np.array_equal(ids[0], x) for x in ids):
                raise RuntimeError(f"H3{lvl}: unit ids differ between subjects")
            common = np.logical_and.reduce([per[lvl][s][2] for s in subs])
            units = ids[0][common]
            rows = defaultdict(list)
            for sub in subs:
                v, labs, _, _ = per[lvl][sub]
                for j, arm in enumerate(ARMS):
                    for t, sc in enumerate(labs):
                        rows["subject"] += [sub] * units.size; rows["scanner"] += [sc] * units.size
                        rows["unit"] += units.tolist(); rows["arm"] += [j] * units.size
                        rows["y_raw"] += v[arm][t][common].tolist()
            cc = counts.setdefault("units", {})[f"H3{lvl}/common/{metric}"] = dict(
                candidates=int(ids[0].size), kept_all_subjects=int(common.sum()),
                dropped=int((~common).sum()))
            if metric == "MK" and lvl == "a":
                tn = np.logical_or.reduce([trig_a[s][ARMS[0]] for s in subs])
                td = np.logical_or.reduce([trig_a[s][ARMS[1]] for s in subs])
                cc["dropped_with_d56_trigger"] = I.trigger_split(tn & ~common, td & ~common)
                cc["dropped_without_d56_trigger"] = int((~common & ~tn & ~td).sum())
            h3_write(out, f"h3{lvl}_{metric}", rows, counts)
        print(f"{utcnow()} H3 {metric}: tables written", flush=True)


# --------------------------------------------------------------------------------------------- H4
def build_h4(D: Data, out: Path, counts: dict):
    for sub, rsc in sorted(D.repeat_scanner.items()):
        sess = [(s, sc) for s, sc in D.sessions[sub] if sc == rsc]
        ses = [s for s, _ in sess]
        rk = f"{sub}/repeat"
        tdir = D.template(sub, "repeat")
        wm = D.img(tdir / "T_wm_2mm.nii.gz", rk) > 0
        hz = {arm: np.stack([np.abs(D.img(D.fit(arm, sub, s, "repeat", "field_hz"), rk)[wm]) for s in ses])
              for arm in ARMS}
        off_vox = I.h4_offres(np.concatenate([hz[ARMS[0]], hz[ARMS[1]]]))      # mean |Hz| over repeats x arms
        if not np.isfinite(off_vox).all():
            raise RuntimeError(f"H4 {sub}: non-finite field values in WM")
        rids = roi_ids(tdir / "T_xtract_rois_counts.csv")
        for metric in METRICS:
            u = subject_units(D, sub, "repeat", sess, metric, counts, "H4")
            for lvl in ("a", "b"):
                if lvl == "a":
                    sel = u["lat"] & u["keep"]
                    c = {arm: I.cov(u["vals"][arm][:, sel]) for arm in ARMS}
                    off = off_vox[sel]
                    units = u["flat_idx"][sel]
                else:
                    c = {arm: I.cov(I.roi_means(u["vals"][arm], u["lab"], u["keep"], rids)[0]) for arm in ARMS}
                    off = I.roi_means(off_vox[None, :], u["lab"], u["keep"], rids)[0][0]
                    units = rids
                delta = I.h4_delta(c[ARMS[0]], c[ARMS[1]])                     # z pooled over both arms
                ok = np.isfinite(delta) & np.isfinite(off)
                off_z = I.zscore_pooled(off[ok])[0]
                n = write_table(out / f"h4{lvl}_{sub}_{metric}.csv.gz",
                                ["unit", "cov_nonden", "cov_den", "delta", "off_hz", "off_z"],
                                [units[ok].tolist(), c[ARMS[0]][ok], c[ARMS[1]][ok], delta[ok], off[ok], off_z])
                counts.setdefault("tables", {})[f"h4{lvl}_{sub}_{metric}"] = dict(
                    rows=n, repeat_scanner=rsc, n_repeats=len(ses), units_missing=int((~ok).sum()))
        print(f"{utcnow()} H4 {sub}: tables written", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build confirmatory model input tables (counts only are printed).")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True, help="ds004712_meta/sessions_manifest.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hyp", nargs="+", choices=("H1", "H2", "H3a", "H3b", "H4"), default=["H1", "H2", "H3a", "H3b", "H4"])
    p.add_argument("--checkerboard", choices=("lattice", "parity"), default="lattice",
                   help="voxel sampling: lattice = 1 in 8 white-matter voxels (main analysis); parity = 1 in 2 (Table S20)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    D = Data(a.deriv_dir, a.manifest)
    D.pattern = a.checkerboard
    counts_p = a.out / "counts.json"
    counts = json.loads(counts_p.read_text()) if counts_p.is_file() else {}
    counts["generated_by"] = (f"build_inputs.py v{SCRIPT_VERSION} sha256 "
                              f"{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} "
                              f"inputs.py sha256 {hashlib.sha256((Path(__file__).parent / 'inputs.py').read_bytes()).hexdigest()}")
    counts["repeat_pairs"] = D.repeat_scanner
    counts["n_sessions"] = sum(len(v) for v in D.sessions.values())
    print(f"{utcnow()} subjects {len(D.sessions)}, sessions {counts['n_sessions']}, repeat pairs {len(D.repeat_scanner)}")
    if "H1" in a.hyp: build_h1(D, a.out, counts)
    if "H2" in a.hyp: build_h2(D, a.out, counts)
    lv = tuple(x[-1] for x in a.hyp if x.startswith("H3"))
    if lv: build_h3(D, a.out, counts, lv)
    if "H4" in a.hyp: build_h4(D, a.out, counts)
    counts["finished"] = utcnow()
    counts_p.write_text(json.dumps(counts, indent=1))
    print(f"{utcnow()} done; counts in {counts_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
