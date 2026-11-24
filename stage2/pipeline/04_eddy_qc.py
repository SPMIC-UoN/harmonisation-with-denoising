#!/usr/bin/env python3
"""
04_eddy_qc.py - Per-session EDDY / eddy_quad QC summary for one BRC arm
(routine QC; motion is reported against reference datasets such as UK Biobank).

Reads, per session, only text/JSON QC outputs written by BRC 1.6.3:
  analysis/dMRI/preproc/qc/qc.json                      (eddy_quad)
  analysis/dMRI/preproc/eddy/eddy_unwarped_images.eddy_movement_rms
  analysis/dMRI/preproc/eddy/eddy_unwarped_images.eddy_restricted_movement_rms
  analysis/dMRI/preproc/eddy/eddy_unwarped_images.eddy_outlier_report
No images are read, and no diffusion model outputs (FA/MD) are touched.

Flags (QC screening only - not exclusion rules; exclusions follow the protocol):
  - robust z > 3.5 (median/MAD over all sessions of the arm) for mean absolute or
    relative motion, maximum relative motion, or total outlier percentage;
  - robust z < -3.5 within scanner for the b1000/b2000 CNR (CNR depends on scanner);
  - absolute: any volume-to-volume motion > 2 mm (one voxel) within the main run, or outliers > 5 %.
Relative-motion flags use the main run only: the jump into the appended reverse b0 (last volume, a
separate series) is reported separately (rel_jump_into_revpe_mm); EDDY's own qc_mot_rel includes it.

Outputs: <deriv>/qc/eddy_qc_<arm>.csv and eddy_qc_<arm>.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

SCRIPT_VERSION = "1.0.0"
Z_MOTION = 3.5
Z_CNR = -3.5
MAX_REL_MM = 2.0
MAX_OUTLIER_PCT = 5.0

COLUMNS = ("subject", "session", "scanner_label", "arm", "n_vols", "mot_abs_mean_mm", "mot_rel_mean_mm",
           "main_rel_mean_mm", "main_rel_max_mm", "rel_jump_into_revpe_mm", "mot_rel_max_mm", "mot_abs_max_mm", "restricted_rel_mean_mm", "n_vols_rel_gt_1mm",
           "outliers_total_pct", "outliers_b1000_pct", "outliers_b2000_pct", "n_outlier_slices",
           "snr_b0", "cnr_b1000", "cnr_b2000", "z_mot_abs", "z_mot_rel", "z_mot_rel_max", "z_outliers",
           "z_cnr_b1000_scanner", "z_cnr_b2000_scanner", "flags")


def robust_z(values: list[float]) -> list[float | None]:
    vals = [v for v in values if v is not None]
    if len(vals) < 5:
        return [None] * len(values)
    med = statistics.median(vals)
    mad = statistics.median(abs(v - med) for v in vals) * 1.4826
    return [None if v is None or mad == 0 else (v - med) / mad for v in values]


def read_rms(path: Path) -> list[tuple[float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    return rows


def session_metrics(row: dict, arm: str, brc_root: Path) -> dict:
    base = brc_root / arm / row["brc_subject"] / "analysis/dMRI/preproc"
    rec = {"subject": row["subject"], "session": row["session"], "scanner_label": row["scanner_label"], "arm": arm}
    qc = json.loads((base / "qc/qc.json").read_text())
    rms = read_rms(base / "eddy/eddy_unwarped_images.eddy_movement_rms")
    rrms = read_rms(base / "eddy/eddy_unwarped_images.eddy_restricted_movement_rms")
    report = (base / "eddy/eddy_unwarped_images.eddy_outlier_report").read_text().splitlines()
    shells = qc.get("data_unique_bvals", [])
    ob = dict(zip(shells, qc.get("qc_outliers_b", [])))
    cnr = qc.get("qc_cnr_avg", [])
    rec.update(
        n_vols=len(rms),
        mot_abs_mean_mm=qc.get("qc_mot_abs"), mot_rel_mean_mm=qc.get("qc_mot_rel"),
        mot_rel_max_mm=max(r for _, r in rms[1:]) if len(rms) > 1 else None,
        # The reverse b0 is the last volume and a separate series: its jump is repositioning, not
        # within-scan motion. main_* exclude it; EDDY's qc_mot_* include it.
        main_rel_mean_mm=statistics.mean(r for _, r in rms[1:-1]) if len(rms) > 2 else None,
        main_rel_max_mm=max(r for _, r in rms[1:-1]) if len(rms) > 2 else None,
        rel_jump_into_revpe_mm=rms[-1][1] if rms else None,
        mot_abs_max_mm=max(a for a, _ in rms) if rms else None,
        restricted_rel_mean_mm=statistics.mean(r for _, r in rrms[1:]) if len(rrms) > 1 else None,
        n_vols_rel_gt_1mm=sum(r > 1.0 for _, r in rms[1:]),
        outliers_total_pct=qc.get("qc_outliers_tot"),
        outliers_b1000_pct=next((v for b, v in ob.items() if 900 <= b <= 1100), None),
        outliers_b2000_pct=next((v for b, v in ob.items() if 1900 <= b <= 2100), None),
        n_outlier_slices=sum(1 for ln in report if ln.strip()),
        snr_b0=cnr[0] if len(cnr) > 0 else None, cnr_b1000=cnr[1] if len(cnr) > 1 else None,
        cnr_b2000=cnr[2] if len(cnr) > 2 else None)
    return rec


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="EDDY / eddy_quad QC summary per session for one BRC arm.")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--arm", choices=("nondenoised", "denoised"), required=True)
    a = p.parse_args(argv)
    rows = list(csv.DictReader(open(a.deriv_dir / "inputs" / "preproc_inputs.csv", newline="")))
    recs, missing = [], []
    for r in rows:
        try:
            recs.append(session_metrics(r, a.arm, a.deriv_dir / "brc"))
        except FileNotFoundError as exc:
            missing.append(f"{r['subject']}/{r['session']}: {exc.filename}")
    for key, col in (("z_mot_abs", "mot_abs_mean_mm"), ("z_mot_rel", "main_rel_mean_mm"),
                     ("z_mot_rel_max", "main_rel_max_mm"), ("z_outliers", "outliers_total_pct")):
        for rec, z in zip(recs, robust_z([rec[col] for rec in recs])):
            rec[key] = z
    for col, key in (("cnr_b1000", "z_cnr_b1000_scanner"), ("cnr_b2000", "z_cnr_b2000_scanner")):
        for lab in sorted({r["scanner_label"] for r in recs}):
            group = [r for r in recs if r["scanner_label"] == lab]
            for rec, z in zip(group, robust_z([g[col] for g in group])):
                rec[key] = z
    for rec in recs:
        f = []
        for key, name in (("z_mot_abs", "motion_abs"), ("z_mot_rel", "motion_rel"), ("z_mot_rel_max", "motion_rel_max"),
                          ("z_outliers", "outliers")):
            if rec.get(key) is not None and rec[key] > Z_MOTION:
                f.append(f"high_{name}")
        for key, name in (("z_cnr_b1000_scanner", "cnr_b1000"), ("z_cnr_b2000_scanner", "cnr_b2000")):
            if rec.get(key) is not None and rec[key] < Z_CNR:
                f.append(f"low_{name}")
        if rec["main_rel_max_mm"] is not None and rec["main_rel_max_mm"] > MAX_REL_MM:
            f.append("rel_motion_gt_2mm")
        if rec["outliers_total_pct"] is not None and rec["outliers_total_pct"] > MAX_OUTLIER_PCT:
            f.append("outliers_gt_5pct")
        rec["flags"] = ";".join(f)

    out = a.deriv_dir / "qc"
    out.mkdir(exist_ok=True)
    with open(out / f"eddy_qc_{a.arm}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for rec in recs:
            w.writerow({k: fmt(rec.get(k)) for k in COLUMNS})

    def summ(col):
        v = [r[col] for r in recs if r[col] is not None]
        return f"{statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]" if v else "n/a"

    L = [f"# EDDY QC summary: {a.arm} arm\n",
         f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by 04_eddy_qc.py v{SCRIPT_VERSION} from BRC eddy_quad "
         "`qc.json` and EDDY movement/outlier text files (routine QC; no images, no diffusion maps).\n",
         f"Sessions: {len(recs)}; missing: {len(missing)}\n"]
    if missing:
        L.append("\n".join(f"- {m}" for m in missing) + "\n")
    L.append("## Per scanner: median [min-max]\n")
    L.append("| scanner | n | abs motion (mm) | rel motion, EDDY (mm) | rel motion, main run (mm) | max rel, main run (mm) | "
             "jump into reverse b0 (mm) | outliers (%) | b0 SNR | b1000 CNR | b2000 CNR |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in sorted({r["scanner_label"] for r in recs}):
        g = [r for r in recs if r["scanner_label"] == lab]

        def s(col):
            v = [r[col] for r in g if r[col] is not None]
            return f"{statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]" if v else "n/a"
        L.append(f"| {lab} | {len(g)} | {s('mot_abs_mean_mm')} | {s('mot_rel_mean_mm')} | {s('main_rel_mean_mm')} | "
                 f"{s('main_rel_max_mm')} | {s('rel_jump_into_revpe_mm')} | {s('outliers_total_pct')} | {s('snr_b0')} | "
                 f"{s('cnr_b1000')} | {s('cnr_b2000')} |")
    L.append(f"| **all** | {len(recs)} | {summ('mot_abs_mean_mm')} | {summ('mot_rel_mean_mm')} | {summ('main_rel_mean_mm')} | "
             f"{summ('main_rel_max_mm')} | {summ('rel_jump_into_revpe_mm')} | {summ('outliers_total_pct')} | | | |\n")
    flagged = [r for r in recs if r["flags"]]
    L.append(f"## Flagged sessions ({len(flagged)})\n")
    L.append(f"Screening flags: robust z > {Z_MOTION} over all sessions for motion/outliers; robust z < {Z_CNR} within "
             f"scanner for CNR; any volume-to-volume motion > {MAX_REL_MM} mm; outliers > {MAX_OUTLIER_PCT} %. These are "
             "for review, not exclusions.\n")
    if flagged:
        L.append("| session | scanner | abs | rel (main) | max rel (main) | outliers % | CNR b1000 | CNR b2000 | flags |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in flagged:
            L.append(f"| {r['subject']}/{r['session']} | {r['scanner_label']} | {fmt(r['mot_abs_mean_mm'], 2)} | "
                     f"{fmt(r['main_rel_mean_mm'], 2)} | {fmt(r['main_rel_max_mm'], 2)} | {fmt(r['outliers_total_pct'], 2)} | "
                     f"{fmt(r['cnr_b1000'], 2)} | {fmt(r['cnr_b2000'], 2)} | {r['flags']} |")
    else:
        L.append("None.")
    (out / f"eddy_qc_{a.arm}.md").write_text("\n".join(L) + "\n")
    print(f"Wrote {out / f'eddy_qc_{a.arm}.csv'} ({len(recs)} sessions, {len(flagged)} flagged, {len(missing)} missing)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
