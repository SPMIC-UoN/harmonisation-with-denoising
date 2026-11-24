#!/usr/bin/env python3
"""
descriptives.py - Native-unit descriptives and secondary summaries of the frozen confirmatory results.
No model is fitted and no confirmatory output is changed. Inputs: <deriv>/analysis/inputs (tables), the H4 fits in
<deriv>/analysis/results (posterior draws), and, for the H2 metric means, the template-space maps (same unit
selection as build_inputs.py).

Outputs in <out>/:
  h1_cov_native.csv    H1: mean/median cross-scanner CoV (%) per arm, absolute and relative change
  h2_bias_native.csv   H2: mean Delta Bias in native units, % of the mean metric value, SD over units, scanner
  h4_native.csv        H4: beta1 per 10 Hz |off-resonance| in CoV percentage points; beta0 (z and native);
                       mean/median scan-rescan CoV (%) per arm
Units: FA and MK dimensionless; MD in mm^2/s (b in s/mm^2), reported x 1e-3 mm^2/s.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import arviz as az
import numpy as np

import build_inputs as B
import inputs as I

METRICS = ("FA", "MD", "MK")
LEVELS = {"a": "voxel", "b": "ROI"}


def table(p: Path) -> dict:
    with gzip.open(p, "rt") as fh:
        r = csv.reader(fh)
        head = next(r)
        cols = list(zip(*r))
    return {h: np.array(c) for h, c in zip(head, cols)}


def ci(x):
    q = np.quantile(x, [0.025, 0.975])
    return float(np.mean(x)), float(q[0]), float(q[1])


def write(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def h1(inputs: Path) -> list[dict]:
    rows = []
    for m in METRICS:
        for lv, name in LEVELS.items():
            t = table(inputs / f"h1{lv}_{m}.csv.gz")
            cov, den = t["cov"].astype(float) * 100, t["den"].astype(int)
            key = np.char.add(np.char.add(t["subject"], "|"), t["unit"])
            o0, o1 = np.argsort(key[den == 0]), np.argsort(key[den == 1])
            c0, c1 = cov[den == 0][o0], cov[den == 1][o1]
            assert np.array_equal(key[den == 0][o0], key[den == 1][o1])
            d = c1 - c0
            rows.append(dict(metric=m, level=name, n_units=int(c0.size),
                             cov_pct_mean_nonden=c0.mean(), cov_pct_mean_den=c1.mean(),
                             cov_pct_median_nonden=float(np.median(c0)), cov_pct_median_den=float(np.median(c1)),
                             abs_change_mean_pp=c1.mean() - c0.mean(), abs_change_median_paired_pp=float(np.median(d)),
                             rel_change_of_means_pct=100 * (c1.mean() - c0.mean()) / c0.mean(),
                             rel_change_of_medians_pct=100 * (np.median(c1) - np.median(c0)) / np.median(c0),
                             units_lower_cov_den_pct=100 * float(np.mean(d < 0))))
    return rows


def h2(D: B.Data, inputs: Path) -> list[dict]:
    rows = []
    for sub, sc in sorted(D.repeat_scanner.items()):
        sess = D.sessions[sub]
        for m in METRICS:
            u = B.subject_units(D, sub, "all", sess, m, {}, "desc")
            rids = B.roi_ids(D.template(sub, "all") / "T_xtract_rois_counts.csv")
            for lv, name in LEVELS.items():
                t = table(inputs / f"h2{lv}_{sub}_{m}.csv.gz")
                db = t["delta_bias"].astype(float)
                units = t["unit"].astype(int)
                if lv == "a":
                    pos = {k: i for i, k in enumerate(u["flat_idx"])}
                    sel = np.array([pos[k] for k in units])
                    level_mean = np.mean([u["vals"][arm][:, sel].mean() for arm in B.ARMS])
                else:
                    means = [I.roi_means(u["vals"][arm], u["lab"], u["keep"], rids)[0] for arm in B.ARMS]
                    idx = np.searchsorted(rids, units)
                    level_mean = np.mean([mm[:, idx].mean() for mm in means])
                scale = 1e3 if m == "MD" else 1.0
                rows.append(dict(pair=sub, repeat_scanner=sc, metric=m, level=name, n_units=int(db.size),
                                 units=("x1e-3 mm2/s" if m == "MD" else "dimensionless"),
                                 delta_bias_mean=db.mean() * scale, delta_bias_sd=db.std(ddof=1) * scale,
                                 delta_bias_median=float(np.median(db)) * scale,
                                 metric_mean=level_mean * scale, delta_bias_pct_of_mean=100 * db.mean() / level_mean,
                                 delta_bias_sd_pct_of_mean=100 * db.std(ddof=1) / level_mean))
    return rows


def h4(D: B.Data, inputs: Path, results: Path) -> list[dict]:
    rows = []
    for sub, sc in sorted(D.repeat_scanner.items()):
        for m in METRICS:
            for lv, name in LEVELS.items():
                t = table(inputs / f"h4{lv}_{sub}_{m}.csv.gz")
                c0, c1 = t["cov_nonden"].astype(float), t["cov_den"].astype(float)
                off = t["off_hz"].astype(float)
                pooled = np.concatenate([c0, c1])
                s_cov = pooled.std(ddof=0)                         # zscore_pooled scale
                s_off = off.std(ddof=0)                            # SD of |off| (Hz)
                post = az.from_netcdf(str(results / f"H4{lv}_{sub}_{m}" / "idata.nc"), engine="h5netcdf").posterior
                b1 = post["beta1"].values.reshape(-1)
                b0 = post["beta0"].values.reshape(-1)
                n1 = b1 * s_cov * 100 * 10                         # b1 is per Hz -> CoV percentage points per 10 Hz |off|
                n0 = b0 * s_cov * 100                              # CoV percentage points at zero off-resonance
                (b1m, b1l, b1h), (n1m, n1l, n1h), (b0m, b0l, b0h), (n0m, n0l, n0h) = map(ci, (b1, n1, b0, n0))
                rows.append(dict(pair=sub, repeat_scanner=sc, metric=m, level=name, n_units=int(c0.size),
                                 cov_pct_mean_nonden=100 * c0.mean(), cov_pct_mean_den=100 * c1.mean(),
                                 cov_pct_median_nonden=100 * float(np.median(c0)),
                                 cov_pct_median_den=100 * float(np.median(c1)),
                                 off_hz_mean=off.mean(), off_hz_sd=s_off, cov_pooled_sd_pct=100 * s_cov,
                                 beta1_z=b1m * s_off, beta1_z_low=b1l * s_off, beta1_z_high=b1h * s_off,
                                 beta1_pp_per_10Hz=n1m, beta1_pp_per_10Hz_low=n1l, beta1_pp_per_10Hz_high=n1h,
                                 beta0_z=b0m, beta0_z_low=b0l, beta0_z_high=b0h,
                                 beta0_pp=n0m, beta0_pp_low=n0l, beta0_pp_high=n0h))
    return rows


def h2_per_arm(D: B.Data, inputs: Path, out: Path) -> int:
    """Per-arm Bias (mu_WITHIN - mu_ACROSS) for the confirmatory H2 units, for plotting. Checks that
    Bias_den - Bias_nonden reproduces the stored Delta Bias exactly. Native units (MD in mm^2/s)."""
    n = 0
    with gzip.open(out / "h2_bias_per_arm.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["pair", "repeat_scanner", "metric", "level", "unit", "bias_nonden", "bias_den", "metric_mean",
                    "value"])
        for sub, sc in sorted(D.repeat_scanner.items()):
            sess = D.sessions[sub]
            rep = np.array([s == sc for _, s in sess])
            for m in METRICS:
                u = B.subject_units(D, sub, "all", sess, m, {}, "desc")
                rids = B.roi_ids(D.template(sub, "all") / "T_xtract_rois_counts.csv")
                for lv, name in LEVELS.items():
                    t = table(inputs / f"h2{lv}_{sub}_{m}.csv.gz")
                    units = t["unit"].astype(int)
                    if lv == "a":
                        pos = {k: i for i, k in enumerate(u["flat_idx"])}
                        sel = np.array([pos[k] for k in units])
                        v = {arm: u["vals"][arm][:, sel] for arm in B.ARMS}
                    else:
                        idx = np.searchsorted(rids, units)
                        v = {arm: I.roi_means(u["vals"][arm], u["lab"], u["keep"], rids)[0][:, idx] for arm in B.ARMS}
                    bias = {arm: v[arm][rep].mean(0) - v[arm][~rep].mean(0) for arm in B.ARMS}
                    diff = bias[B.ARMS[1]] - bias[B.ARMS[0]]
                    if not np.allclose(diff, t["delta_bias"].astype(float), rtol=1e-6, atol=1e-12):
                        raise RuntimeError(f"{sub} {m} {name}: per-arm bias does not reproduce Delta Bias")
                    mm = float(np.mean([v[arm].mean() for arm in B.ARMS]))
                    val = 0.5 * (v[B.ARMS[0]].mean(0) + v[B.ARMS[1]].mean(0))      # per unit: all sessions, both arms
                    for k in range(units.size):
                        w.writerow([sub, sc, m, name, int(units[k]), repr(float(bias[B.ARMS[0]][k])),
                                    repr(float(bias[B.ARMS[1]][k])), repr(mm), repr(float(val[k]))])
                        n += 1
    return n


def h1_value_dcov(D: B.Data, out: Path) -> int:
    """Per subject x unit (H1 units: checkerboard voxels, ROIs): metric value (mean over scanners and both arms) and
    cross-scanner CoV per arm (%), for plotting Delta CoV against the metric value. Same rules as build_inputs."""
    n = 0
    with gzip.open(out / "h1_value_vs_dcov.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["subject", "metric", "level", "unit", "value", "cov_nonden", "cov_den"])
        for sub in sorted(D.sessions):
            sess = D.sessions[sub]
            scanners = [sc for _, sc in sess]
            rids = B.roi_ids(D.template(sub, "all") / "T_xtract_rois_counts.csv")
            for m in METRICS:
                u = B.subject_units(D, sub, "all", sess, m, {}, "desc")
                sel = u["lat"] & u["keep"]
                for lvl, x, units in (
                        ("voxel", {a: u["vals"][a][:, sel] for a in B.ARMS}, u["flat_idx"][sel]),
                        ("ROI", {a: I.roi_means(u["vals"][a], u["lab"], u["keep"], rids)[0] for a in B.ARMS}, rids)):
                    per = {a: I.average_by_group(x[a], scanners)[0] for a in B.ARMS}
                    c0, c1 = I.missing_in_either_arm(I.cov(per[B.ARMS[0]]) * 100, I.cov(per[B.ARMS[1]]) * 100)
                    val = 0.5 * (per[B.ARMS[0]].mean(0) + per[B.ARMS[1]].mean(0))
                    ok = np.isfinite(c0) & np.isfinite(val)
                    for k in np.flatnonzero(ok):
                        w.writerow([sub, m, lvl, int(units[k]), repr(float(val[k])), repr(float(c0[k])), repr(float(c1[k]))])
                        n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only-h2-per-arm", action="store_true")
    p.add_argument("--only-h1-value", action="store_true")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    inputs, results = a.deriv_dir / "analysis" / "inputs", a.deriv_dir / "analysis" / "results"
    D = B.Data(a.deriv_dir, a.manifest)
    if a.only_h1_value:
        print(f"h1_value_vs_dcov: {h1_value_dcov(D, a.out)} rows")
        return 0
    if a.only_h2_per_arm:
        print(f"h2_bias_per_arm: {h2_per_arm(D, inputs, a.out)} rows (Delta Bias reproduced)")
        return 0
    write(a.out / "h1_cov_native.csv", h1(inputs))
    write(a.out / "h4_native.csv", h4(D, inputs, results))
    write(a.out / "h2_bias_native.csv", h2(D, inputs))
    print(json.dumps({"written": [str(x) for x in sorted(a.out.glob("*.csv"))]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
