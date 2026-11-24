#!/usr/bin/env python3
"""
summarise_results.py - Convergence and confirmatory results tables from run_models.py outputs.

  convergence_all.csv      R-hat (rank-normalised), bulk and tail ESS for EVERY parameter element of every fit
  convergence_by_var.csv   per fit x variable: max R-hat, min bulk/tail ESS over elements
  ess_extend.txt           fits failing the convergence rule (R-hat < 1.01 and bulk ESS > 1000 for the inferential
                           quantities: alpha1 / alpha0 / beta1; H3: sd_u, sd_w, sigma_cond[0])
  results.csv              one row per hypothesis x metric x level (x pair): posterior mean and 95% CrI of the
                           target, BF10 (capped flag), effect size in SD units, verdict (Section 3.5, Table 3)

  python summarise_results.py --results <deriv>/analysis/results [--tag 4000]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import arviz as az
import numpy as np

CAP_DENSITY = 1e-300                  # simulation code's floor on the posterior density at 0 -> BF10 capped
INFER = {"H1a": ["alpha1"], "H1b": ["alpha1"], "H2a": ["alpha0"], "H2b": ["alpha0"], "H4": ["beta1"],
         "H3": ["sd_u", "sd_w", "sigma_cond[0]"]}


def load(d: Path):
    return json.loads((d / "result.json").read_text()), az.from_netcdf(str(d / "idata.nc"), engine="h5netcdf")


def convergence(job: str, model: str, idata) -> list[dict]:
    post = idata.posterior
    names = list(post.data_vars)
    rh, eb, et = az.rhat(idata, var_names=names), az.ess(idata, var_names=names, method="bulk"), \
        az.ess(idata, var_names=names, method="tail")
    rows = []
    for v in names:
        r, b, t = (np.atleast_1d(x[v].values).ravel() for x in (rh, eb, et))
        for j in range(r.size):
            if model == "H3" and v == "sigma_cond" and j == 1:
                continue                                      # no data in a one-arm fit
            rows.append(dict(job=job, model=model, var=v, param=v if r.size == 1 else f"{v}[{j}]",
                             rhat=float(r[j]), ess_bulk=float(b[j]), ess_tail=float(t[j])))
    return rows


def ci(x):
    q = np.quantile(x, [0.025, 0.975])
    return float(np.mean(x)), float(q[0]), float(q[1])


def verdict(model: str, bf: float | None, mean: float | None, p: float | None = None) -> str:
    if model in ("H1a", "H1b"):
        if bf > 10:
            return "supports H1" if mean < 0 else "inconclusive (BF10 > 10, opposite sign)"
        return "supports null" if bf < 0.1 else "inconclusive"
    if model in ("H2a", "H2b"):
        if bf < 0.1:
            return "supports H2 (no change in bias)"
        return "denoising changes bias" if bf > 10 else "inconclusive"
    if model == "H4":
        if bf > 10:
            return "supports H4" if mean > 0 else "inconclusive (BF10 > 10, opposite sign)"
        return "supports null" if bf < 0.1 else "inconclusive"
    if model in ("H3a", "H3b"):
        return "supports H3" if p > 0.95 else ("supports null" if p < 0.05 else "inconclusive")
    raise KeyError(model)


def parse_job(job: str) -> dict:
    parts = job.split("_")
    if parts[0].startswith("H3"):
        return dict(hypothesis="H3", level={"a": "voxel", "b": "ROI"}[parts[0][2]], metric=parts[1], pair="")
    h, lvl = parts[0][:2], {"a": "voxel", "b": "ROI"}[parts[0][2]]
    return dict(hypothesis=h, level=lvl, metric=parts[-1], pair=parts[1] if len(parts) == 3 else "")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default: --results")
    a = p.parse_args(argv)
    out = a.out or a.results
    conv, res, extend, h3 = [], [], [], {}
    for d in sorted(x for x in a.results.iterdir() if (x / "result.json").is_file()):
        r, idata = load(d)
        job, model = r["job"], r["model"]
        c = convergence(job, model, idata)
        conv += c
        inf = [x for x in c if x["param"] in INFER[model]]
        bad = [f"{x['param']} rhat={x['rhat']:.4f} ess_bulk={x['ess_bulk']:.0f}" for x in inf
               if not (x["rhat"] < 1.01 and x["ess_bulk"] > 1000)]
        if bad:
            extend.append(f"{job}: " + "; ".join(bad))
        post = idata.posterior
        info = parse_job(job)
        conv_ok = not bad
        if model == "H3":
            dr = np.load(d / "icc_draws.npz")
            h3[job] = (dr["icc"], conv_ok, int(r["n_obs"]), r.get("n_divergences", 0))
            continue
        tgt = r["target"]
        x = post[tgt].values.reshape(-1)
        sig = post["sigma"].values.reshape(-1)
        mean, lo, hi = ci(x)
        es_mean, es_lo, es_hi = ci(x / sig)
        capped = r["post_d0_used"] <= CAP_DENSITY
        res.append(dict(**info, target=tgt, n_obs=r["n_obs"], mean=mean, ci_low=lo, ci_high=hi,
                        BF10=r["BF10"], BF10_capped=capped, BF10_report="capped" if capped else f"{r['BF10']:.4g}",
                        effect_sd_units=mean, effect_over_sigma=es_mean, effect_over_sigma_ci_low=es_lo,
                        effect_over_sigma_ci_high=es_hi, sigma_mean=float(sig.mean()),
                        converged=conv_ok, n_divergences=r.get("n_divergences"),
                        verdict=verdict(model, r["BF10"], mean)))
    for job in sorted(j for j in h3 if j.endswith("_den1")):
        base = job[:-5]
        if base + "_den0" not in h3:
            continue
        i1, ok1, n1, d1 = h3[job]
        i0, ok0, n0, d0 = h3[base + "_den0"]
        delta = i1 - i0
        info = parse_job(base)
        m1, l1, u1 = ci(i1); m0, l0, u0 = ci(i0); md, ld, ud = ci(delta)
        pg = float(np.mean(delta > 0))
        res.append(dict(**info, target="delta_icc", n_obs=n0 + n1, icc_den=m1, icc_den_low=l1, icc_den_high=u1,
                        icc_nonden=m0, icc_nonden_low=l0, icc_nonden_high=u0, mean=md, ci_low=ld, ci_high=ud,
                        p_delta_gt0=pg, converged=ok0 and ok1, n_divergences=d0 + d1,
                        verdict=verdict(f"H3{base[2]}", None, None, pg)))
    keys = []
    for r in res:
        keys += [k for k in r if k not in keys]
    order = {"H1": 1, "H2": 2, "H3": 3, "H4": 4}
    res.sort(key=lambda r: (order[r["hypothesis"]], r["level"] != "voxel", r["metric"], r["pair"]))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(res)
    with open(out / "convergence_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conv[0]), lineterminator="\n"); w.writeheader(); w.writerows(conv)
    byvar = {}
    for c in conv:
        k = (c["job"], c["var"])
        b = byvar.setdefault(k, dict(job=c["job"], model=c["model"], var=c["var"], n=0, max_rhat=0.0,
                                     min_ess_bulk=np.inf, min_ess_tail=np.inf))
        b["n"] += 1; b["max_rhat"] = max(b["max_rhat"], c["rhat"])
        b["min_ess_bulk"] = min(b["min_ess_bulk"], c["ess_bulk"]); b["min_ess_tail"] = min(b["min_ess_tail"], c["ess_tail"])
    with open(out / "convergence_by_var.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(byvar.values()))), lineterminator="\n")
        w.writeheader(); w.writerows(byvar.values())
    (out / "ess_extend.txt").write_text("\n".join(extend) + ("\n" if extend else ""))
    print(f"{len(res)} result rows; {len(conv)} parameter elements; {len(extend)} fits fail the convergence rule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
