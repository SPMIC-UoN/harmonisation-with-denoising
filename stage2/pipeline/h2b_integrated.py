#!/usr/bin/env python3
"""
h2b_integrated.py - Exploratory (Table S16a): the ROI-wise H2 model (H2b) with the ROI effects integrated out.

The registered H2b model (simulations/H2/H2b.py) is
    Delta_r ~ N(alpha0 + v_r, sigma_e),  v_r ~ N(0, sigma_v),  alpha0 ~ N(0, 1),  sigma_v, sigma_e ~ HalfNormal(1).
Each ROI has a single Delta_r, so v_r and the residual cannot be separated, and seven of the fifteen fits did not meet
the convergence criteria. Integrating v_r out gives
    Delta_r ~ N(alpha0, sqrt(sigma_v^2 + sigma_e^2)),
which has the same parameters, priors and joint posterior of (alpha0, sigma_v, sigma_e) but no ROI effects to sample.
Sampler settings and the Bayes factor (Savage-Dickey, normal approximation) are those of the registered fit:
NUTS, target_accept 0.95, max_treedepth 12, 4 chains x (2,000 warm-up + 2,000 draws).

  python h2b_integrated.py --inputs <deriv>/analysis/inputs --out <deriv>/analysis/results_h2b_integrated

Writes <out>/<job>/result.json and <out>/<job>/summary.csv for each H2b table, and <out>/results.csv.
Runs in the pipeline Python environment (environment.yml: PyMC 5.10.4, ArviZ 0.23.4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261126


def fit(y: np.ndarray, draws: int, tune: int, cores: int):
    import pymc as pm
    with pm.Model():
        a0 = pm.Normal("alpha0", 0.0, 1.0)
        sd_v = pm.HalfNormal("sd_v", 1.0)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y_obs", mu=a0, sigma=pm.math.sqrt(sd_v ** 2 + sigma ** 2), observed=y)
        return pm.sample(draws=draws, tune=tune, chains=4, cores=cores, random_seed=SEED,
                         step=pm.NUTS(target_accept=0.95, max_treedepth=12), progressbar=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--cores", type=int, default=4)
    a = p.parse_args()

    import arviz as az
    from scipy.stats import norm
    rows = []
    for t in sorted(a.inputs.glob("h2b_*.csv.gz")):
        job = "H2b" + t.name[3:].replace(".csv.gz", "")
        y = pd.read_csv(t, usecols=["y"])["y"].to_numpy(float)
        idata = fit(y, a.draws, a.tune, a.cores)
        post = idata.posterior["alpha0"].values.ravel()
        s = az.summary(idata, var_names=["alpha0", "sd_v", "sigma"], hdi_prob=0.95)
        m, sd = float(post.mean()), float(post.std(ddof=1))
        log10_bf = float((norm.logpdf(0, 0, 1) - norm.logpdf(0, m, sd)) / np.log(10))
        conv = bool((s["r_hat"] < 1.01).all() and (s["ess_bulk"] > 1000).all())
        res = dict(job=job, model="H2b, v_r integrated out", n_obs=len(y), draws=a.draws, tune=a.tune, chains=4, seed=SEED,
                   mean_alpha0=m, sd_alpha0=sd, q2_5=float(np.quantile(post, 0.025)), q97_5=float(np.quantile(post, 0.975)),
                   log10_BF10=log10_bf, rhat_alpha0=float(s.loc["alpha0", "r_hat"]), ess_alpha0=float(s.loc["alpha0", "ess_bulk"]),
                   n_divergences=int(idata.sample_stats["diverging"].sum()), converged_all_params=conv)
        o = a.out / job
        o.mkdir(parents=True, exist_ok=True)
        (o / "result.json").write_text(json.dumps(res, indent=1))
        s.rename_axis("param").to_csv(o / "summary.csv")
        rows.append(res)
        print(job, f"alpha0 {m:+.3f}  R-hat {res['rhat_alpha0']:.3f}  ESS {res['ess_alpha0']:.0f}  "
                   f"divergences {res['n_divergences']}", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "results.csv", index=False)


if __name__ == "__main__":
    main()
