#!/usr/bin/env python3
"""
H1b.py

Hierarchical Bayesian power simulation for H1/2b.

Model:
    y_{s,r,c} ~ Normal(alpha0 + alpha1*den + u_s + v_r, sigma)

Usage (example):
    python H1b.py --output h1b.csv --replicates 50 --draws 1000 --tune 500 --cores 2
"""

import argparse
import csv
import os
import sys
import math
import logging
from functools import partial
import numpy as np
import pymc as pm
from scipy.stats import norm
from scipy.stats import gaussian_kde
from tqdm import trange, tqdm
import multiprocessing as mp
import arviz as az

# Make invalid floating-point operations raise errors
try:
    np.seterr(invalid='raise')
except Exception:
    pass

def _check_array(name, a):
    a = np.asarray(a)
    info = dict(name=name, shape=a.shape, dtype=str(a.dtype))
    try:
        info['has_nan'] = bool(np.isnan(a).any())
    except Exception:
        info['has_nan'] = 'error'
    try:
        info['has_inf'] = bool(np.isinf(a).any())
    except Exception:
        info['has_inf'] = 'error'
    try:
        info['min'] = float(np.nanmin(a))
        info['max'] = float(np.nanmax(a))
    except Exception:
        info['min'] = info['max'] = None
    print(f"CHECK {name}: shape={info['shape']} dtype={info['dtype']} has_nan={info['has_nan']} has_inf={info['has_inf']} min={info['min']} max={info['max']}")
    sys.stdout.flush()
    return info

# ---------------------------
# Simulation function (subject-level and ROI-level)
# ---------------------------
def simulate_data(S=10, R_per_subject=1298,
                  alpha0=0.0, alpha1=0.20,
                  tau_u=0.6, tau_v=0.2,
                  sigma=1.0, seed=None):
    """
        Generate data with:
            - S subjects (s = 0..S-1)
            - R_per_subject ROI/location units per subject (r = 0..R-1), shared across subjects
            - den is a 0/1 covariate with half zeros and half ones per subject

    Returns dict(y, subj, r, den)
    """
    rng = np.random.default_rng(seed)

    # den template (half 0, half 1)
    half = R_per_subject // 2
    den_template = np.concatenate([np.zeros(half), np.ones(R_per_subject - half)])

    # full vectors repeated across subjects
    subj = np.repeat(np.arange(S), R_per_subject)
    r_idx = np.tile(np.arange(R_per_subject), S)
    den = np.tile(den_template, S)

    # subject-level random intercepts
    u_s = rng.normal(0.0, tau_u, size=S)

    # ROI-level random intercepts (shared across subjects)
    v_r = rng.normal(0.0, tau_v, size=R_per_subject)

    # linear predictor
    mu = (alpha0
          + alpha1 * den
          + u_s[subj]
        + v_r[r_idx])

    y = rng.normal(mu, sigma, size=mu.shape)

    return dict(y=y, subj=subj, r=r_idx, den=den)

# ---------------------------
# Fit hierarchical model and compute BF via Savage-Dickey
# ---------------------------
def fit_and_compute_bf(data, prior_sd_alpha1=1.0, draws=2000, tune=2000, chains=4, cores=1, random_seed=None):
    y = data['y']
    subj = data['subj']
    den = data['den']
    r = data['r']
    S = int(subj.max()) + 1
    R = int(r.max()) + 1

    _check_array('y', y)
    _check_array('subj', subj)
    _check_array('r', r)
    _check_array('den', den)

    prior_density_at0 = norm.pdf(0.0, loc=0.0, scale=prior_sd_alpha1)

    with pm.Model() as model:
        # Fixed effects
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        alpha1 = pm.Normal("alpha1", mu=0.0, sigma=prior_sd_alpha1)

        # Subject-level random intercept (non-centered)
        sd_u = pm.HalfNormal("sd_u", sigma=1.0)
        z_u = pm.Normal("z_u", mu=0.0, sigma=1.0, shape=S)
        u = pm.Deterministic("u", z_u * sd_u)

        # ROI-level random intercept (non-centered)
        sd_v = pm.HalfNormal("sd_v", sigma=1.0)
        z_v = pm.Normal("z_v", mu=0.0, sigma=1.0, shape=R)
        v = pm.Deterministic("v", z_v * sd_v)

        # Residual SD
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # Linear predictor combining subject- and ROI-level intercepts
        mu = (alpha0
            + u[subj]
            + v[r]
            + alpha1 * den)

        # Observed outcomes
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sampling
        nuts = pm.NUTS(target_accept=0.99, max_treedepth=15)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          random_seed=random_seed, step=nuts, progressbar=False)

    posterior_alpha1 = idata.posterior["alpha1"].values.reshape(-1)
    n_post = posterior_alpha1.size

    # -------------------------------
    # Compute BF10 (Option C: normal + KDE)
    # -------------------------------
    mean_alpha1 = float(np.mean(posterior_alpha1))
    sd_alpha1 = float(np.std(posterior_alpha1, ddof=1)) if n_post > 1 else 1e-8
    sd_alpha1 = max(sd_alpha1, 1e-8)  # guard

    # Normal approximation density at 0
    post_d0_norm = float(norm.pdf(0.0, loc=mean_alpha1, scale=sd_alpha1))

    # KDE density at 0 for diagnostics
    try:
        kde = gaussian_kde(posterior_alpha1, bw_method='scott')
        post_d0_kde = float(kde.evaluate(0.0)[0])
    except Exception:
        post_d0_kde = 1e-300

    # Log-space stable BF
    post_d0_use = max(post_d0_norm, 1e-300)
    prior_d0 = max(prior_density_at0, 1e-300)
    log_BF10 = np.log(prior_d0) - np.log(post_d0_use)
    BF10 = float(np.exp(log_BF10))

    # Diagnostics: rhat, ess, divergences, acceptance rate
    try:
        rhat_alpha1 = float(az.rhat(idata, var_names=["alpha1"]).to_array().values.flatten()[0])
    except Exception:
        rhat_alpha1 = float('nan')

    try:
        ess_alpha1 = float(az.ess(idata, var_names=["alpha1"]).to_array().values.flatten()[0])
    except Exception:
        ess_alpha1 = float('nan')

    try:
        n_div = int(np.sum(idata.sample_stats["diverging"].values))
    except Exception:
        n_div = 0

    try:
        ar = idata.sample_stats.get("acceptance_rate")
        acceptance_rate = float(np.mean(ar.values)) if ar is not None else float('nan')
    except Exception:
        acceptance_rate = float('nan')

    return dict(
        BF10=BF10,
        BF10_kde=post_d0_kde,
        prior_d0=prior_d0,
        post_d0_norm=post_d0_norm,
        post_d0_kde=post_d0_kde,
        post_d0_used=post_d0_use,
        posterior_alpha1=posterior_alpha1,
        mean_alpha1=mean_alpha1,
        sd_alpha1=sd_alpha1,
        rhat_alpha1=rhat_alpha1,
        ess_alpha1=ess_alpha1,
        n_divergences=n_div,
        acceptance_rate=acceptance_rate,
        n_posterior_samples=n_post,
        draws=draws,
        tune=tune,
        chains=chains
    )

# ---------------------------
# Monte Carlo replicate worker
# ---------------------------
def _replicate_worker(i, S, R_per_subject, alpha1, tau_u, tau_v, sigma, prior_sd_alpha1, draws, tune, chains, cores, rng):
    seed = int(rng.integers(1, 2**30))
    data = simulate_data(S=S, R_per_subject=R_per_subject, alpha1=alpha1,
                         tau_u=tau_u, tau_v=tau_v,
                         sigma=sigma, seed=seed)
    fit_res = fit_and_compute_bf(
        data,
        prior_sd_alpha1=prior_sd_alpha1,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        random_seed=seed
    )

    out = dict(rep=i, seed=seed)
    for key in ['BF10','BF10_kde','prior_d0','post_d0_norm','post_d0_kde','post_d0_used',
                'mean_alpha1','sd_alpha1','rhat_alpha1','ess_alpha1','n_divergences','acceptance_rate','n_posterior_samples',
                'draws','tune','chains']:
        out[key] = fit_res.get(key, float('nan'))
    # also record sim parameters for traceability
    out.update(dict(S=S, R_per_subject=R_per_subject, alpha1=alpha1,
                    tau_u=tau_u, tau_v=tau_v, sigma=sigma))
    return out

# ---------------------------
# Estimate power across replicates
# ---------------------------
def estimate_power_hierarchical(S=9, R_per_subject=1298, alpha1=0.20,
                                tau_u=0.6, tau_v=0.2,
                                sigma=1.0,
                                prior_sd_alpha1=1.0,
                                replicates=200,
                                draws=2000, tune=2000, chains=4, cores=1,
                                BF_threshold=6.0, rng_seed=1234, parallel=False, n_workers=1,
                                progress_pos=0):
    rng = np.random.default_rng(rng_seed)
    results = []

    if parallel and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            func = partial(_replicate_worker, S=S, R_per_subject=R_per_subject, alpha1=alpha1,
                           tau_u=tau_u, tau_v=tau_v,
                           sigma=sigma, prior_sd_alpha1=prior_sd_alpha1,
                           draws=draws, tune=tune, chains=chains, cores=cores, rng=rng)
            for res in tqdm(pool.imap_unordered(func, range(replicates)), total=replicates, desc="MC replicates", position=progress_pos):
                results.append(res)
    else:
        for i in tqdm(range(replicates), desc="MC replicates", position=progress_pos):
            res = _replicate_worker(i, S, R_per_subject, alpha1, tau_u, tau_v, sigma, prior_sd_alpha1, draws, tune, chains, cores, rng)
            results.append(res)

    bf_values = np.array([r['BF10'] for r in results], dtype=float)
    valid_mask = np.isfinite(bf_values)
    n_failed = int(np.sum(~valid_mask))
    power_est = float(np.mean(bf_values[valid_mask] > BF_threshold)) if np.any(valid_mask) else float('nan')

    return dict(power=power_est, bf_values=bf_values, n_failed=n_failed, details=results)

# ---------------------------
# CSV output
# ---------------------------
def save_results_csv(outpath, details, append=False):
    base_cols = ['rep', 'BF10','BF10_kde','prior_d0','post_d0_norm','post_d0_kde','post_d0_used','seed',
                 'mean_alpha1','sd_alpha1','rhat_alpha1','ess_alpha1',
                 'n_divergences','acceptance_rate','n_posterior_samples',
                 'draws','tune','chains',
                 # sim params
                 'S','R_per_subject','alpha1','tau_u','tau_v','sigma']
    extra_cols = [k for k in details[0].keys() if k not in base_cols] if details else []
    header = base_cols + extra_cols
    dirname = os.path.dirname(outpath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    mode = 'a' if append and os.path.exists(outpath) else 'w'
    write_header = not (append and os.path.exists(outpath))
    with open(outpath, mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for r in details:
            row = [r.get(k) for k in base_cols]
            for c in extra_cols:
                row.append(r.get(c))
            writer.writerow(row)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="H1b Bayesian power simulation (hierarchical with ROI-level)")
    p.add_argument('--output', '-o', required=True, help='CSV output path for replicate results')
    p.add_argument('--replicates', type=int, default=200)
    p.add_argument('--S', type=int, default=10)
    p.add_argument('--R_per_subject', type=int, default=1298,
                   help='number of r-level units per subject (observations per subject)')
    p.add_argument('--alpha1', type=float, default=0.20)
    p.add_argument('--tau_u', type=float, default=0.6)
    p.add_argument('--tau_v', type=float, default=0.2)
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--prior_sd_alpha1', type=float, default=1.0)
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--tune', type=int, default=2000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--cores', type=int, default=1)
    p.add_argument('--BF_threshold', type=float, default=10.0)
    p.add_argument('--rng_seed', type=int, default=1234)
    p.add_argument('--parallel', action='store_true')
    p.add_argument('--n_workers', type=int, default=5)
    p.add_argument('--append', action='store_true')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.info(f'Starting H1b power simulation (r-level): replicates={args.replicates}, S={args.S}, R_per_subject={args.R_per_subject}')

    result = estimate_power_hierarchical(
        S=args.S,
        R_per_subject=args.R_per_subject,
        alpha1=args.alpha1,
        tau_u=args.tau_u,
        tau_v=args.tau_v,
        sigma=args.sigma,
        prior_sd_alpha1=args.prior_sd_alpha1,
        replicates=args.replicates,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        BF_threshold=args.BF_threshold,
        rng_seed=args.rng_seed,
        parallel=args.parallel,
        n_workers=args.n_workers if args.parallel else 1
    )

    logging.info(f"Estimated power (proportion BF>{args.BF_threshold}): {result['power']}")
    logging.info(f"Replicates failed: {result['n_failed']} / {args.replicates}")

    save_results_csv(args.output, result['details'], append=args.append)
    logging.info(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()
