#!/usr/bin/env python3
"""
H3.py

Script for Bayesian power simulation under linear model:

  y_i ~ Normal(beta0 + beta1 * off_r, sigma)

Usage (example):
  python H3.py --output results_h3.csv --N 1298 --replicates 1000 --draws 1000 --tune 1000 --cores 1

"""

import argparse
import csv
import os
import sys
import logging
from functools import partial
import numpy as np
import pymc as pm
from scipy.stats import norm
from scipy.stats import gaussian_kde
from tqdm import tqdm
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
# Simulation function
# ---------------------------

def simulate_data(N=1298, beta0=0.0, beta1=0.20, sigma=1.0,
                  off_r_dist='normal', off_r_scale=1.0, seed=None):
    """
    Simulate a simple dataset for the model:
        y ~ Normal(beta0 + beta1 * off_r, sigma)

    Args:
      - N: number of observations
      - beta0, beta1: true coefficients
      - sigma: residual standard deviation
      - off_r_dist: distribution of covariate off_r: 'normal' or 'binary'
      - off_r_scale: scaling factor applied to off_r (e.g., std dev for normal)
      - seed: RNG seed

    Returns dict(y, off_r)
    """
    rng = np.random.default_rng(seed)
    if off_r_dist == 'normal':
        off_r = rng.normal(0.0, off_r_scale, size=N)
    elif off_r_dist == 'binary':
        # Half zeros, half ones
        half = N // 2
        off_r = np.concatenate([np.zeros(half), np.ones(N - half)])
        rng.shuffle(off_r)
    else:
        raise ValueError(f"Unsupported off_r_dist: {off_r_dist}")

    mu = beta0 + beta1 * off_r
    y = rng.normal(mu, sigma, size=N)
    return dict(y=y, off_r=off_r)


# ---------------------------
# Fit simple model and compute BF via Savage–Dickey
# ---------------------------

def fit_and_compute_bf(data, prior_sd_beta1=1.0, draws=1000, tune=1000, chains=4, cores=1, random_seed=None):
    y = np.asarray(data['y'])
    off_r = np.asarray(data['off_r'])

    _check_array('y', y)
    _check_array('off_r', off_r)

    prior_density_at0 = norm.pdf(0.0, loc=0.0, scale=prior_sd_beta1)

    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)
        beta1 = pm.Normal("beta1", mu=0.0, sigma=prior_sd_beta1)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = beta0 + beta1 * off_r
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        nuts = pm.NUTS(target_accept=0.95, max_treedepth=12)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          random_seed=random_seed, step=nuts, progressbar=False)

    posterior_beta1 = idata.posterior["beta1"].values.reshape(-1)
    n_post = posterior_beta1.size

    mean_b1 = float(np.mean(posterior_beta1))
    sd_b1 = float(np.std(posterior_beta1, ddof=1)) if n_post > 1 else 1e-8
    sd_b1 = max(sd_b1, 1e-8)

    post_d0_norm = float(norm.pdf(0.0, loc=mean_b1, scale=sd_b1))
    try:
        kde = gaussian_kde(posterior_beta1, bw_method='scott')
        post_d0_kde = float(kde.evaluate(0.0)[0])
    except Exception:
        post_d0_kde = 1e-300

    post_d0_use = max(post_d0_norm, 1e-300)
    prior_d0 = max(prior_density_at0, 1e-300)
    log_BF10 = np.log(prior_d0) - np.log(post_d0_use)
    BF10 = float(np.exp(log_BF10))

    # Diagnostics
    try:
        rhat_b1 = float(az.rhat(idata, var_names=["beta1"]).to_array().values.flatten()[0])
    except Exception:
        rhat_b1 = float('nan')
    try:
        ess_b1 = float(az.ess(idata, var_names=["beta1"]).to_array().values.flatten()[0])
    except Exception:
        ess_b1 = float('nan')
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
        posterior_beta1=posterior_beta1,
        mean_beta1=mean_b1,
        sd_beta1=sd_b1,
        rhat_beta1=rhat_b1,
        ess_beta1=ess_b1,
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

def _replicate_worker(i, N, beta0, beta1, sigma, off_r_dist, off_r_scale,
                      prior_sd_beta1, draws, tune, chains, cores, rng):
    seed = int(rng.integers(1, 2**30))
    data = simulate_data(N=N, beta0=beta0, beta1=beta1, sigma=sigma,
                         off_r_dist=off_r_dist, off_r_scale=off_r_scale, seed=seed)
    fit_res = fit_and_compute_bf(
        data,
        prior_sd_beta1=prior_sd_beta1,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        random_seed=seed
    )

    out = dict(rep=i, seed=seed)
    for key in ['BF10','BF10_kde','prior_d0','post_d0_norm','post_d0_kde','post_d0_used',
                'mean_beta1','sd_beta1','rhat_beta1','ess_beta1','n_divergences','acceptance_rate','n_posterior_samples',
                'draws','tune','chains']:
        out[key] = fit_res.get(key, float('nan'))
    # record sim params for traceability
    out.update(dict(N=N, beta0=beta0, beta1=beta1, sigma=sigma, off_r_dist=off_r_dist, off_r_scale=off_r_scale))
    return out


# ---------------------------
# Estimate power across replicates
# ---------------------------

def estimate_power_simple(N=1298, beta0=0.0, beta1=0.20, sigma=1.0,
                          off_r_dist='normal', off_r_scale=1.0,
                          prior_sd_beta1=1.0,
                          replicates=200,
                          draws=1000, tune=1000, chains=4, cores=1,
                          BF_threshold=10.0, rng_seed=1234, parallel=False, n_workers=1,
                          progress_pos=0):
    rng = np.random.default_rng(rng_seed)
    results = []

    if parallel and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            func = partial(_replicate_worker, N=N, beta0=beta0, beta1=beta1, sigma=sigma,
                           off_r_dist=off_r_dist, off_r_scale=off_r_scale,
                           prior_sd_beta1=prior_sd_beta1,
                           draws=draws, tune=tune, chains=chains, cores=cores, rng=rng)
            for res in tqdm(pool.imap_unordered(func, range(replicates)), total=replicates, desc="MC replicates", position=progress_pos):
                results.append(res)
    else:
        for i in tqdm(range(replicates), desc="MC replicates", position=progress_pos):
            res = _replicate_worker(i, N, beta0, beta1, sigma, off_r_dist, off_r_scale, prior_sd_beta1, draws, tune, chains, cores, rng)
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
                 'mean_beta1','sd_beta1','rhat_beta1','ess_beta1',
                 'n_divergences','acceptance_rate','n_posterior_samples',
                 'draws','tune','chains',
                 # sim params
                 'N','beta0','beta1','sigma','off_r_dist','off_r_scale']
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
    p = argparse.ArgumentParser(description="H3 Bayesian power simulation (simple normal-linear model)")
    p.add_argument('--output', '-o', required=True, help='CSV output path for replicate results')
    p.add_argument('--replicates', type=int, default=200)
    p.add_argument('--N', type=int, default=1298, help='number of observations')
    p.add_argument('--beta0', type=float, default=0.0)
    p.add_argument('--beta1', type=float, default=0.20)
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--off_r_dist', choices=['normal','binary'], default='normal')
    p.add_argument('--off_r_scale', type=float, default=1.0, help='scale (std dev) for off_r if normal')
    p.add_argument('--prior_sd_beta1', type=float, default=1.0)
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--tune', type=int, default=1000)
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

    logging.info(f'Starting H3 power simulation: replicates={args.replicates}, N={args.N}, off_r_dist={args.off_r_dist}')

    result = estimate_power_simple(
        N=args.N,
        beta0=args.beta0,
        beta1=args.beta1,
        sigma=args.sigma,
        off_r_dist=args.off_r_dist,
        off_r_scale=args.off_r_scale,
        prior_sd_beta1=args.prior_sd_beta1,
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
