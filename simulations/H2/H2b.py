#!/usr/bin/env python3
"""
H2b.py

Null-point Bayesian power simulation for H2b.

Model:
    delta_bias_r ~ Normal(alpha0 + v_r, sigma)
    v_r ~ Normal(0, tau_v)

Evidence for the null (alpha0 = 0) is assessed via BF10 < BF_threshold.
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
    print(
        f"CHECK {name}: shape={info['shape']} dtype={info['dtype']} "
        f"has_nan={info['has_nan']} has_inf={info['has_inf']} "
        f"min={info['min']} max={info['max']}"
    )
    sys.stdout.flush()
    return info


# ---------------------------
# Simulation function
# ---------------------------

def simulate_data(R=360, alpha0=0.0, tau_v=0.2, sigma=1.0, seed=None):
    rng = np.random.default_rng(seed)
    v_r = rng.normal(0.0, tau_v, size=R)
    y = rng.normal(alpha0 + v_r, sigma, size=R)
    r = np.arange(R, dtype=int)
    return dict(y=y, r=r)


# ---------------------------
# Fit model and compute BF via Savage-Dickey
# ---------------------------

def fit_and_compute_bf(data, prior_sd_alpha0=1.0, draws=1000, tune=1000,
                       chains=4, cores=1, random_seed=None):
    y = np.asarray(data['y'])
    r = np.asarray(data['r'])
    R = int(r.max()) + 1

    _check_array('y', y)
    _check_array('r', r)

    prior_density_at0 = norm.pdf(0.0, loc=0.0, scale=prior_sd_alpha0)

    with pm.Model() as model:
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=prior_sd_alpha0)

        sd_v = pm.HalfNormal("sd_v", sigma=1.0)
        z_v = pm.Normal("z_v", mu=0.0, sigma=1.0, shape=R)
        v = pm.Deterministic("v", z_v * sd_v)

        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = alpha0 + v[r]
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        nuts = pm.NUTS(target_accept=0.95, max_treedepth=12)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            step=nuts,
            progressbar=False
        )

    posterior_alpha0 = idata.posterior["alpha0"].values.reshape(-1)
    n_post = posterior_alpha0.size

    mean_a0 = float(np.mean(posterior_alpha0))
    sd_a0 = float(np.std(posterior_alpha0, ddof=1)) if n_post > 1 else 1e-8
    sd_a0 = max(sd_a0, 1e-8)

    post_d0_norm = float(norm.pdf(0.0, loc=mean_a0, scale=sd_a0))
    try:
        kde = gaussian_kde(posterior_alpha0, bw_method='scott')
        post_d0_kde = float(kde.evaluate(0.0)[0])
    except Exception:
        post_d0_kde = 1e-300

    post_d0_use = max(post_d0_norm, 1e-300)
    prior_d0 = max(prior_density_at0, 1e-300)
    log_BF10 = np.log(prior_d0) - np.log(post_d0_use)
    BF10 = float(np.exp(log_BF10))

    # Diagnostics
    try:
        rhat_a0 = float(az.rhat(idata, var_names=["alpha0"]).to_array().values.flatten()[0])
    except Exception:
        rhat_a0 = float('nan')
    try:
        ess_a0 = float(az.ess(idata, var_names=["alpha0"]).to_array().values.flatten()[0])
    except Exception:
        ess_a0 = float('nan')
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
        posterior_alpha0=posterior_alpha0,
        mean_alpha0=mean_a0,
        sd_alpha0=sd_a0,
        rhat_alpha0=rhat_a0,
        ess_alpha0=ess_a0,
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

def _replicate_worker(i, R, alpha0, tau_v, sigma, prior_sd_alpha0, draws, tune, chains, cores, rng):
    seed = int(rng.integers(1, 2**30))
    data = simulate_data(R=R, alpha0=alpha0, tau_v=tau_v, sigma=sigma, seed=seed)
    fit_res = fit_and_compute_bf(
        data,
        prior_sd_alpha0=prior_sd_alpha0,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        random_seed=seed
    )

    out = dict(rep=i, seed=seed)
    for key in ['BF10','BF10_kde','prior_d0','post_d0_norm','post_d0_kde','post_d0_used',
                'mean_alpha0','sd_alpha0','rhat_alpha0','ess_alpha0','n_divergences',
                'acceptance_rate','n_posterior_samples','draws','tune','chains']:
        out[key] = fit_res.get(key, float('nan'))
    out.update(dict(R=R, alpha0=alpha0, tau_v=tau_v, sigma=sigma))
    return out


# ---------------------------
# Estimate null-support probability across replicates
# ---------------------------

def estimate_null_support(R=360, alpha0=0.0, tau_v=0.2, sigma=1.0,
                          prior_sd_alpha0=1.0,
                          replicates=200,
                          draws=1000, tune=1000, chains=4, cores=1,
                          BF_threshold=0.1, rng_seed=1234, parallel=False, n_workers=1,
                          progress_pos=0):
    rng = np.random.default_rng(rng_seed)
    results = []

    if parallel and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            func = partial(_replicate_worker, R=R, alpha0=alpha0, tau_v=tau_v, sigma=sigma,
                           prior_sd_alpha0=prior_sd_alpha0,
                           draws=draws, tune=tune, chains=chains, cores=cores, rng=rng)
            for res in tqdm(pool.imap_unordered(func, range(replicates)), total=replicates,
                            desc="MC replicates", position=progress_pos):
                results.append(res)
    else:
        for i in tqdm(range(replicates), desc="MC replicates", position=progress_pos):
            res = _replicate_worker(i, R, alpha0, tau_v, sigma, prior_sd_alpha0, draws, tune, chains, cores, rng)
            results.append(res)

    bf_values = np.array([r['BF10'] for r in results], dtype=float)
    valid_mask = np.isfinite(bf_values)
    n_failed = int(np.sum(~valid_mask))
    null_support = float(np.mean(bf_values[valid_mask] < BF_threshold)) if np.any(valid_mask) else float('nan')

    return dict(null_support=null_support, bf_values=bf_values, n_failed=n_failed, details=results)


# ---------------------------
# CSV output
# ---------------------------

def save_results_csv(outpath, details, append=False):
    base_cols = ['rep', 'BF10','BF10_kde','prior_d0','post_d0_norm','post_d0_kde','post_d0_used','seed',
                 'mean_alpha0','sd_alpha0','rhat_alpha0','ess_alpha0',
                 'n_divergences','acceptance_rate','n_posterior_samples',
                 'draws','tune','chains',
                 # sim params
                 'R','alpha0','tau_v','sigma']
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
    p = argparse.ArgumentParser(description="H2b Bayesian null-support simulation")
    p.add_argument('--output', '-o', required=True, help='CSV output path for replicate results')
    p.add_argument('--replicates', type=int, default=200)
    p.add_argument('--R', type=int, default=1298)
    p.add_argument('--alpha0', type=float, default=0.0)
    p.add_argument('--tau_v', type=float, default=0.2)
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--prior_sd_alpha0', type=float, default=1.0)
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--tune', type=int, default=2000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--cores', type=int, default=1)
    p.add_argument('--BF_threshold', type=float, default=0.1)
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

    logging.info(
        f'Starting H2b null-support simulation: replicates={args.replicates}, R={args.R}'
    )

    result = estimate_null_support(
        R=args.R,
        alpha0=args.alpha0,
        tau_v=args.tau_v,
        sigma=args.sigma,
        prior_sd_alpha0=args.prior_sd_alpha0,
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

    logging.info(f"Null-support (proportion BF10<{args.BF_threshold}): {result['null_support']}")
    logging.info(f"Replicates failed: {result['n_failed']} / {args.replicates}")

    save_results_csv(args.output, result['details'], append=args.append)
    logging.info(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
