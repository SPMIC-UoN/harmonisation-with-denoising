#!/usr/bin/env python3
"""
H3.py

Bayesian power simulation for H3 (voxel/ROI reliability).

Model per condition:
    y_{s,t,r} ~ Normal(u_s + w_t, sigma_cond)

Reliability is assessed via ICC:
    ICC = var_subject / (var_subject + var_scanner + sigma_cond^2)

Decision criterion:
    P(Delta_ICC > 0) > 0.95
"""

import argparse
import csv
import os
import sys
import logging
from functools import partial
import numpy as np
import pymc as pm
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
# Session layout
# ---------------------------

def build_sessions(S=10, T=6, n_repeat_scanners=5, repeats_total=6, seed=None):
    rng = np.random.default_rng(seed)
    scanners = np.arange(T)
    repeat_scanners = scanners[:n_repeat_scanners]
    non_repeat_scanners = scanners[n_repeat_scanners:]

    if n_repeat_scanners <= S:
        repeat_subjects = rng.choice(S, size=n_repeat_scanners, replace=False)
    else:
        repeat_subjects = rng.integers(0, S, size=n_repeat_scanners)

    rep_subject_map = {int(sc): int(subj) for sc, subj in zip(repeat_scanners, repeat_subjects)}

    subj_sessions = []
    scanner_sessions = []

    for sc in scanners:
        # One session for each subject
        for s in range(S):
            subj_sessions.append(s)
            scanner_sessions.append(sc)
        # Add extra repeats for the designated scanner
        if sc in rep_subject_map:
            rep_subj = rep_subject_map[sc]
            for _ in range(repeats_total - 1):
                subj_sessions.append(rep_subj)
                scanner_sessions.append(sc)

    subj_sessions = np.asarray(subj_sessions, dtype=int)
    scanner_sessions = np.asarray(scanner_sessions, dtype=int)
    return subj_sessions, scanner_sessions


# ---------------------------
# Simulation function
# ---------------------------

def simulate_data(R=6800, S=10, T=6,
                  sigma_mu=1.0, tau_u=0.6, tau_w=0.4,
                  sigma_nonden=0.6, sigma_den=0.3,
                  n_repeat_scanners=5, repeats_total=6,
                  seed=None):
    rng = np.random.default_rng(seed)

    subj_sess, scanner_sess = build_sessions(
        S=S, T=T, n_repeat_scanners=n_repeat_scanners, repeats_total=repeats_total, seed=seed
    )
    n_sessions = subj_sess.size

    mu_r = rng.normal(0.0, sigma_mu, size=R)
    u_s = rng.normal(0.0, tau_u, size=S)
    w_t = rng.normal(0.0, tau_w, size=T)

    base_sessions = u_s[subj_sess] + w_t[scanner_sess]
    base = base_sessions[:, None] + mu_r[None, :]

    y_n = base + rng.normal(0.0, sigma_nonden, size=base.shape)
    y_d = base + rng.normal(0.0, sigma_den, size=base.shape)

    r_idx = np.tile(np.arange(R, dtype=int), n_sessions)
    subj_idx = np.repeat(subj_sess, R)
    scanner_idx = np.repeat(scanner_sess, R)

    y = np.concatenate([y_n.reshape(-1), y_d.reshape(-1)])
    cond = np.concatenate([
        np.zeros(y_n.size, dtype=int),
        np.ones(y_d.size, dtype=int)
    ])

    r_idx = np.concatenate([r_idx, r_idx])
    subj_idx = np.concatenate([subj_idx, subj_idx])
    scanner_idx = np.concatenate([scanner_idx, scanner_idx])

    return dict(
        y=y,
        subj=subj_idx,
        scanner=scanner_idx,
        r=r_idx,
        cond=cond,
        n_sessions=n_sessions
    )


# ---------------------------
# Fit model and compute Delta ICC posterior
# ---------------------------

def fit_and_compute_delta_icc(data, draws=1000, tune=1000, chains=4, cores=1,
                              random_seed=None):
    y = data['y']
    subj = data['subj']
    scanner = data['scanner']
    r = data['r']
    cond = data['cond']

    S = int(subj.max()) + 1
    T = int(scanner.max()) + 1
    R = int(r.max()) + 1

    _check_array('y', y)
    _check_array('subj', subj)
    _check_array('scanner', scanner)
    _check_array('r', r)
    _check_array('cond', cond)

    # Center each voxel/ROI by condition-specific mean.
    y_centered = np.empty_like(y, dtype=float)
    for c in (0, 1):
        mask = cond == c
        sums = np.bincount(r[mask], weights=y[mask], minlength=R).astype(float)
        counts = np.bincount(r[mask], minlength=R).astype(float)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        y_centered[mask] = y[mask] - means[r[mask]]

    with pm.Model() as model:
        # Subject random intercept
        sd_u = pm.HalfNormal("sd_u", sigma=1.0)
        z_u = pm.Normal("z_u", mu=0.0, sigma=1.0, shape=S)
        u = pm.Deterministic("u", z_u * sd_u)

        # Scanner random intercept
        sd_w = pm.HalfNormal("sd_w", sigma=1.0)
        z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=T)
        w = pm.Deterministic("w", z_w * sd_w)

        # Condition-specific residual SD
        sigma_cond = pm.HalfNormal("sigma_cond", sigma=1.0, shape=2)

        mu = u[subj] + w[scanner]
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_cond[cond], observed=y_centered)

        nuts = pm.NUTS(target_accept=0.99, max_treedepth=12)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            step=nuts,
            progressbar=False
        )

    sd_u = idata.posterior["sd_u"].values.reshape(-1)
    sd_w = idata.posterior["sd_w"].values.reshape(-1)
    sigma_cond = idata.posterior["sigma_cond"].values.reshape(-1, 2)

    # Correct sigma_cond for centering loss of degrees of freedom.
    n_obs = y_centered.size
    scale = np.sqrt(n_obs / max(n_obs - R, 1))
    sigma_cond = sigma_cond * scale

    var_u = sd_u ** 2
    var_w = sd_w ** 2
    icc_n = var_u / (var_u + var_w + sigma_cond[:, 0] ** 2)
    icc_d = var_u / (var_u + var_w + sigma_cond[:, 1] ** 2)
    delta = icc_d - icc_n

    prob_delta_gt0 = float(np.mean(delta > 0.0))

    # Diagnostics
    try:
        rhat_sd_u = float(az.rhat(idata, var_names=["sd_u"]).to_array().values.flatten()[0])
    except Exception:
        rhat_sd_u = float('nan')
    try:
        rhat_sd_w = float(az.rhat(idata, var_names=["sd_w"]).to_array().values.flatten()[0])
    except Exception:
        rhat_sd_w = float('nan')
    try:
        ess_sd_u = float(az.ess(idata, var_names=["sd_u"]).to_array().values.flatten()[0])
    except Exception:
        ess_sd_u = float('nan')
    try:
        ess_sd_w = float(az.ess(idata, var_names=["sd_w"]).to_array().values.flatten()[0])
    except Exception:
        ess_sd_w = float('nan')
    try:
        ess_sigma_cond = az.ess(idata, var_names=["sigma_cond"]).to_array().values.flatten()
        ess_sigma_cond0 = float(ess_sigma_cond[0]) if ess_sigma_cond.size > 0 else float('nan')
        ess_sigma_cond1 = float(ess_sigma_cond[1]) if ess_sigma_cond.size > 1 else float('nan')
    except Exception:
        ess_sigma_cond0 = float('nan')
        ess_sigma_cond1 = float('nan')
    try:
        n_div = int(np.sum(idata.sample_stats["diverging"].values))
    except Exception:
        n_div = 0

    return dict(
        prob_delta_gt0=prob_delta_gt0,
        mean_delta=float(np.mean(delta)),
        mean_icc_nonden=float(np.mean(icc_n)),
        mean_icc_den=float(np.mean(icc_d)),
        rhat_sd_u=rhat_sd_u,
        rhat_sd_w=rhat_sd_w,
        ess_sd_u=ess_sd_u,
        ess_sd_w=ess_sd_w,
        ess_sigma_cond0=ess_sigma_cond0,
        ess_sigma_cond1=ess_sigma_cond1,
        n_divergences=n_div,
        draws=draws,
        tune=tune,
        chains=chains
    )


# ---------------------------
# Monte Carlo replicate worker
# ---------------------------

def _replicate_worker(i, R, S, T, sigma_mu, tau_u, tau_w, sigma_nonden, sigma_den,
                      n_repeat_scanners, repeats_total, draws, tune, chains, cores, rng):
    seed = int(rng.integers(1, 2**30))
    data = simulate_data(
        R=R, S=S, T=T,
        sigma_mu=sigma_mu, tau_u=tau_u, tau_w=tau_w,
        sigma_nonden=sigma_nonden, sigma_den=sigma_den,
        n_repeat_scanners=n_repeat_scanners, repeats_total=repeats_total,
        seed=seed
    )
    fit_res = fit_and_compute_delta_icc(
        data,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        random_seed=seed
    )

    out = dict(rep=i, seed=seed)
    out.update(fit_res)
    out.update(dict(
        R=R, S=S, T=T, sigma_mu=sigma_mu, tau_u=tau_u, tau_w=tau_w,
        sigma_nonden=sigma_nonden, sigma_den=sigma_den,
        n_repeat_scanners=n_repeat_scanners, repeats_total=repeats_total,
        n_sessions=data.get('n_sessions')
    ))
    return out


# ---------------------------
# Estimate power across replicates
# ---------------------------

def estimate_power_delta_icc(R=6800, S=10, T=6,
                             sigma_mu=1.0, tau_u=0.6, tau_w=0.4,
                             sigma_nonden=0.6, sigma_den=0.3,
                             n_repeat_scanners=5, repeats_total=6,
                             replicates=200,
                             draws=1000, tune=1000, chains=4, cores=1,
                             prob_threshold=0.95, rng_seed=1234,
                             parallel=False, n_workers=1,
                             progress_pos=0):
    rng = np.random.default_rng(rng_seed)
    results = []

    if parallel and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            func = partial(
                _replicate_worker,
                R=R, S=S, T=T,
                sigma_mu=sigma_mu, tau_u=tau_u, tau_w=tau_w,
                sigma_nonden=sigma_nonden, sigma_den=sigma_den,
                n_repeat_scanners=n_repeat_scanners, repeats_total=repeats_total,
                draws=draws, tune=tune, chains=chains, cores=cores, rng=rng
            )
            for res in tqdm(pool.imap_unordered(func, range(replicates)), total=replicates,
                            desc="MC replicates", position=progress_pos):
                results.append(res)
    else:
        for i in tqdm(range(replicates), desc="MC replicates", position=progress_pos):
            res = _replicate_worker(
                i, R, S, T, sigma_mu, tau_u, tau_w, sigma_nonden, sigma_den,
                n_repeat_scanners, repeats_total, draws, tune, chains, cores, rng
            )
            results.append(res)

    probs = np.array([r['prob_delta_gt0'] for r in results], dtype=float)
    valid_mask = np.isfinite(probs)
    n_failed = int(np.sum(~valid_mask))
    power_est = float(np.mean(probs[valid_mask] > prob_threshold)) if np.any(valid_mask) else float('nan')

    return dict(power=power_est, probs=probs, n_failed=n_failed, details=results)


# ---------------------------
# CSV output
# ---------------------------

def save_results_csv(outpath, details, append=False):
    base_cols = [
        'rep', 'prob_delta_gt0', 'mean_delta', 'mean_icc_nonden', 'mean_icc_den',
        'rhat_sd_u', 'rhat_sd_w', 'ess_sd_u', 'ess_sd_w', 'ess_sigma_cond0', 'ess_sigma_cond1',
        'n_divergences',
        'draws', 'tune', 'chains',
        'R', 'S', 'T', 'sigma_mu', 'tau_u', 'tau_w',
        'sigma_nonden', 'sigma_den',
        'n_repeat_scanners', 'repeats_total', 'n_sessions'
    ]
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
    p = argparse.ArgumentParser(description="H3 Bayesian power simulation (delta ICC)")
    p.add_argument('--output', '-o', required=True, help='CSV output path for replicate results')
    p.add_argument('--replicates', type=int, default=200)
    p.add_argument('--R', type=int, default=400)
    p.add_argument('--S', type=int, default=10)
    p.add_argument('--T', type=int, default=6)
    p.add_argument('--sigma_mu', type=float, default=1.0)
    p.add_argument('--tau_u', type=float, default=0.6)
    p.add_argument('--tau_w', type=float, default=0.7)
    p.add_argument('--sigma_nonden', type=float, default=0.3)
    p.add_argument('--sigma_den', type=float, default=0.15)
    p.add_argument('--n_repeat_scanners', type=int, default=5)
    p.add_argument('--repeats_total', type=int, default=6)
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--tune', type=int, default=2000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--cores', type=int, default=1)
    p.add_argument('--prob_threshold', type=float, default=0.95)
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
        'Starting H3 power simulation: replicates=%s, R=%s, S=%s, T=%s',
        args.replicates, args.R, args.S, args.T
    )

    result = estimate_power_delta_icc(
        R=args.R,
        S=args.S,
        T=args.T,
        sigma_mu=args.sigma_mu,
        tau_u=args.tau_u,
        tau_w=args.tau_w,
        sigma_nonden=args.sigma_nonden,
        sigma_den=args.sigma_den,
        n_repeat_scanners=args.n_repeat_scanners,
        repeats_total=args.repeats_total,
        replicates=args.replicates,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        prob_threshold=args.prob_threshold,
        rng_seed=args.rng_seed,
        parallel=args.parallel,
        n_workers=args.n_workers if args.parallel else 1
    )

    logging.info(
        'Estimated power (P(Delta_ICC>0)>%s): %s',
        args.prob_threshold, result['power']
    )
    logging.info('Replicates failed: %s / %s', result['n_failed'], args.replicates)

    save_results_csv(args.output, result['details'], append=args.append)
    logging.info('Results saved to %s', args.output)


if __name__ == '__main__':
    main()
