"""
models.py - Execution layer around the UNCHANGED Stage 1 simulation model code.

The fitting functions are imported from the Stage 1 simulations/ folder (each file pinned by SHA-256) and called as they are:
  H1a/H1b/H2a/H2b/H4  fit_and_compute_bf(data, draws=2000, tune=2000, chains=4, cores=...)
  H3a and H3b         H3a.py fit_and_compute_delta_icc(...) once per arm (H3b uses the H3a code)
Only the execution layer is new: parallel chains (cores), seeds, and a wrapper that keeps the InferenceData that
the unchanged code produces (pm.sample is wrapped, called with identical arguments, and its result stored), so that
posterior means, 95% intervals and convergence diagnostics can be reported for every parameter.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

import os  # noqa: E402

# Stage 1 simulation code: the simulations/ folder of the public repository (set RR_SIM_DIR to override)
_HERE = Path(__file__).resolve()
SIM_DIR = Path(os.environ["RR_SIM_DIR"]) if "RR_SIM_DIR" in os.environ else next(
    (p / "simulations" for p in _HERE.parents[1:3] if (p / "simulations").is_dir()), _HERE.parents[1] / "simulations")
SIM_FILES = {   # model -> (file, sha256 at 2026-10-01)
    "H1a": ("H1/H1a.py", "006a4b12f4b2fb9c312c6000117de9ac476360cc7459840f33d325344309cd55"),
    "H1b": ("H1/H1b.py", "613bc76e08d84309e5b3cd6925c3d58d5fb599fe5f235a203448519a8fa8ee21"),
    "H2a": ("H2/H2a.py", "3316b1bb53189c3352cc91f7538123155057a71b8e12ac3327b37611666049f7"),
    "H2b": ("H2/H2b.py", "c2aa8805828efd750ea6f227a555f69d3c5fd8dea7422e6de6445040fbe1aa5a"),
    "H3":  ("H3/H3.py", "d526a8ae04ceb15689f703f8e584b7051b13d347c9528ad855d642b2f9d4880b"),   # public name of H3a.py (same code)
    "H4":  ("H4/H4.py", "44435e413fb1fb04e503fd5b9eae085a13c83ba0a29eec23010f900451cb9187"),
}
DRAWS, TUNE, CHAINS = 2000, 2000, 4      # Section 3.4: 4 chains, 4000 iterations, warmup 2000
TARGET = {"H1a": "alpha1", "H1b": "alpha1", "H2a": "alpha0", "H2b": "alpha0", "H4": "beta1"}


def load_sim(key: str):
    rel, sha = SIM_FILES[key]
    path = SIM_DIR / rel
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        raise RuntimeError(f"{path} sha256 {got} != pinned {sha}; the simulation code must be used unchanged")
    spec = importlib.util.spec_from_file_location(f"sim_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def capture_idata(mod):
    """Wrap pm.sample as seen by the simulation module; the call and its arguments are unchanged."""
    pm = mod.pm
    real = pm.sample
    store = {}

    def sample(*args, **kw):
        store["model"] = pm.modelcontext(None)                 # the model the unchanged code built
        store["idata"] = real(*args, **kw)
        store["kwargs"] = {k: v for k, v in kw.items() if k != "step"}
        return store["idata"]

    pm.sample = sample
    try:
        yield store
    finally:
        pm.sample = real


def summarise(idata, var_names=None) -> list[dict]:
    """Posterior mean, SD, 2.5/97.5% quantiles, R-hat, bulk/tail ESS for scalar parameters (vectors elementwise)."""
    import arviz as az
    post = idata.posterior
    names = var_names or [v for v in post.data_vars if post[v].ndim <= 3 and not v.startswith("z_")]
    rhat = az.rhat(idata, var_names=names)
    essb = az.ess(idata, var_names=names, method="bulk")
    esst = az.ess(idata, var_names=names, method="tail")
    rows = []
    for v in names:
        x = post[v].values                                   # (chain, draw, ...)
        if x.ndim > 3 or (x.ndim == 3 and x.shape[2] > 12):  # skip long vectors (e.g. v_r) - summarised separately
            continue
        flat = x.reshape(x.shape[0] * x.shape[1], -1)
        rh, eb, et = (np.atleast_1d(d[v].values).ravel() for d in (rhat, essb, esst))
        for j in range(flat.shape[1]):
            name = v if flat.shape[1] == 1 else f"{v}[{j}]"
            q = np.quantile(flat[:, j], [0.025, 0.975])
            rows.append(dict(param=name, mean=float(flat[:, j].mean()), sd=float(flat[:, j].std(ddof=1)),
                             q2_5=float(q[0]), q97_5=float(q[1]), rhat=float(rh[j]), ess_bulk=float(eb[j]),
                             ess_tail=float(et[j])))
    return rows


def fit_bf(key: str, data: dict, seed: int, cores: int = 4, draws: int = DRAWS, tune: int = TUNE):
    """H1a/H1b/H2a/H2b/H4 with the unchanged fit_and_compute_bf. Returns (result dict, idata)."""
    mod = load_sim(key)
    with capture_idata(mod) as st:
        res = mod.fit_and_compute_bf(data, draws=draws, tune=tune, chains=CHAINS, cores=cores, random_seed=seed)
    res = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    res["target"] = TARGET[key]
    res["sample_kwargs"] = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in st["kwargs"].items()}
    return res, st["idata"]


def fit_h3_arm(data: dict, seed: int, cores: int = 4, draws: int = DRAWS, tune: int = TUNE):
    """One arm of H3: data has y, subj, scanner, r (contiguous 0..R-1), cond (all 0).

    Returns (draws dict for the ICC, result dict, idata)."""
    if np.any(np.asarray(data["cond"]) != 0):
        raise ValueError("fit_h3_arm takes one arm at a time with cond == 0")
    r = np.asarray(data["r"])
    if not np.array_equal(np.unique(r), np.arange(r.max() + 1)):
        raise ValueError("unit index r must be contiguous 0..R-1 (the simulation code uses R = max(r) + 1)")
    mod = load_sim("H3")
    with capture_idata(mod) as st:
        res = mod.fit_and_compute_delta_icc(data, draws=draws, tune=tune, chains=CHAINS, cores=cores,
                                            random_seed=seed)
    idata = st["idata"]
    post = idata.posterior
    n_obs, R = int(np.asarray(data["y"]).size), int(r.max()) + 1
    scale = np.sqrt(n_obs / max(n_obs - R, 1))                # the simulation code's centring correction
    d = dict(sd_u=post["sd_u"].values.reshape(-1), sd_w=post["sd_w"].values.reshape(-1),
             sigma=post["sigma_cond"].values.reshape(-1, 2)[:, 0] * scale)
    d["icc"] = icc(d["sd_u"], d["sd_w"], d["sigma"])
    # the simulation's own ICC for cond 0 must agree with ours (same arithmetic)
    if not np.isclose(res["mean_icc_nonden"], d["icc"].mean(), rtol=1e-10):
        raise RuntimeError("ICC recomputation disagrees with the simulation code")
    return d, {k: v for k, v in res.items() if k not in ("mean_icc_den", "mean_delta", "prob_delta_gt0")}, idata


def icc(sd_u, sd_w, sigma):
    """ICC = s_subject^2 / (s_subject^2 + s_scanner^2 + s_eps^2) per posterior draw (Section 3.4)."""
    vu, vw, ve = np.square(sd_u), np.square(sd_w), np.square(sigma)
    return vu / (vu + vw + ve)


def delta_icc(icc_den: np.ndarray, icc_non: np.ndarray) -> dict:
    """Delta ICC = ICC_den - ICC_nonden, draw i with draw i of the two independent posteriors."""
    if icc_den.shape != icc_non.shape:
        raise ValueError("arms must have the same number of draws")
    d = icc_den - icc_non
    q = np.quantile(d, [0.025, 0.975])
    return dict(p_delta_gt0=float(np.mean(d > 0)), mean_delta=float(d.mean()), q2_5=float(q[0]),
                q97_5=float(q[1]), mean_icc_den=float(icc_den.mean()), mean_icc_nonden=float(icc_non.mean()))


def interpret(key: str, res: dict) -> str:
    """Pre-registered criteria (Section 3.5, Table 3)."""
    if key in ("H1a", "H1b", "H4"):
        bf, m = res["BF10"], res[f"mean_{TARGET[key]}"]
        want_neg = key.startswith("H1")
        if bf > 10:
            return "supports H" if (m < 0 if want_neg else m > 0) else "BF10>10, opposite direction"
        return "supports null" if bf < 0.1 else "inconclusive"
    if key in ("H2a", "H2b"):
        bf = res["BF10"]
        return "supports H2 (no change in bias)" if bf < 0.1 else ("evidence against H2" if bf > 10 else "inconclusive")
    if key in ("H3a", "H3b"):
        p = res["p_delta_gt0"]
        return "supports H3" if p > 0.95 else ("supports null" if p < 0.05 else "inconclusive")
    raise KeyError(key)


# --------------------------------------------------------------------------------------------- sufficient-statistics form
def h3_suffstats_model(data: dict):
    """Sufficient-statistics form: the H3a.py model for ONE arm, with identical parameters,
    priors and names, but the Normal likelihood written on per-cell sufficient statistics. Within a subject x
    scanner cell the mean u_s + w_t is constant, so sum_i log N(y_i | mu_c, s) = -n_c/2 log(2 pi s^2)
    - (SS_c - 2 mu_c S_c + n_c mu_c^2) / (2 s^2): the same log-density for every parameter value
    (tests/test_models.py checks this against the model built by the unchanged code). Input y must already be centred
    exactly as H3a.py centres it (the centring is a no-op for build_inputs.py tables)."""
    import pymc as pm
    import pytensor.tensor as pt
    from inputs import cell_sufficient_stats, h3_centre
    y = np.asarray(data["y"], float)
    subj, scanner, r = (np.asarray(data[k]) for k in ("subj", "scanner", "r"))
    yc = h3_centre(y, r, np.zeros_like(r))
    S, T = int(subj.max()) + 1, int(scanner.max()) + 1
    cell = subj * T + scanner
    st = cell_sufficient_stats(yc, cell)
    used = np.flatnonzero(st["n"])
    cs, ct = used // T, used % T
    n, s1, s2 = st["n"][used].astype(float), st["s"][used], st["ss"][used]
    with pm.Model() as model:
        sd_u = pm.HalfNormal("sd_u", sigma=1.0)
        z_u = pm.Normal("z_u", mu=0.0, sigma=1.0, shape=S)
        u = pm.Deterministic("u", z_u * sd_u)
        sd_w = pm.HalfNormal("sd_w", sigma=1.0)
        z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=T)
        w = pm.Deterministic("w", z_w * sd_w)
        sigma_cond = pm.HalfNormal("sigma_cond", sigma=1.0, shape=2)
        mu = u[cs] + w[ct]
        sig = sigma_cond[0]
        pm.Potential("y_obs", pt.sum(-0.5 * n * pt.log(2 * np.pi * sig ** 2)
                                     - (s2 - 2 * mu * s1 + n * mu ** 2) / (2 * sig ** 2)))
    return model


def fit_h3_arm_suffstats(data: dict, seed: int, cores: int = 4, draws: int = DRAWS, tune: int = TUNE):
    """Sample h3_suffstats_model with H3a.py's sampler settings; returns the same draws dict as fit_h3_arm."""
    import pymc as pm
    r = np.asarray(data["r"])
    model = h3_suffstats_model(data)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=CHAINS, cores=cores, random_seed=seed,
                          step=pm.NUTS(target_accept=0.99, max_treedepth=12), progressbar=False)
    post = idata.posterior
    n_obs, R = int(np.asarray(data["y"]).size), int(r.max()) + 1
    scale = np.sqrt(n_obs / max(n_obs - R, 1))
    d = dict(sd_u=post["sd_u"].values.reshape(-1), sd_w=post["sd_w"].values.reshape(-1),
             sigma=post["sigma_cond"].values.reshape(-1, 2)[:, 0] * scale)
    d["icc"] = icc(d["sd_u"], d["sd_w"], d["sigma"])
    return d, dict(implementation="suffstats", n_obs=n_obs, R=R), idata


def savage_dickey_bf10(post: np.ndarray, prior_sd: float = 1.0) -> dict:
    """Savage-Dickey Bayes factor exactly as in the Stage 1 simulation scripts (fit_and_compute_bf): normal
    approximation of the posterior density at 0 (mean, SD ddof=1, SD floor 1e-8, density floor 1e-300) against the
    N(0, prior_sd) prior density; a KDE density is reported as a diagnostic. tests/test_models.py checks that it
    reproduces the simulation's BF10 exactly."""
    from scipy.stats import gaussian_kde, norm
    post = np.asarray(post).reshape(-1)
    m = float(np.mean(post))
    sd = max(float(np.std(post, ddof=1)) if post.size > 1 else 1e-8, 1e-8)
    d0 = float(norm.pdf(0.0, loc=m, scale=sd))
    try:
        kde = float(gaussian_kde(post, bw_method="scott").evaluate(0.0)[0])
    except Exception:  # noqa: BLE001 - as in the simulation code
        kde = 1e-300
    prior_d0 = max(norm.pdf(0.0, loc=0.0, scale=prior_sd), 1e-300)
    used = max(d0, 1e-300)
    return dict(BF10=float(np.exp(np.log(prior_d0) - np.log(used))), BF10_kde=kde, prior_d0=prior_d0,
                post_d0_norm=d0, post_d0_kde=kde, post_d0_used=used, mean=m, sd=sd)


def h1a_suffstats_model(data: dict):
    """H1a as fitted for the reported results: the H1a.py model with identical parameters, priors and names, and a
    Normal likelihood on per-cell (subject x arm) sufficient statistics instead of one term per voxel. The mean
    alpha0 + alpha1*den + u_s is constant within a cell, so the log-density is identical (tests/test_models.py);
    this only makes the fit of ~600,000 rows feasible."""
    import pymc as pm
    import pytensor.tensor as pt
    from inputs import cell_sufficient_stats
    y, subj, den = (np.asarray(data[k], float) for k in ("y", "subj", "den"))
    subj = subj.astype(int)
    if not np.isin(den, (0.0, 1.0)).all():
        raise ValueError("den must be 0/1")
    S = int(subj.max()) + 1
    cell = subj * 2 + den.astype(int)
    st = cell_sufficient_stats(y, cell)
    used = np.flatnonzero(st["n"])
    cs, cd = used // 2, (used % 2).astype(float)
    n, s1, s2 = st["n"][used].astype(float), st["s"][used], st["ss"][used]
    with pm.Model() as model:
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        alpha1 = pm.Normal("alpha1", mu=0.0, sigma=1.0)
        sd_u = pm.HalfNormal("sd_u", sigma=1.0)
        z_u = pm.Normal("z_u", mu=0.0, sigma=1.0, shape=S)
        u = pm.Deterministic("u", z_u * sd_u)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        mu = alpha0 + u[cs] + alpha1 * cd
        pm.Potential("y_obs", pt.sum(-0.5 * n * pt.log(2 * np.pi * sigma ** 2)
                                     - (s2 - 2 * mu * s1 + n * mu ** 2) / (2 * sigma ** 2)))
    return model


def fit_h1a_suffstats(data: dict, seed: int, cores: int = 4, draws: int = DRAWS, tune: int = TUNE):
    """Sample h1a_suffstats_model with H1a.py's sampler settings; result keys as fit_bf."""
    import arviz as az
    import pymc as pm
    with h1a_suffstats_model(data):
        idata = pm.sample(draws=draws, tune=tune, chains=CHAINS, cores=cores, random_seed=seed,
                          step=pm.NUTS(target_accept=0.99, max_treedepth=15), progressbar=False)
    post = idata.posterior["alpha1"].values.reshape(-1)
    bf = savage_dickey_bf10(post, 1.0)
    res = {k: v for k, v in bf.items() if k not in ("mean", "sd")}
    res.update(mean_alpha1=bf["mean"], sd_alpha1=bf["sd"],
               rhat_alpha1=float(az.rhat(idata, var_names=["alpha1"])["alpha1"].values),
               ess_alpha1=float(az.ess(idata, var_names=["alpha1"])["alpha1"].values),
               n_divergences=int(idata.sample_stats["diverging"].values.sum()),
               acceptance_rate=float(np.mean(idata.sample_stats["acceptance_rate"].values)),
               n_posterior_samples=int(post.size), draws=draws, tune=tune, chains=CHAINS,
               target="alpha1", implementation="suffstats (H1a)")
    return res, idata
