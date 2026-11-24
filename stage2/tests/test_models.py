"""Tests of the execution layer (models.py) on small simulated data. Short chains: plumbing only, not inference."""
import numpy as np
import pytest

import models as M


def test_sim_files_pinned_and_mismatch_rejected(monkeypatch):
    for k in M.SIM_FILES:
        M.load_sim(k)
    monkeypatch.setitem(M.SIM_FILES, "H4", ("H4/H4.py", "0" * 64))
    with pytest.raises(RuntimeError):
        M.load_sim("H4")


def test_icc_and_delta_icc_known():
    assert M.icc(np.array([1.0]), np.array([1.0]), np.array([np.sqrt(2.0)]))[0] == pytest.approx(0.25)
    d = M.delta_icc(np.array([0.6, 0.5, 0.4, 0.7]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert d["p_delta_gt0"] == 0.5 and d["mean_delta"] == pytest.approx(0.05)


def test_interpret_rules():
    assert M.interpret("H1a", {"BF10": 50, "mean_alpha1": -0.1}) == "supports H"
    assert M.interpret("H1a", {"BF10": 50, "mean_alpha1": 0.1}) == "BF10>10, opposite direction"
    assert M.interpret("H4", {"BF10": 50, "mean_beta1": 0.1}) == "supports H"          # H4 is supported by a positive beta1
    assert M.interpret("H4", {"BF10": 0.05, "mean_beta1": 0.1}) == "supports null"
    assert M.interpret("H2a", {"BF10": 0.05}) == "supports H2 (no change in bias)"     # H2 is supported by BF10 < 1/10
    assert M.interpret("H2b", {"BF10": 3}) == "inconclusive"
    assert M.interpret("H3a", {"p_delta_gt0": 0.97}) == "supports H3"
    assert M.interpret("H3b", {"p_delta_gt0": 0.02}) == "supports null"


def test_fit_bf_h2a_plumbing():
    rng = np.random.default_rng(1)
    res, idata = M.fit_bf("H2a", dict(y=rng.normal(0.5, 1.0, 400)), seed=3, cores=1, draws=150, tune=150)
    assert res["target"] == "alpha0" and res["sample_kwargs"]["draws"] == 150
    assert idata.posterior["alpha0"].shape == (4, 150)
    assert 0.3 < res["mean_alpha0"] < 0.7 and res["BF10"] > 10
    rows = {r["param"]: r for r in M.summarise(idata)}
    assert set(rows) == {"alpha0", "sigma"} and rows["alpha0"]["mean"] == pytest.approx(res["mean_alpha0"])


def test_fit_h3_arm_plumbing_and_guards():
    rng = np.random.default_rng(2)
    S, T, R = 6, 4, 20
    subj = np.repeat(np.repeat(np.arange(S), T), R)
    scanner = np.repeat(np.tile(np.arange(T), S), R)
    r = np.tile(np.arange(R), S * T)
    y = rng.normal(0, 0.6, S)[subj] + rng.normal(0, 0.4, T)[scanner] + rng.normal(0, 0.5, subj.size)
    data = dict(y=y, subj=subj, scanner=scanner, r=r, cond=np.zeros_like(r))
    d, res, idata = M.fit_h3_arm(data, seed=5, cores=1, draws=100, tune=100)
    assert d["icc"].shape == (400,) and 0 < d["icc"].mean() < 1
    with pytest.raises(ValueError):
        M.fit_h3_arm(dict(data, cond=np.ones_like(r)), seed=5, cores=1, draws=10, tune=10)
    with pytest.raises(ValueError):
        M.fit_h3_arm(dict(data, r=r * 2), seed=5, cores=1, draws=10, tune=10)


def test_h3_suffstats_logp_identical_to_unchanged_model():
    """Sufficient-statistics H3 model: same log-density as the model built by the unchanged H3a.py code at random parameter values."""
    rng = np.random.default_rng(7)
    S, T, R = 5, 6, 30
    subj = np.repeat(np.repeat(np.arange(S), T), R)
    scanner = np.repeat(np.tile(np.arange(T), S), R)
    r = np.tile(np.arange(R), S * T)
    keep = rng.random(subj.size) > 0.1                       # unbalanced cells, as with missing units
    y = (rng.normal(0, 1, R)[r] + rng.normal(0, 0.6, S)[subj] + rng.normal(0, 0.4, T)[scanner]
         + rng.normal(0, 0.5, subj.size))[keep]
    data = dict(y=y, subj=subj[keep], scanner=scanner[keep], r=r[keep], cond=np.zeros(int(keep.sum()), int))
    mod = M.load_sim("H3")
    with M.capture_idata(mod) as st:
        mod.fit_and_compute_delta_icc(data, draws=10, tune=10, chains=1, cores=1, random_seed=1)
    m_sim, m_ss = st["model"], M.h3_suffstats_model(data)
    f_sim, f_ss = m_sim.compile_logp(), m_ss.compile_logp()
    for _ in range(20):
        pt_ = {"sd_u_log__": rng.normal(-0.5, 1), "z_u": rng.normal(size=S), "sd_w_log__": rng.normal(-0.5, 1),
               "z_w": rng.normal(size=T), "sigma_cond_log__": rng.normal(-0.5, 0.5, 2)}
        assert f_ss(pt_) == pytest.approx(f_sim(pt_), rel=1e-9, abs=1e-6)


def _h1a_data(rng, S=4, n=200):
    subj = np.repeat(np.arange(S), 2 * n)
    den = np.tile(np.repeat([0.0, 1.0], n), S)
    keep = rng.random(subj.size) > 0.15                        # unequal cells (missing units)
    y = rng.normal(0, 0.6, S)[subj] - 0.2 * den + rng.normal(0, 1, subj.size)
    return dict(y=y[keep], subj=subj[keep], den=den[keep])


def test_h1a_suffstats_logp_identical_to_unchanged_model():
    rng = np.random.default_rng(11)
    data = _h1a_data(rng)
    mod = M.load_sim("H1a")
    with M.capture_idata(mod) as st:
        mod.fit_and_compute_bf(data, draws=10, tune=10, chains=1, cores=1, random_seed=1)
    f_sim, f_ss = st["model"].compile_logp(), M.h1a_suffstats_model(data).compile_logp()
    for _ in range(20):
        p = {"alpha0": rng.normal(), "alpha1": rng.normal(), "sd_u_log__": rng.normal(-0.5, 1),
             "z_u": rng.normal(size=4), "sigma_log__": rng.normal(0, 0.5)}
        assert f_ss(p) == pytest.approx(f_sim(p), rel=1e-9, abs=1e-6)


def test_savage_dickey_reproduces_simulation_bf():
    rng = np.random.default_rng(12)
    data = _h1a_data(rng)
    res, idata = M.fit_bf("H1a", data, seed=2, cores=1, draws=100, tune=100)
    bf = M.savage_dickey_bf10(idata.posterior["alpha1"].values, 1.0)
    assert bf["BF10"] == pytest.approx(res["BF10"], rel=1e-12)
    assert bf["post_d0_kde"] == pytest.approx(res["post_d0_kde"], rel=1e-12)
    assert bf["mean"] == pytest.approx(res["mean_alpha1"], rel=1e-12)
