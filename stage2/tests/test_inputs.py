"""Unit tests for inputs.py on toy arrays with known answers (run from the repository folder: python -m pytest tests)."""
import numpy as np
import pytest

import inputs as I

nan = np.nan


def test_valid_in_all_sessions():
    v = np.array([[1, 2, nan, 4], [1, nan, 3, 4], [1, 2, 3, 4.]])
    assert I.valid_in_all_sessions(v).tolist() == [True, False, False, True]


def test_average_by_group_repeats():
    # scanner B has 3 repeats: (2,4,6) -> 4 ; scanner A single 10 ; C single 1
    v = np.array([[10.], [2.], [4.], [6.], [1.]])
    out, labs = I.average_by_group(v, ["A", "B", "B", "B", "C"])
    assert labs == ["A", "B", "C"]
    assert out[:, 0].tolist() == [10., 4., 1.]


def test_cov_known_values():
    v = np.array([[1., 2.], [2., 2.], [3., 2.]])
    out = I.cov(v)
    assert out[0] == pytest.approx(1.0 / 2.0)          # sd(1,2,3; ddof=1) = 1, mean 2
    assert out[1] == 0.0


def test_cov_nonpositive_mean_and_nan():
    v = np.array([[-1., 1., 1.], [0.5, 1., nan], [0.4, 1., 1.]])
    out = I.cov(v)
    assert np.isnan(out[0])                            # mean <= 0 -> undefined
    assert out[1] == 0.0
    assert np.isnan(out[2])                            # any NaN -> NaN


def test_missing_in_either_arm():
    a, b = I.missing_in_either_arm(np.array([1., nan, 3.]), np.array([1., 2., nan]))
    assert np.isnan(a[1]) and np.isnan(a[2]) and np.isnan(b[1]) and np.isnan(b[2])
    assert a[0] == 1 and b[0] == 1


def test_zscore_pooled_not_per_arm():
    non, den = np.array([1., 2., 3.]), np.array([3., 4., 5.])
    zn, zd = I.zscore_pooled(non, den)
    allv = np.array([1, 2, 3, 3, 4, 5.])
    m, s = allv.mean(), allv.std()
    assert np.allclose(zn, (non - m) / s) and np.allclose(zd, (den - m) / s)
    # the arm difference survives pooled standardisation (it would vanish per arm)
    assert zd.mean() - zn.mean() == pytest.approx(2 / s)
    pooled = np.concatenate([zn, zd])
    assert pooled.mean() == pytest.approx(0) and pooled.std() == pytest.approx(1)


def test_zscore_pooled_ignores_nan():
    zn, zd = I.zscore_pooled(np.array([0., nan]), np.array([2., nan]))
    assert zn[0] == pytest.approx(-1) and zd[0] == pytest.approx(1) and np.isnan(zn[1])


def test_scale_by_sd_uncentred_keeps_mean_sign():
    y, s = I.scale_by_sd_uncentred(np.array([1., 2., 3.]))
    assert s == pytest.approx(1.0) and y.tolist() == [1., 2., 3.]
    y, s = I.scale_by_sd_uncentred(np.array([2., 4., 6.]))
    assert s == pytest.approx(2.0) and y.mean() == pytest.approx(2.0)


def test_delta_bias():
    within_non = np.array([[1.0, 5.], [3.0, 5.]])       # mean 2, 5
    across_non = np.array([[1.0, 4.], [1.0, 6.]])       # mean 1, 5   -> bias_non = 1, 0
    within_den = np.array([[2.0, 5.], [2.0, 5.]])       # mean 2, 5
    across_den = np.array([[0.5, 5.], [0.5, 4.]])       # mean .5, 4.5 -> bias_den = 1.5, .5
    d = I.delta_bias(within_non, across_non, within_den, across_den)
    assert d.tolist() == [0.5, 0.5]
    across_den[0, 1] = nan
    assert np.isnan(I.delta_bias(within_non, across_non, within_den, across_den)[1])


def test_roi_means_fixed_voxel_set():
    # 2 sessions, 5 voxels; ROI 1 = vox 0,1,2 ; ROI 2 = vox 3 ; ROI 3 = vox 4 (dropped)
    vals = np.array([[1., 2., 9., 4., 7.], [3., 4., nan, 6., nan]])
    labels = np.array([1, 1, 1, 2, 3])
    keep = I.valid_in_all_sessions(vals)                # vox 2 and 4 dropped in every session
    means, n_kept, n_drop = I.roi_means(vals, labels, keep, np.array([1, 2, 3, 4]))
    assert means[:, 0].tolist() == [1.5, 3.5]           # vox 2 excluded in BOTH sessions
    assert means[:, 1].tolist() == [4., 6.]
    assert np.isnan(means[:, 2]).all() and np.isnan(means[:, 3]).all()   # empty ROI -> missing
    assert n_kept.tolist() == [2, 1, 0, 0] and n_drop.tolist() == [1, 0, 1, 0]


def test_h3_centre_matches_simulation_code():
    rng = np.random.default_rng(0)
    r = np.tile(np.arange(4), 6)
    cond = np.repeat([0, 1], 12)
    y = rng.normal(size=24)
    out = I.h3_centre(y, r, cond)
    for c in (0, 1):
        for k in range(4):
            m = (cond == c) & (r == k)
            assert out[m].mean() == pytest.approx(0, abs=1e-12)
            assert np.allclose(out[m], y[m] - y[m].mean())


def test_h3_centre_unbalanced():
    y = np.array([1., 3., 10., 20.])
    r = np.array([0, 0, 1, 1])
    cond = np.array([0, 0, 0, 1])                       # unit 1 has one row per condition
    assert I.h3_centre(y, r, cond).tolist() == [-1., 1., 0., 0.]


def test_h3_shared_scale_same_constant_both_arms():
    c = np.array([-1., 1., -3., 3.])
    out, s = I.h3_shared_scale(c)
    assert s == pytest.approx(np.std(c, ddof=1))
    assert np.allclose(out, c / s)


def test_cell_sufficient_stats():
    st = I.cell_sufficient_stats(np.array([1., 2., 3., 4.]), np.array([0, 0, 1, 1]))
    assert st["n"].tolist() == [2, 2] and st["s"].tolist() == [3., 7.] and st["ss"].tolist() == [5., 25.]


def test_h4_delta_and_offres():
    non = np.array([0.1, 0.2, 0.3, nan])
    den = np.array([0.05, 0.1, 0.15, 0.2])
    d = I.h4_delta(non, den)
    allv = np.array([0.1, 0.2, 0.3, 0.05, 0.1, 0.15])
    s = allv.std()
    assert np.allclose(d[:3], (den[:3] - non[:3]) / s) and np.isnan(d[3])
    assert I.h4_offres(np.abs(np.array([[-2., 4.], [2., 0.]]))).tolist() == [2., 2.]


def test_range_validity_d56():
    a = np.array([1.0, -0.1, 0.0, 3.0, 3.5, nan, 1.0])
    b = np.array([1.2, 1.0, 1.0, 2.0, 4.0, nan, 3.01])
    a2, b2, ta, tb = I.range_validity(a, b)
    assert ta.tolist() == [False, True, True, False, True, False, False]
    assert tb.tolist() == [False, False, False, False, True, False, True]
    bad = [False, True, True, False, True, True, True]
    assert np.isnan(a2).tolist() == bad and np.isnan(b2).tolist() == bad      # identical in both arms
    assert a2[0] == 1.0 and a2[3] == 3.0                                     # 3 is valid (MK <= 3)
    assert I.trigger_split(ta, tb) == {"nondenoised_only": 2, "denoised_only": 1, "both": 1}
