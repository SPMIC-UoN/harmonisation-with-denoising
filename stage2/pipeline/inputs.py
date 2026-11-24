"""
inputs.py - Pure functions that turn per-session metric values into the confirmatory model inputs
(Stage 1 Sections 3.3-3.4).

No file I/O here: every function takes numpy arrays and returns numpy arrays, so each one is unit-tested on
toy arrays with known answers (tests/test_inputs.py). Conventions:
  - values: float arrays shaped (n_sessions, n_units); NaN marks a missing value (`*_valid` maps).
  - arms: always handled together; nothing is standardised per arm.
"""
from __future__ import annotations

import numpy as np

ARMS = ("nondenoised", "denoised")       # condition c: 0 = non-denoised, 1 = denoised


# ----------------------------------------------------------------------------------------------- missingness
def valid_in_all_sessions(values: np.ndarray) -> np.ndarray:
    """(n_sessions, n_units) -> (n_units) bool: unit is finite in every session."""
    return np.isfinite(values).all(axis=0)


def roi_means(values: np.ndarray, labels: np.ndarray, keep: np.ndarray, roi_ids: np.ndarray):
    """ROI means per session over a fixed voxel set.

    values: (n_sessions, n_vox) voxel values in WM; labels: (n_vox) ROI label (0 = none);
    keep: (n_vox) bool voxels valid in every session of the subject; roi_ids: (n_roi) labels to report.
    Returns (means (n_sessions, n_roi) with NaN for ROIs with no kept voxel, n_kept (n_roi), n_dropped (n_roi)).
    """
    values = np.asarray(values, float)
    means = np.full((values.shape[0], roi_ids.size), np.nan)
    n_kept = np.zeros(roi_ids.size, int)
    n_drop = np.zeros(roi_ids.size, int)
    for j, k in enumerate(roi_ids):
        in_roi = labels == k
        sel = in_roi & keep
        n_kept[j] = int(sel.sum())
        n_drop[j] = int((in_roi & ~keep).sum())
        if n_kept[j]:
            means[:, j] = values[:, sel].mean(axis=1)
    return means, n_kept, n_drop


# ----------------------------------------------------------------------------------------------- per scanner
def average_by_group(values: np.ndarray, groups) -> tuple[np.ndarray, list]:
    """Average sessions sharing a group label (repeats on one scanner, Section 3.3).

    values: (n_sessions, n_units); groups: length n_sessions. Returns ((n_groups, n_units), sorted group labels).
    """
    groups = np.asarray(groups)
    labs = sorted(set(groups.tolist()))
    return np.stack([values[groups == g].mean(axis=0) for g in labs]), labs


def cov(values: np.ndarray) -> np.ndarray:
    """Coefficient of variation across axis 0: sample SD (ddof=1) / mean.

    NaN if any input is NaN or if the mean is <= 0 (CoV undefined)."""
    values = np.asarray(values, float)
    mu = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    out = np.full(mu.shape, np.nan)
    ok = np.isfinite(mu) & np.isfinite(sd) & (mu > 0)
    out[ok] = sd[ok] / mu[ok]
    return out


def missing_in_either_arm(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set an entry to NaN in both arrays if it is NaN in either (arms keep identical unit sets)."""
    a = np.array(a, float, copy=True)
    b = np.array(b, float, copy=True)
    bad = ~np.isfinite(a) | ~np.isfinite(b)
    a[bad] = np.nan
    b[bad] = np.nan
    return a, b


# ----------------------------------------------------------------------------------------------- scaling
def zscore_pooled(*arrays: np.ndarray) -> list[np.ndarray]:
    """z-standardise several arrays with ONE mean and SD computed over all their finite entries.

    Used with the two arms' CoV arrays (and, for H1, all subjects), never per arm. SD uses ddof=0."""
    flat = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    flat = flat[np.isfinite(flat)]
    m, s = flat.mean(), flat.std(ddof=0)
    if not s > 0:
        raise ValueError("zero or undefined pooled SD")
    return [(np.asarray(a, float) - m) / s for a in arrays]


def scale_by_sd_uncentred(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Divide by the SD (ddof=1) of the finite entries without centring (H2 Delta Bias)."""
    y = np.asarray(y, float)
    s = float(np.nanstd(y, ddof=1))
    if not s > 0:
        raise ValueError("zero or undefined SD")
    return y / s, s


# ----------------------------------------------------------------------------------------------- H2
def delta_bias(within_non, across_non, within_den, across_den) -> np.ndarray:
    """Delta Bias = (mu_WITHIN - mu_ACROSS)_den - (mu_WITHIN - mu_ACROSS)_nonden (Section 3.3).

    within_*: (n_repeats, n_units) the repeat-scanner sessions; across_*: (n_other, n_units) one session per
    other scanner. Missing (NaN) anywhere for a unit -> NaN."""
    b_non = np.asarray(within_non, float).mean(0) - np.asarray(across_non, float).mean(0)
    b_den = np.asarray(within_den, float).mean(0) - np.asarray(across_den, float).mean(0)
    return b_den - b_non


# ----------------------------------------------------------------------------------------------- H3
def h3_centre(y: np.ndarray, r: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """y - mu_r with mu_r the mean over the available rows of unit r within the same condition (Section 3.4). Identical arithmetic to the bincount centring in STAGE1-SIMULATIONS/H3/H3a.py."""
    y = np.asarray(y, float)
    out = np.empty_like(y)
    R = int(r.max()) + 1
    for c in np.unique(cond):
        m = cond == c
        sums = np.bincount(r[m], weights=y[m], minlength=R)
        counts = np.bincount(r[m], minlength=R)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        out[m] = y[m] - means[r[m]]
    return out


def h3_shared_scale(centred: np.ndarray) -> tuple[np.ndarray, float]:
    """Divide the centred values of BOTH arms by one constant, their pooled SD (ddof=1)."""
    s = float(np.std(centred, ddof=1))
    if not s > 0:
        raise ValueError("zero SD")
    return centred / s, s


def cell_sufficient_stats(y: np.ndarray, cell: np.ndarray) -> dict:
    """Per-cell n, sum and sum of squares (sufficient-statistics form of H3a)."""
    n_cells = int(cell.max()) + 1
    return dict(n=np.bincount(cell, minlength=n_cells), s=np.bincount(cell, weights=y, minlength=n_cells),
                ss=np.bincount(cell, weights=y * y, minlength=n_cells))


# ----------------------------------------------------------------------------------------------- H4
def h4_delta(cov_non: np.ndarray, cov_den: np.ndarray) -> np.ndarray:
    """Delta_r = z_den - z_nonden with z pooled over both arms of this pair (Section 3.4)."""
    a, b = missing_in_either_arm(cov_non, cov_den)
    z_non, z_den = zscore_pooled(a, b)
    return z_den - z_non


def h4_offres(abs_hz: np.ndarray) -> np.ndarray:
    """off_r = mean |Hz| over repeats and both arms (abs_hz: (n_maps, n_units))."""
    return np.asarray(abs_hz, float).mean(axis=0)


# ----------------------------------------------------------------------------------------------- MK validity
MK_RANGE = (0.0, 3.0)        # MK valid iff 0 < MK <= 3, in both arms


def range_validity(a: np.ndarray, b: np.ndarray, lo: float = MK_RANGE[0], hi: float = MK_RANGE[1]):
    """an entry is invalid in BOTH arms if, in either arm, it is <= lo or > hi (entries already NaN, are
    left as they are). a, b: same-shape arrays (non-denoised, denoised). Returns (a', b', trig_a, trig_b) where
    trig_x marks entries made invalid by arm x's value (both can be True)."""
    a = np.array(a, float, copy=True)
    b = np.array(b, float, copy=True)
    fin = np.isfinite(a) & np.isfinite(b)
    trig_a = fin & ((a <= lo) | (a > hi))
    trig_b = fin & ((b <= lo) | (b > hi))
    bad = trig_a | trig_b
    a[bad] = np.nan
    b[bad] = np.nan
    return a, b, trig_a, trig_b


def trigger_split(trig_a: np.ndarray, trig_b: np.ndarray) -> dict:
    """Counts of units excluded by arm a only, arm b only, or both (inputs: unit-level bool arrays)."""
    return dict(nondenoised_only=int((trig_a & ~trig_b).sum()), denoised_only=int((~trig_a & trig_b).sum()),
                both=int((trig_a & trig_b).sum()))
