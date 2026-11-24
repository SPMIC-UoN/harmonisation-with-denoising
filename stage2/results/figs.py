"""
figs.py - Figures for the Stage 2 results notebooks (H1.ipynb ... H4.ipynb): registered results (Figures 2-5) and
exploratory results (Figures 6-8). Reads the outputs of the pipeline (<deriv>/analysis/{inputs, results,
results_extended_x4, descriptives}) and of the exploratory cells of the notebooks (<deriv>/analysis/exploratory).
"""
from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


C_NON, C_DEN = "#2a78d6", "#eb6834"


INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def read_gz(p: Path) -> dict:
    with gzip.open(p, "rt") as fh:
        r = csv.reader(fh)
        head = next(r)
        cols = list(zip(*r))
    return {h: np.array(c) for h, c in zip(head, cols)}


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def results(deriv: Path) -> dict:
    """(hypothesis, level, metric, pair) -> row of the frozen results table."""
    return {(r["hypothesis"], r["level"], r["metric"], r["pair"]): r
            for r in read_csv(deriv / "analysis" / "results" / "results.csv")}


def extended(deriv: Path) -> set:
    p = deriv / "analysis" / "results_extended_x4" / "results.csv"
    return {(r["hypothesis"], r["level"], r["metric"], r["pair"]) for r in read_csv(p)} if p.is_file() else set()


def fmt_bf(r: dict) -> str:
    """BF10 for display: values above 100 (including capped ones) are shown as '> 100'."""
    if r.get("BF10_report") == "capped" or float(r["BF10"]) > 100:
        return "> 100"
    x = float(r["BF10"])
    if x >= 0.01:
        return f"{x:.2g}"
    e = int(np.floor(np.log10(x)))
    return rf"${x / 10**e:.0f}\times10^{{{e}}}$"


def bf_label(r: dict) -> str:
    b = fmt_bf(r)
    return f"BF10 {b}" if b.startswith(">") else f"BF10 = {b}"


def _kde(x, lo, hi, n=200, max_pts=40000, seed=0):
    x = x[np.isfinite(x)]
    if x.size > max_pts:                                     # shape only; summary statistics use all points
        x = np.random.default_rng(seed).choice(x, max_pts, replace=False)
    g = np.linspace(lo, hi, n)
    return g, gaussian_kde(x)(g)


def half_violin(ax, pos, a, b, width=0.42, clip=(0.5, 99.5), lo=None, hi=None, stats=True):
    """Left half = a (non-denoised), right half = b (denoised); both halves share one density scale.
    stats: draw the median and interquartile range."""
    pooled = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    lo = np.percentile(pooled, clip[0]) if lo is None else lo
    hi = np.percentile(pooled, clip[1]) if hi is None else hi
    ga, da = _kde(a, lo, hi)
    gb, db = _kde(b, lo, hi)
    s = width / max(da.max(), db.max())
    for g, d, sign, c in ((ga, da, -1, C_NON), (gb, db, 1, C_DEN)):
        ax.fill_betweenx(g, pos, pos + sign * d * s, color=c, alpha=0.30, linewidth=0)
        ax.plot(pos + sign * d * s, g, color=c, linewidth=1.2)
    for x, sign, c in ((a, -1, C_NON), (b, 1, C_DEN)) if stats else ():
        q1, q2, q3 = np.nanpercentile(x, [25, 50, 75])
        ax.plot([pos + sign * 0.03] * 2, [q1, q3], color=c, linewidth=3, solid_capstyle="round")
        ax.plot([pos, pos + sign * 0.16], [q2, q2], color=INK, linewidth=1.5)
    ax.axvline(pos, color=INK3, linewidth=0.6, zorder=0)
    return lo, hi


DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                                                         "#f3b0af", "#e34948", "#9b2423"])


FSL_STD = Path(os.environ.get("FSLDIR", "/usr/local/fsl")) / "data" / "standard"


def _load(p: Path):
    import nibabel as nib
    im = nib.load(str(p))
    return np.asarray(im.dataobj, dtype=float), im.affine


def _axial(vol, aff, z_mm):
    k = int(round((z_mm - aff[2, 3]) / aff[2, 2]))
    return np.rot90(vol[:, :, k])                    # FSL MNI grid: left of image = subject right (radiological)


def _crop(under):
    rows, cols = np.where(under > 0)
    return slice(max(rows.min() - 2, 0), rows.max() + 3), slice(max(cols.min() - 2, 0), cols.max() + 3)


def _short(sc: str) -> str:
    return sc.split("_")[0]


def pair_info(deriv: Path) -> dict:
    """pair -> EDDY QC CNR (b = 1000) of the repeat scanner ('cnr_b1000_within') and relative to the subject's
    other scanners ('cnr_b1000_ratio')."""
    return {r["pair"]: r for r in read_csv(deriv / "analysis" / "exploratory" / "Ea_pair" / "Ea_pair_descriptive.csv")}


SC_MARK = {"Ingenia32": "o", "Premier21": "s", "Premier48": "D", "Achieva32": "v", "Prisma64": "^", "Prisma32": "P"}


WHITE = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}


ARM_NON, ARM_DEN = "#4a3aa7", "#eda100"


class _arms:
    """Temporarily set the arm colours used by figures.half_violin."""
    def __enter__(self):
        global C_NON, C_DEN
        self.old = C_NON, C_DEN
        C_NON, C_DEN = ARM_NON, ARM_DEN

    def __exit__(self, *a):
        global C_NON, C_DEN
        C_NON, C_DEN = self.old


def _arm_handles():
    return [Patch(facecolor=ARM_NON, alpha=0.5, edgecolor=ARM_NON, label="non-denoised"),
            Patch(facecolor=ARM_DEN, alpha=0.5, edgecolor=ARM_DEN, label="denoised"),
            Line2D([], [], color=INK, linewidth=1.5, label="median")]


def save(fig, out: Path, name: str):
    """Save PNG (300 dpi) and PDF on a white background; cropped to the content unless fig.no_crop is set
    (e.g. the thumbnail, whose aspect ratio is fixed)."""
    out.mkdir(parents=True, exist_ok=True)
    if getattr(fig, "caption", None):
        (out / f"{name}_caption.md").write_text(fig.caption + "\n")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches=None if getattr(fig, "no_crop", False) else "tight",
                    facecolor="white")


def _style(fs: float) -> dict:
    return {**WHITE, "font.size": fs, "axes.titlesize": fs + 2, "axes.labelsize": fs + 1,
            "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs}


def _violin_stats(ax, deriv, R, level, m, fs, tfs=None, bf=True):
    lv = "a" if level == "voxel" else "b"
    t = read_gz(deriv / "analysis" / "inputs" / f"h1{lv}_{m}.csv.gz")
    cov, den = t["cov"].astype(float) * 100, t["den"].astype(int)
    a, b = cov[den == 0], cov[den == 1]
    half_violin(ax, 0, a, b, clip=(0.5, 99), width=0.45)
    ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6, integer=True, steps=[1, 2, 5, 10]))
    r = R[("H1", level, m, "")]
    ma, mb = np.median(a), np.median(b)
    ax.text(0.5, -0.06, f"α1 = {float(r['mean']):+.2f} [{float(r['ci_low']):+.2f}, {float(r['ci_high']):+.2f}]\n"
            + (f"{bf_label(r)}\n" if bf else "") + f"median {ma:.1f}% → {mb:.1f}% ({100 * (mb - ma) / ma:+.0f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=tfs or fs - 2, linespacing=1.3)


def _map(fig, ax, cax, deriv, level, m, z, fs, clip_pct, cb_label=True, map_dir=None, label="ΔCoV (%)",
         orientation="vertical"):
    tag, grid = ("vox", "MNI2mm") if level == "voxel" else ("roi", "MNI1mm")
    under, ua = _load(FSL_STD / ("MNI152_T1_2mm_brain.nii.gz" if level == "voxel" else "MNI152_T1_1mm_brain.nii.gz"))
    vol = _load((map_dir or deriv / "analysis" / "descriptives" / "h1_maps") / f"{tag}_{m}_diff_{grid}.nii.gz")[0]
    dv = np.nanpercentile(np.abs(vol), clip_pct)
    u = _axial(under, ua, z)
    cr = _crop(u)
    ax.imshow(np.ma.masked_where(u[cr] <= 0, u[cr]), cmap="gray", vmin=0, vmax=1.9 * np.percentile(u[u > 0], 99),
              interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(_axial(vol, ua, z)[cr]), cmap=DIV, vmin=-dv, vmax=dv,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    if orientation == "horizontal":
        cb.ax.xaxis.set_major_locator(plt.MaxNLocator(2, symmetric=True))
    if cb_label:
        cb.set_label(label, fontsize=fs)
    cb.ax.tick_params(labelsize=fs - 1)
    cb.outline.set_visible(False)


SYS_C, UNIT_C = "#52514e", "#c3c2b7"      # variance of offsets / within-scanner variance (greys: separable by lightness under any CVD)


SC_COL = {"Ingenia32": "#0072B2", "Premier21": "#D55E00", "Premier48": "#009E73", "Prisma32": "#CC79A7",
          "Prisma64": "#56B4E9", "Achieva32": "#000000"}


QC_LAB = {"snr_b0": "SNR (b = 0)", "cnr_b1000": "CNR (b = 1000)", "cnr_b2000": "CNR (b = 2000)"}


X0 = 0.012                                  # left edge for panel headers (figure fraction)


def _leg_rows(p):
    n = len(p["legend"])
    return -(-n // p.get("legend_ncol", n))


def _compose(panels: list, fs: float, width: float = 16, left: float = 0.075, right: float = 0.98, gap: float = 0.25):
    """Stack panels vertically. Each panel dict: letter, title, sub (grey text, may contain newlines), legend
    (handles or None), legend_ncol, body (inches), top_pad / bottom_pad (inches for axis titles / tick labels),
    draw(fig, left, right, bottom, top). Headers sit at the figure's left edge."""
    def head_h(p):
        return 0.5 + (0.38 * (p["sub"].count("\n") + 1) if p.get("sub") else 0) + (0.45 * _leg_rows(p) if p.get("legend") else 0)
    H = 0.1 + sum(head_h(p) + p.get("top_pad", 0.3) + p["body"] + p.get("bottom_pad", 0.8) for p in panels) \
        + gap * (len(panels) - 1)
    fig = plt.figure(figsize=(width, H), facecolor="white")
    y = H - 0.1
    for k, p in enumerate(panels):
        fig.text(X0, y / H, (f"{p['letter']}) " if p.get("letter") else "") + p["title"], va="top", ha="left",
                 fontsize=fs + 6, fontweight="bold")
        y -= 0.5
        if p.get("sub"):
            fig.text(X0, y / H, p["sub"], va="top", ha="left", fontsize=fs - 1, color=INK2, linespacing=1.45)
            y -= 0.38 * (p["sub"].count("\n") + 1)
        if p.get("legend"):
            fig.legend(handles=p["legend"], loc="upper left", bbox_to_anchor=(X0 - 0.006, y / H + 0.05 / H),
                       ncol=p.get("legend_ncol", len(p["legend"])), fontsize=fs - 1, frameon=False,
                       columnspacing=1.4, handletextpad=0.4)
            y -= 0.45 * _leg_rows(p)
        y -= p.get("top_pad", 0.3)
        top = y / H
        y -= p["body"]
        p["draw"](fig, left, right, y / H, top)
        y -= p.get("bottom_pad", 0.8) + gap
    return fig


def _col_titles(fig, axes_labels, fs, dy_in=0.15):
    """Column titles left-aligned with the left edge of each column's axes including its y label."""
    r = fig.canvas.get_renderer()
    H = fig.get_figheight()
    for ax, lab in axes_labels:
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        fig.text(bb.x0, bb.y1 + dy_in / H, lab, fontsize=fs + 3, fontweight="bold", ha="left", va="bottom")


def _map_violin_grid(fig, left, right, bottom, top, metrics, violin, mapper, fs):
    """rows = voxel-wise / ROI-wise, columns = metrics; each cell: violin (with its stats), then the map with a
    horizontal colour bar underneath. violin(ax, level, m); mapper(ax, cax, level, m)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    wr = []
    for j in range(len(metrics)):
        wr += [1.0, 0.3, 1.35] + ([0.68] if j < len(metrics) - 1 else [])
    gs = GridSpec(2, len(wr), figure=fig, width_ratios=wr, wspace=0.0, hspace=0.42, left=left, right=right,
                  top=top, bottom=bottom)
    rows = []
    for i, level in enumerate(("voxel", "ROI")):
        for j, m in enumerate(metrics):
            c0 = 4 * j
            ax = fig.add_subplot(gs[i, c0], facecolor="white")
            violin(ax, level, m)
            ax.yaxis.label.set_size(fs + 2)
            sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[i, c0 + 2], height_ratios=[1, 0.05],
                                          width_ratios=[0.12, 1, 0.12], hspace=0.1, wspace=0)
            mapper(fig.add_subplot(sub[0, :], facecolor="white"), fig.add_subplot(sub[1, 1]), level, m)
            if j == 0:
                rows.append(ax)
    _col_titles(fig, [(rows[0], "Voxel-wise"), (rows[1], "ROI-wise")], fs)


def _h1_main(deriv, metrics, z, fs, clip_pct, layout="metric_cols"):
    R = results(deriv)

    def draw_metric_cols(fig, left, right, bottom, top):
        """rows = voxel-wise / ROI-wise, columns = metrics; each cell: map (colour bar below) | violin + stats."""
        def violin(ax, level, m):
            _violin_stats(ax, deriv, R, level, m, fs, tfs=fs - 3, bf=False)
            ax.set_ylabel(f"{m} CoV (%)")

        def mapper(ax, cax, level, m):
            _map(fig, ax, cax, deriv, level, m, z, fs, clip_pct, label=f"{m} ΔCoV (%)", orientation="horizontal")
        _map_violin_grid(fig, left, right, bottom, top, metrics, violin, mapper, fs)

    def draw(fig, left, right, bottom, top):
        if layout == "metric_cols":
            return draw_metric_cols(fig, left, right, bottom, top)
        gs = GridSpec(len(metrics), 7, figure=fig, width_ratios=[1.3, 1.25, 0.07, 0.95, 1.3, 1.25, 0.07],
                      wspace=0.08, hspace=0.62, left=left, right=right - 0.03, top=top, bottom=bottom)
        first = {}
        for i, m in enumerate(metrics):
            for c0, level in ((0, "voxel"), (4, "ROI")):
                ax = fig.add_subplot(gs[i, c0], facecolor="white")
                _violin_stats(ax, deriv, R, level, m, fs)
                ax.set_ylabel(f"{m} CoV (%)")
                _map(fig, fig.add_subplot(gs[i, c0 + 1], facecolor="white"), fig.add_subplot(gs[i, c0 + 2]),
                     deriv, level, m, z, fs, clip_pct)
                if i == 0:
                    first[level] = ax
        _col_titles(fig, [(first["voxel"], "Voxel-wise"), (first["ROI"], "ROI-wise")], fs)
    return draw


def _spread_parts(d):
    T = d.shape[0]
    off = d.mean(1)
    tot = (d ** 2).sum(0).mean() / (T - 1)
    return off, (off ** 2).sum() / (T - 1), tot - (off ** 2).sum() / (T - 1)


def _h1_example(z, scs, dn, dd, metric, fs):
    on, sn, un = _spread_parts(dn)
    od, sd, ud = _spread_parts(dd)
    T = dn.shape[0]

    def draw(fig, left, right, bottom, top):
        ax = fig.add_axes([left, bottom, right - left, top - bottom], facecolor="white")
        lo, hi = np.percentile(np.concatenate([dn.ravel(), dd.ravel()]), [2.5, 97.5])
        for t in range(T):
            half_violin(ax, t, dn[t], dd[t], width=0.4, lo=lo, hi=hi, stats=False)
            ax.plot(t - 0.08, on[t], "D", color=ARM_NON, markeredgecolor="white", markersize=11, zorder=5)
            ax.plot(t + 0.08, od[t], "D", color=ARM_DEN, markeredgecolor=INK, markersize=11, zorder=5)
        ax.axhline(0, color=INK, linewidth=1, zorder=0)
        ax.set_xticks(range(T), scs); ax.set_xlim(-0.6, T - 0.4); ax.grid(axis="x", visible=False)
        rng = hi - lo
        ax.set_ylim(lo - 0.05 * rng, hi + 0.46 * rng)
        ax.set_ylabel(f"{metric} deviation from\ncross-scanner mean (%)")
        yb = hi + 0.24 * rng                                           # text band above the violins
        # the split, both arms
        xm = (T - 1) / 2
        ax.text(xm, yb + 0.14 * rng, "mean CoV²  =  cross-scanner variance of ◆  +  within-scanner variance around ◆", ha="center", va="center",
                fontsize=fs - 1, fontweight="bold")
        for dy, (lab, c_, s_, u_) in zip((0.0, -0.12), (("non-denoised", ARM_NON, sn, un),
                                                         ("denoised", ARM_DEN, sd, ud))):
            ax.text(xm, yb + dy * rng, f"{lab}:  {s_ + u_:.1f}  =  {s_:.1f}  +  {u_:.1f} %²", ha="center",
                    va="center", fontsize=fs - 1, color=c_ if c_ == ARM_NON else "#9a6800", fontweight="bold")
    return draw


def _h1_bars(dec, fs):
    def draw(fig, left, right, bottom, top):
        gs = GridSpec(1, 2, figure=fig, wspace=0.18, left=left, right=right, top=top, bottom=bottom)
        axes = []
        for c, lvl in enumerate(("voxel", "ROI")):
            ax = fig.add_subplot(gs[0, c], facecolor="white")
            for j, m in enumerate(("FA", "MD", "MK")):
                for k, arm in enumerate(("nondenoised", "denoised")):
                    x = 3 * j + k
                    s_, u = dec.loc[(lvl, m, arm)]
                    ax.bar(x, s_, color=SYS_C, width=0.8, edgecolor="white", linewidth=1.5)
                    ax.bar(x, u, bottom=s_, color=UNIT_C, width=0.8, edgecolor="white", linewidth=1.5)
                    ax.annotate(f"{s_ + u:.0f}", (x, s_ + u), xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=fs - 3, color=INK)
                    ax.annotate(f"{s_:.1f}", (x, s_), xytext=(0, 3), textcoords="offset points", ha="center",
                                va="bottom", fontsize=fs - 3, color=SYS_C, fontweight="bold")
            ax.set_xticks([0, 1, 3, 4, 6, 7], ["non\nden", "den"] * 3, fontsize=fs - 3, va="top")
            for tl, c_ in zip(ax.get_xticklabels(), [ARM_NON, ARM_DEN] * 3):
                tl.set_color(c_); tl.set_fontweight("bold")
            for j, m in enumerate(("FA", "MD", "MK")):
                top_ = max(dec.loc[(lvl, m, a_)].sum() for a_ in ("nondenoised", "denoised"))
                ax.annotate(m, (3 * j + 0.5, top_), xytext=(0, 26), textcoords="offset points", ha="center",
                            va="bottom", fontsize=fs + 1, fontweight="bold")
            ax.set_xlim(-0.7, 7.7); ax.grid(axis="x", visible=False)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
            if c == 0:
                ax.set_ylabel("mean CoV² (%²),\naverage of 10 subjects")
            axes.append(ax)
        _col_titles(fig, [(axes[0], "Voxel-wise"), (axes[1], "ROI-wise")], fs)
    return draw


def _qc_col(deriv: Path, col: str) -> dict:
    """(subject, scanner) -> mean EDDY QC value (non-denoised) over that subject's sessions on that scanner."""
    out = {}
    for r in read_csv(deriv / "qc" / "eddy_qc_nondenoised.csv"):
        out.setdefault((r["subject"], r["scanner_label"]), []).append(float(r[col]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def _sc(sc: str):
    s_ = _short(sc)
    return SC_COL[s_], SC_MARK[s_]


def _sc_handles(scanners, ms=12):
    return [Line2D([], [], marker=_sc(s_)[1], color=_sc(s_)[0], linestyle="none", markersize=ms, label=_short(s_))
            for s_ in scanners]


def _h2_violins(deriv, metrics, pairs, info, nat, EXT, fs):
    t = read_gz(deriv / "analysis" / "descriptives" / "h2_bias_per_arm.csv.gz")

    def draw(fig, left, right, bottom, top):
        gs = GridSpec(len(metrics), 2, figure=fig, wspace=0.16, hspace=0.3, left=left, right=right, top=top,
                      bottom=bottom)
        first = []
        for i, m in enumerate(metrics):
            for c, level in enumerate(("voxel", "ROI")):
                ax = fig.add_subplot(gs[i, c], facecolor="white")
                sel = (t["metric"] == m) & (t["level"] == level)
                for k, p in enumerate(pairs):
                    s_ = sel & (t["pair"] == p)
                    mm = t["metric_mean"][s_].astype(float)[0]
                    half_violin(ax, k, 100 * t["bias_nonden"][s_].astype(float) / mm,
                                  100 * t["bias_den"][s_].astype(float) / mm, clip=(1, 99), width=0.42)
                    ax.text(k, 1.02, f"{nat[(p, m, level)]:+.1f}%" + ("†" if ("H2", level, m, p) in EXT else ""),
                            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=fs - 2,
                            fontweight="bold", color=INK)
                ax.axhline(0, color=INK2, linewidth=0.9, linestyle="--", zorder=0)
                ax.set_xlim(-0.6, len(pairs) - 0.4); ax.grid(axis="x", visible=False)
                ax.set_xticks(range(len(pairs)), [_short(info[p]["repeat_scanner"]) for p in pairs]
                              if i == len(metrics) - 1 else [], fontsize=fs - 2)
                if c == 0:
                    ax.set_ylabel(f"{m} bias (%)")
                if i == 0:
                    first.append(ax)
        _col_titles(fig, [(first[0], "Voxel-wise"), (first[1], "ROI-wise")], fs, dy_in=0.1)
    return draw


def _ols(x, y):
    """Least-squares line: slope, intercept, R^2, residuals."""
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return b[1], b[0], 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum(), r


def pair_cnr_fit(deriv: Path, qc: str = "cnr_b1000"):
    """Exploratory H2: Delta Bias (% of the pair's mean) per repeat pair regressed on the repeat scanner's relative QC
    (repeat scanner / the subject's other scanners), per metric and level (n = 5 pairs). Returns (fit table with R^2,
    slope and Delta Bias predicted at relative QC = 1; per-pair residuals)."""
    import pandas as pd
    nat = pd.read_csv(deriv / "analysis" / "descriptives" / "h2_bias_native.csv")
    info = pair_info(deriv)
    nat["rel_qc"] = [float(info[p][f"{qc}_ratio"]) for p in nat.pair]
    fits, res = [], []
    for (m, lv), g in nat.groupby(["metric", "level"], sort=False):
        x, y = g.rel_qc.to_numpy() - 1, g.delta_bias_pct_of_mean.to_numpy()
        slope, icpt, r2, r = _ols(x, y)
        fits.append(dict(metric=m, level=lv, pairs=len(x), r2=r2, slope_pct_per_unit=slope, dbias_at_equal_qc=icpt))
        res.append(g.assign(residual_pct=r)[["pair", "repeat_scanner", "metric", "level", "rel_qc",
                                            "delta_bias_pct_of_mean", "residual_pct"]])
    return pd.DataFrame(fits), pd.concat(res, ignore_index=True)


def _title2(ax, title, sub, fs):
    """Bold panel title with a smaller grey line underneath."""
    ax.set_title(title, loc="left", fontweight="bold", fontsize=fs + 1, pad=fs * 1.7)
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", ha="left",
                va="bottom", fontsize=fs - 1, color=INK2)


def _h2_pairs(metrics, pairs, info, nat, rq, qc, fs):
    def draw(fig, left, right, bottom, top):
        gs = GridSpec(1, len(metrics), figure=fig, wspace=0.28, left=left, right=right, top=top, bottom=bottom)
        for j, m in enumerate(metrics):
            ax = fig.add_subplot(gs[0, j], facecolor="white")
            for p in pairs:
                col, mk = _sc(info[p]["repeat_scanner"])
                for level, filled in (("voxel", True), ("ROI", False)):
                    ax.plot(rq[p], nat[(p, m, level)], mk, markersize=12, markeredgewidth=2, markeredgecolor=col,
                            markerfacecolor=col if filled else "white", linestyle="none", zorder=3)
            ax.axhline(0, color=INK2, linewidth=0.9)
            ax.axvline(1, color=INK3, linewidth=0.9, linestyle=":")
            xs = np.array([rq[p] for p in pairs])
            xx = np.linspace(xs.min(), xs.max(), 2)
            r2 = {}
            for level, ls in (("voxel", "-"), ("ROI", "--")):
                slope, icpt, r2[level], _ = _ols(xs, np.array([nat[(p, m, level)] for p in pairs]))
                ax.plot(xx, icpt + slope * xx, color=INK2, linewidth=1.4, linestyle=ls, zorder=2)
            ax.set_xlabel(f"relative {QC_LAB[qc]}")
            _title2(ax, m, f"R² = {r2['voxel']:.2f} (voxel) | {r2['ROI']:.2f} (ROI)", fs)
            if j == 0:
                ax.set_ylabel("ΔBias (% of mean)")
    return draw


def _h2_shift(metrics, shift, q, scs, qc, col, fs):
    def draw(fig, left, right, bottom, top):
        gs = GridSpec(1, len(metrics), figure=fig, wspace=0.28, left=left, right=right, top=top, bottom=bottom)
        for j, m in enumerate(metrics):
            ax = fig.add_subplot(gs[0, j], facecolor="white")
            rs = [r for r in shift if r["metric"] == m]
            x = np.array([q[(r["subject"], r["scanner"])] for r in rs])
            y = np.array([float(r[col]) for r in rs])
            for sc in scs:
                k = np.array([r["scanner"] == sc for r in rs])
                c_, mk = _sc(sc)
                ax.plot(x[k], y[k], mk, color=c_, markersize=7, alpha=0.45, linestyle="none", zorder=2)
                ax.plot(np.median(x[k]), np.median(y[k]), mk, color=c_, markersize=14, markeredgecolor=INK,
                        markeredgewidth=1.2, linestyle="none", zorder=5)
            ax.axhline(0, color=INK2, linewidth=0.9)
            slope, icpt, r2, _ = _ols(x, y)
            xx = np.linspace(x.min(), x.max(), 2)
            ax.plot(xx, icpt + slope * xx, color=INK2, linewidth=1.4, zorder=3)
            ax.set_xlabel(QC_LAB[qc])
            _title2(ax, m, f"R² = {r2:.2f}", fs)
            if j == 0:
                ax.set_ylabel("median value change\nwith denoising (%)")
    return draw


H3_DIR = ("analysis", "exploratory", "H3_unit_icc")


def _h3_posterior(deriv, metrics, fs):
    R = results(deriv)

    def draw(fig, left, right, bottom, top):
        gs = GridSpec(1, 2, figure=fig, wspace=0.16, left=left, right=right, top=top, bottom=bottom)
        axes = []
        for c, level in enumerate(("voxel", "ROI")):
            ax = fig.add_subplot(gs[0, c], facecolor="white")
            lv = "a" if level == "voxel" else "b"
            hi = 0
            for k, m in enumerate(metrics):
                base = deriv / "analysis" / "results" / f"H3{lv}_{m}"
                a = np.load(base.with_name(base.name + "_den0") / "icc_draws.npz")["icc"]
                b = np.load(base.with_name(base.name + "_den1") / "icc_draws.npz")["icc"]
                half_violin(ax, k, a, b, clip=(0, 99.9), lo=0, width=0.42)
                hi = max(hi, np.percentile(a, 99.9), np.percentile(b, 99.9))
                r = R[("H3", level, m, "")]
                ax.text(k, -0.04, f"ΔICC {float(r['mean']):+.3f}\n[{float(r['ci_low']):+.2f}, {float(r['ci_high']):+.2f}]"
                        f"\nP(ΔICC > 0) {float(r['p_delta_gt0']):.2f}" + (" †" if r["converged"] == "False" else ""), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=fs - 3, linespacing=1.3)
            ax.set_xticks(range(len(metrics)), metrics, fontsize=fs + 1, fontweight="bold")
            ax.tick_params(axis="x", pad=-2, length=0)
            ax.xaxis.set_ticks_position("top")
            ax.set_xlim(-0.6, len(metrics) - 0.4); ax.set_ylim(0, hi * 1.05); ax.grid(axis="x", visible=False)
            if c == 0:
                ax.set_ylabel("pre-registered ICC\n(posterior)")
            axes.append(ax)
        _col_titles(fig, [(axes[0], "Voxel-wise"), (axes[1], "ROI-wise")], fs)
    return draw


def _h3_units(deriv, metrics, z, fs, clip_pct):
    d = deriv.joinpath(*H3_DIR)

    def violin(ax, level, m):
        lv = "a" if level == "voxel" else "b"
        t = read_gz(d / f"unit_icc_{lv}_{m}.csv.gz")
        a, b = t["icc_nondenoised"].astype(float), t["icc_denoised"].astype(float)
        half_violin(ax, 0, a, b, clip=(0.5, 99.5), width=0.45)
        ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.grid(axis="x", visible=False)
        ax.axhline(0, color=INK3, linewidth=0.6)
        ax.set_ylabel(f"{m} ICC(2,1)")
        dicc = t["icc_diff"].astype(float)
        ax.text(0.5, -0.06, f"median {np.median(a):.2f} → {np.median(b):.2f}\nmedian ΔICC {np.median(dicc):+.3f}; "
                f"{100 * np.mean(dicc > 0):.0f}% higher",
                transform=ax.transAxes, ha="center", va="top", fontsize=fs - 3, linespacing=1.3)

    def mapper(ax, cax, level, m):
        _map(ax.figure, ax, cax, deriv, level, m, z, fs, clip_pct, map_dir=d / "wm_maps" if level == "voxel" else d,
             label=f"{m} ΔICC",
             orientation="horizontal")

    def draw(fig, left, right, bottom, top):
        _map_violin_grid(fig, left, right, bottom, top, metrics, violin, mapper, fs)
    return draw


def _h3_shares(deriv, metrics, fs):
    """Per-unit variance shares (subject = per-unit ICC, scanner, residual) as half-violins; dashed = posterior-mean
    shares of the pre-registered model (subject share = pre-registered ICC)."""
    d = deriv.joinpath(*H3_DIR)
    keys = (("share_subject", "subject"), ("share_scanner", "scanner"), ("share_residual", "residual"))

    def prereg(lv, m):
        out = []
        for dd in (0, 1):
            z = np.load(deriv / "analysis" / "results" / f"H3{lv}_{m}_den{dd}" / "icc_draws.npz")
            vu, vw, ve = z["sd_u"] ** 2, z["sd_w"] ** 2, z["sigma"] ** 2
            tot = vu + vw + ve
            out.append([float(np.mean(v / tot)) for v in (vu, vw, ve)])
        return out                                                      # [arm][component]

    def draw(fig, left, right, bottom, top):
        gs = GridSpec(2, len(metrics), figure=fig, wspace=0.12, hspace=0.42, left=left, right=right, top=top,
                      bottom=bottom)
        rows = []
        for i, (lv, level) in enumerate((("a", "voxel"), ("b", "ROI"))):
            for j, m in enumerate(metrics):
                ax = fig.add_subplot(gs[i, j], facecolor="white")
                t = read_gz(d / f"unit_icc_{lv}_{m}.csv.gz")
                pr = prereg(lv, m)
                for k, (key, _) in enumerate(keys):
                    half_violin(ax, k, t[f"{key}_nondenoised"].astype(float), t[f"{key}_denoised"].astype(float),
                                  clip=(0, 100), lo=0, hi=1, width=0.42)
                    for sign, arm, c_ in ((-1, 0, ARM_NON), (1, 1, ARM_DEN)):
                        ax.plot([k + sign * 0.04, k + sign * 0.44], [pr[arm][k]] * 2, color=c_, linewidth=2.6,
                                linestyle=(0, (2, 1.2)), zorder=6)
                ax.set_xticks(range(3), [lab for _, lab in keys], fontsize=fs - 2)
                ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
                if j == 0:
                    ax.set_ylabel("share of variance")
                    rows.append(ax)
                else:
                    ax.set_yticklabels([])
                if i == 0:
                    ax.set_title(m, loc="center", fontweight="bold", fontsize=fs + 1)
        _col_titles(fig, [(rows[0], "Voxel-wise"), (rows[1], "ROI-wise")], fs, dy_in=0.05)
    return draw


def _bfs(R, hyp, metrics, pair=""):
    return "; ".join(f"{lv}-wise " + ", ".join(f"{m} {fmt_bf(R[(hyp, lv, m, pair)])}" for m in metrics)
                     for lv in ("voxel", "ROI"))


def fig_h1(deriv: Path, metrics=("FA", "MD", "MK"), z: float = 8, fs: float = 16, clip_pct: float = 98,
           row_h: float = 3.8):
    """H1 (confirmatory): CoV half-violins with alpha1 and Delta CoV map, voxel-wise | ROI-wise."""
    R = results(deriv)
    cap = (
        "**Figure 2. Cross-scanner variability with and without MPPCA denoising (H1).** Coefficient of variation (CoV) "
        "of each unit across the six scanners (10 subjects), non-denoised (left half, violet) and denoised (right "
        "half, yellow); top row voxel-wise (white-matter sparse checkerboard voxels), bottom row ROI-wise (ROI means). Black "
        "line = median, bar = interquartile range; display clipped to the 0.5–99th percentile. Below each violin: α1 "
        "posterior mean [95% CrI] (denoising effect on the z-standardised CoV; negative = lower CoV) and the median "
        f"CoV without → with denoising. BF₁₀: {_bfs(R, 'H1', metrics)}. Maps: ΔCoV = denoised − non-denoised "
        f"(percentage points), mean over subjects, axial slice z = {z:+.0f} mm (MNI); blue = lower CoV with denoising. "
        "For display, the voxel-wise map shows all white-matter voxels (each subject's CoV warped to MNI space; voxels valid "
        "in at least 8 of the 10 subjects); the analysis used the sparse checkerboard voxels only.")
    with plt.rc_context(_style(fs)), _arms():
        fig = _compose([dict(letter=None, title="H1: Does Denoising Decrease Cross-Scanner CoV?",
                               legend=_arm_handles(), top_pad=0.5, body=row_h * 2, bottom_pad=1.1,
                               draw=_h1_main(deriv, metrics, z, fs, clip_pct))], fs)
    fig.caption = cap
    return fig


def fig_h2(deriv: Path, metrics=("FA", "MD", "MK"), qc: str = "cnr_b1000", fs: float = 16, row_h: float = 3.0):
    """H2 (confirmatory): per-unit bias half-violins per repeat pair, voxel-wise | ROI-wise; Delta Bias above."""
    R = results(deriv)
    EXT = {(r["hypothesis"], r["level"], r["metric"], r["pair"]) for r in
           read_csv(deriv / "analysis" / "results" / "results.csv") if r["converged"] == "False"}
    nat = {(r["pair"], r["metric"], r["level"]): float(r["delta_bias_pct_of_mean"])
           for r in read_csv(deriv / "analysis" / "descriptives" / "h2_bias_native.csv")}
    info = pair_info(deriv)
    pairs = sorted(info, key=lambda p: float(info[p][f"{qc}_ratio"]))
    n_bf = sum(fmt_bf(R[("H2", l, m, p)]) == "> 100" for l in ("voxel", "ROI") for m in metrics for p in pairs)
    cap = (
        "**Figure 3. Cross-scanner bias with and without MPPCA denoising (H2).** Five repeat pairs (one subject "
        "scanned six times on one scanner). Per-unit bias = mean over the six repeat sessions − mean over the same "
        "subject's other five scanners, as % of the pair's mean metric value; non-denoised (left half, violet) and "
        "denoised (right half, yellow); black line = median, bar = interquartile range. Numbers above each pair: "
        "ΔBias = denoised − non-denoised bias (mean over units, % of mean). "
        f"BF₁₀ > 100 in {n_bf} of {2 * len(metrics) * len(pairs)} tests (strong evidence that denoising changes bias). "
        f"† = ROI-wise fit that did not meet the convergence criteria ({sum(k[0] == 'H2' for k in EXT)} of 15). "
        f"Pairs ordered by the repeat scanner's {QC_LAB[qc]} relative to the subject's other scanners (lowest first).")
    with plt.rc_context(_style(fs)), _arms():
        fig = _compose([dict(letter=None, title="H2: Does Denoising Change Cross-Scanner Bias?",
                               legend=_arm_handles() + [Line2D([], [], marker="$†$", color=INK, markersize=12,
                                                                   linestyle="none", label="not converged")],
                               top_pad=0.7, body=row_h * len(metrics) + 0.6, bottom_pad=0.5,
                               draw=_h2_violins(deriv, metrics, pairs, info, nat, EXT, fs))], fs)
    fig.caption = cap
    return fig


def fig_h3(deriv: Path, metrics=("FA", "MD", "MK"), fs: float = 16):
    """H3 (confirmatory): posterior ICC per arm with Delta ICC and P(Delta ICC > 0)."""
    R = results(deriv)
    ps = [float(R[("H3", l, m, "")]["p_delta_gt0"]) for l in ("voxel", "ROI") for m in metrics]
    cap = (
        "**Figure 4. Cross-scanner reliability with and without MPPCA denoising (H3).** Posterior distribution of the "
        "pre-registered ICC = σ²subject / (σ²subject + σ²scanner + σ²ε), fitted separately to non-denoised (left half, "
        "violet) and denoised (right half, yellow) data (10 subjects × 6 scanners); voxel-wise (MNI space, "
        "sparse checkerboard voxels valid in all 85 sessions) and ROI-wise. Below each pair: ΔICC = ICC_den − ICC_nonden, "
        f"posterior mean [95% CrI], and P(ΔICC > 0) (range {min(ps):.2f}–{max(ps):.2f}; > 0.95 supports H3, < 0.05 "
        "supports the null; all six tests inconclusive). † = fits that did not meet the convergence criteria.")
    with plt.rc_context(_style(fs)), _arms():
        fig = _compose([dict(letter=None, title="H3: Does Denoising Increase Cross-Scanner Reliability (ICC)?",
                               legend=_arm_handles() + [Line2D([], [], marker="$†$", color=INK, markersize=12, linestyle="none", label="not converged")], top_pad=0.55, body=3.6, bottom_pad=1.3,
                               draw=_h3_posterior(deriv, metrics, fs))], fs)
    fig.caption = cap
    return fig


H4_MARK = {"supports H4": 1.0, "supports null": 0.0, "inconclusive": 0.35,
           "inconclusive (BF10 > 10, opposite sign)": 0.35}      # marker face opacity per verdict




def figx_h1(deriv: Path, metrics=("FA", "MD", "MK"), subject: str = "sub-16975", ex_metric: str = "FA", ex_level: str = "ROI",
            fs: float = 16):
    """Exploratory H1: worked example of the CoV split for one subject; the split averaged over subjects (X06, X11)."""
    import pandas as pd
    sp = deriv / "analysis" / "exploratory" / "H1_spread"
    zf = np.load(sp / f"example_{subject}.npz")
    scs = [_short(s_) for s_ in zf["scanners"]]
    dn, dd = zf[f"{ex_metric}_{ex_level}_nondenoised"], zf[f"{ex_metric}_{ex_level}_denoised"]
    _, sn, un = _spread_parts(dn)
    _, sd, ud = _spread_parts(dd)
    dec = pd.read_csv(sp / "decomposition.csv").groupby(["level", "metric", "arm"])[["systematic", "unit_specific"]].mean()
    cap = ("**Figure 6. Why cross-scanner CoV changes with denoising (exploratory).** "
           f"**(A)** Worked example ({ex_metric}, {ex_level}-wise, {subject}, {dn.shape[1]:,} units): each unit's deviation from "
           "its cross-scanner mean, per scanner; ◆ = scanner offset (mean over units). The per-unit CoV², averaged over units, "
           f"splits exactly into the cross-scanner variance of the ◆ ({sn:.1f} → {sd:.1f} %²) and the within-scanner variance of "
           f"the units around their ◆ ({un:.1f} → {ud:.1f} %²). "
           "**(B)** The same split for every metric and level, averaged over the 10 subjects; dark = cross-scanner variance of ◆ "
           "(value on the segment), light = within-scanner variance around ◆; total (mean CoV²) above each bar.")
    with plt.rc_context(_style(fs)), _arms():
        fig = _compose([
            dict(letter="A", title=f"How the Mean Cross-Scanner CoV² Splits (one subject: {ex_metric}, {ex_level}-wise)",
                 legend=_arm_handles()[:2] + [Line2D([], [], marker="D", color=INK3, linestyle="none", markersize=11,
                                                         label="scanner offset (mean over units)")],
                 top_pad=0.05, body=5.4, bottom_pad=0.6, draw=_h1_example(zf, scs, dn, dd, ex_metric, fs)),
            dict(letter="B", title="How the Mean Cross-Scanner CoV² Splits (averaged over all 10 subjects)",
                 legend=[Patch(facecolor=SYS_C, label="cross-scanner variance of ◆ (scanner offsets)"),
                         Patch(facecolor=UNIT_C, label="within-scanner variance around ◆")],
                 top_pad=0.45, body=4.4, bottom_pad=0.9, draw=_h1_bars(dec, fs))], fs)
    fig.caption = cap
    return fig


def cnr_residuals(deriv: Path, qc: str = "cnr_b1000", level: str = "ROI"):
    """Exploratory H2: how much of the denoising shift per subject x scanner (H1.ipynb) does scanner CNR explain?
    Per metric, OLS of the shift on CNR across the subject x scanner points; R^2 of CNR alone, scanner alone and
    CNR + scanner; F test of scanner beyond CNR (descriptive: points share subjects, so they are not independent).
    Returns (fit table, per-point residuals of the CNR-only fit in percentage points)."""
    import pandas as pd
    from scipy import stats
    col = "roi_shift_pct_median" if level == "ROI" else "vox_shift_pct_median"
    sh = pd.read_csv(deriv / "analysis" / "exploratory" / "H1_roi_diag" / "C_scanner_shift.csv")
    q = _qc_col(deriv, qc)
    sh["qc"] = [q[(a, b)] for a, b in zip(sh.subject, sh.scanner)]

    def ss(X, y):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
        return (r ** 2).sum(), r, b

    fits, res = [], []
    for m, g in sh.groupby("metric", sort=False):
        y, n = g[col].to_numpy(float), len(g)
        X1 = np.c_[np.ones(n), g.qc.to_numpy(float)]
        S = pd.get_dummies(g.scanner, drop_first=True).to_numpy(float)
        ss1, r1, b1 = ss(X1, y)
        ss2, _, _ = ss(np.c_[X1, S], y)
        ss0, _, _ = ss(np.c_[np.ones(n), S], y)
        tss = ((y - y.mean()) ** 2).sum()
        df1, df2 = S.shape[1], n - X1.shape[1] - S.shape[1]
        F = ((ss1 - ss2) / df1) / (ss2 / df2)
        fits.append(dict(metric=m, level=level, points=n, slope_pct_per_cnr=b1[1], r2_cnr=1 - ss1 / tss,
                         r2_scanner=1 - ss0 / tss, r2_cnr_scanner=1 - ss2 / tss, F_scanner_beyond_cnr=F,
                         p_descriptive=stats.f.sf(F, df1, df2)))
        res.append(g.assign(residual_pp=r1)[["subject", "scanner", "metric", "qc", col, "residual_pp"]])
    return pd.DataFrame(fits), pd.concat(res, ignore_index=True)


def figx_h2(deriv: Path, metrics=("FA", "MD", "MK"), qc: str = "cnr_b1000", shift_level: str = "ROI", fs: float = 16):
    """Exploratory H2: denoising shift per subject x scanner vs QC and Delta Bias per pair vs relative QC, each with
    its least-squares fit and R^2."""
    nat = {(r["pair"], r["metric"], r["level"]): float(r["delta_bias_pct_of_mean"])
           for r in read_csv(deriv / "analysis" / "descriptives" / "h2_bias_native.csv")}
    info = pair_info(deriv)
    rq = {p: float(info[p][f"{qc}_ratio"]) for p in info}
    pairs = sorted(info, key=lambda p: rq[p])
    q = _qc_col(deriv, qc)
    shift = read_csv(deriv / "analysis" / "exploratory" / "H1_roi_diag" / "C_scanner_shift.csv")
    scs = sorted({r["scanner"] for r in shift}, key=lambda s_: np.median([v for k, v in q.items() if k[1] == s_]))
    rep = [info[p]["repeat_scanner"] for p in pairs]
    col, unit = ("roi_shift_pct_median", "ROI") if shift_level == "ROI" else ("vox_shift_pct_median", "checkerboard-voxel")
    n_pts = sum(r["metric"] == metrics[0] for r in shift)
    cap = ("**Figure 7. Relationship between the effect of denoising on metric values and scanner CNR (exploratory).** "
           f"**(A)** Denoising shift per subject × scanner = median relative change of {unit} values (denoised vs non-denoised) vs "
           f"that scanner's {QC_LAB[qc]} (EDDY QC, non-denoised; mean over the subject's sessions on it); large markers = scanner "
           f"medians; line = least-squares fit over the {n_pts} subject × scanner points, with R² under each metric. Descriptive: "
           "points are not independent (they share subjects). "
           f"**(B)** ΔBias per repeat pair vs relative {QC_LAB[qc]} = repeat scanner (6 repeat sessions) ÷ mean of the subject's "
           "other 5 scanners; filled = voxel-wise, open = ROI-wise; lines = least-squares fits "
           "(solid = voxel-wise, dashed = ROI-wise), with R² under each metric. n = 5 pairs, in two CNR clusters that coincide with vendor.")
    fit_line = [Line2D([], [], color=INK2, linewidth=1.4, label="fit, voxel-wise"),
                Line2D([], [], color=INK2, linewidth=1.4, linestyle="--", label="fit, ROI-wise")]
    with plt.rc_context(_style(fs)):
        fig = _compose([
            dict(letter="A", title=f"Denoising Shift of Values vs Scanner {QC_LAB[qc].split(' (')[0]}",
                 legend=_sc_handles(scs) + [
                     Line2D([], [], marker="o", color=INK3, alpha=0.45, linestyle="none", markersize=7, label="subject × scanner"),
                     Line2D([], [], marker="o", color=INK3, markeredgecolor=INK, markeredgewidth=1.2, linestyle="none",
                              markersize=14, label="scanner median"),
                     Line2D([], [], color=INK2, linewidth=1.4, label="linear fit")], legend_ncol=5,
                 top_pad=0.75, body=4.2, bottom_pad=0.9, draw=_h2_shift(metrics, shift, q, scs, qc, col, fs)),
            dict(letter="B", title=f"ΔBias per Repeat Pair vs Relative {QC_LAB[qc].split(' (')[0]}",
                 legend=_sc_handles(rep) + [
                     Line2D([], [], marker="o", color=INK, linestyle="none", markersize=11, label="voxel-wise"),
                     Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=2,
                              linestyle="none", markersize=11, label="ROI-wise")] + fit_line, legend_ncol=5,
                 top_pad=0.75, body=4.2, bottom_pad=0.9, draw=_h2_pairs(metrics, pairs, info, nat, rq, qc, fs))], fs)
    fig.caption = cap
    return fig


def figx_h3(deriv: Path, metrics=("FA", "MD", "MK"), z: float = 8, fs: float = 16, clip_pct: float = 98, row_h: float = 3.8):
    """Exploratory H3: per-unit ICC(2,1) and Delta ICC maps; per-unit variance shares (X07, X08, X10)."""
    cap = ("**Figure 8. Reliability of each voxel and ROI (exploratory).** "
           "**(A)** ICC(2,1) computed separately for each voxel/ROI from the 10 subject × 6 scanner table; black line = median, bar = "
           "interquartile range. Below: median ICC without → with denoising, the median per-unit ΔICC and the percentage of units "
           f"with a higher ICC after denoising. Maps: ΔICC = denoised − non-denoised, axial slice z = {z:+.0f} mm (MNI); red = higher ICC with denoising. "
           "The violins and numbers use the sparse checkerboard voxels valid in all 85 sessions; for display, the voxel-wise map shows all "
           "white-matter voxels, each computed from the subjects with a valid value on all six scanners (at least 8 of 10). "
           "**(B)** Share of each unit's variance due to subjects, scanners and the residual (two-way random-effects ANOVA per "
           "voxel/ROI; the subject share is the per-unit ICC in A), non-denoised (left half) and denoised (right half). Dashed lines: "
           "the same shares in the pre-registered model (posterior means), whose subject and scanner effects are shared by all units, "
           "so unit-specific differences fall into its residual; its subject share is the pre-registered ICC in Figure 4.")
    with plt.rc_context(_style(fs)), _arms():
        fig = _compose([
            dict(letter="A", title="Reliability of Each Voxel / ROI (per-unit ICC)", legend=_arm_handles(), top_pad=0.5,
                 body=row_h * 2, bottom_pad=0.95, draw=_h3_units(deriv, metrics, z, fs, clip_pct)),
            dict(letter="B", title="Variance Components per Voxel / ROI",
                 legend=_arm_handles() + [Line2D([], [], color=INK2, linewidth=2.6, linestyle=(0, (2, 1.2)),
                                                     label="pre-registered model (posterior mean)")],
                 top_pad=0.75, body=7.0, bottom_pad=0.6, draw=_h3_shares(deriv, metrics, fs))], fs)
    fig.caption = cap
    return fig


plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titleweight": "bold", "legend.frameon": False})


def hyp_label(hyp, level, metric):
    """e.g. 'H1a – FA voxel-wise' (a = voxel-wise, b = ROI-wise)."""
    return f"{hyp}{'a' if level == 'voxel' else 'b'} – {metric} {'voxel' if level == 'voxel' else 'ROI'}-wise"


def results_table(deriv: Path, hyp: str, rerun: bool = False):
    """Registered results for one hypothesis as a table: posterior mean and 95% CrI of the target parameter (H3:
    ICC per arm and Delta ICC), BF10 (H3: P(Delta ICC > 0)), convergence and verdict. With rerun=True, the exploratory
    re-runs with 4x draws (fits that did not meet the convergence criteria) are shown on a second row."""
    import pandas as pd
    a = deriv / "analysis"
    runs = [("", pd.read_csv(a / "results" / "results.csv"))]
    if rerun and (a / "results_extended_x4" / "results.csv").is_file():
        runs.append(("x4 draws", pd.read_csv(a / "results_extended_x4" / "results.csv")))
    def bf(r):
        if r["BF10_capped"] or r["BF10"] > 100:
            return ">100"
        return f"{r['BF10']:.1f}" if r["BF10"] >= 1 else f"{r['BF10']:.3f}"

    rows = []
    for run, t in runs:
        for _, r in t[t["hypothesis"] == hyp].iterrows():
            row = {"Hypothesis": hyp_label(hyp, r["level"], r["metric"]),
                   "pair": r["pair"] if isinstance(r["pair"], str) else "", "run": run}
            q = 5 if hyp == "H4" else 3                     # H4 slopes are per Hz
            ci = f"{r['mean']:+.{q}f} [{r['ci_low']:+.{q}f}, {r['ci_high']:+.{q}f}]"
            if hyp == "H3":
                row.update({"ICC non-denoised": f"{r['icc_nonden']:.3f} [{r['icc_nonden_low']:.3f}, {r['icc_nonden_high']:.3f}]",
                            "ICC denoised": f"{r['icc_den']:.3f} [{r['icc_den_low']:.3f}, {r['icc_den_high']:.3f}]",
                            "ΔICC [95% CrI]": ci, "P(ΔICC > 0)": f"{r['p_delta_gt0']:.3f}"})
            else:
                row.update({f"{r['target']} [95% CrI]": ci, "BF10": bf(r)})
            row.update({"converged": "yes" if r["converged"] else "no", "verdict": r["verdict"]})
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["Hypothesis", "pair", "run"], kind="stable")
    out = out.drop(columns=[c for c in ("pair", "run") if (out[c] == "").all()])
    return out.reset_index(drop=True)


def _h4_setup(deriv):
    info = pair_info(deriv)
    pairs = sorted(info, key=lambda p: float(info[p]["cnr_b1000_within"]))
    nat = {(r["pair"], r["metric"], r["level"]): {k: float(v) for k, v in r.items() if k.startswith(("beta", "off", "cov"))}
           for r in read_csv(deriv / "analysis" / "descriptives" / "h4_native.csv")}
    return info, pairs, nat


OFF_CMAP = LinearSegmentedColormap.from_list("off_seq", ["#f6f1e7", "#f3c76b", "#e07b39", "#a83a2a", "#4a1022"])


def _h4_maps(deriv, pairs, info, nat, metric, z, fs, clip_pct):
    """Per repeat pair (columns, ordered by CNR): |off-resonance| (top) and the change in scan-rescan CoV of `metric`
    with denoising (bottom), all WM voxels in MNI space (descriptives/h4_maps)."""
    d = deriv / "analysis" / "descriptives" / "h4_maps"
    under, ua = _load(FSL_STD / "MNI152_T1_2mm_brain.nii.gz")
    u = _axial(under, ua, z)
    cr = _crop(u)
    off = {p: _axial(_load(d / f"vox_{p}_off_MNI2mm.nii.gz")[0], ua, z)[cr] for p in pairs}
    dc = {p: _axial(_load(d / f"vox_{p}_dcov_{metric}_MNI2mm.nii.gz")[0], ua, z)[cr] for p in pairs}
    omax = np.nanpercentile(np.concatenate([v[np.isfinite(v)] for v in off.values()]), clip_pct)
    dv = np.nanpercentile(np.abs(np.concatenate([v[np.isfinite(v)] for v in dc.values()])), clip_pct)
    rows = [(off, OFF_CMAP, 0, omax, "|off-resonance|\n(Hz)"), (dc, DIV, -dv, dv, f"{metric} ΔCoV\n(pp)")]

    def draw(fig, left, right, bottom, top):
        gs = GridSpec(len(rows), len(pairs) + 1, figure=fig, width_ratios=[1] * len(pairs) + [0.05], wspace=0.04,
                      hspace=0.12, left=left - 0.04, right=right - 0.04, top=top, bottom=bottom)
        for j, p in enumerate(pairs):
            for i, (maps, cmap, lo, hi, lab) in enumerate(rows):
                vol = maps[p]
                ax = fig.add_subplot(gs[i, j], facecolor="white")
                ax.imshow(np.ma.masked_where(u[cr] <= 0, u[cr]), cmap="gray", vmin=0,
                          vmax=1.9 * np.percentile(u[u > 0], 99), interpolation="nearest")
                im = ax.imshow(np.ma.masked_where(~np.isfinite(vol) | (u[cr] <= 0), vol), cmap=cmap, vmin=lo, vmax=hi,
                               interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if i == 0:
                    col, _ = _sc(info[p]["repeat_scanner"])
                    ax.set_title(_short(info[p]["repeat_scanner"]), fontsize=fs + 1, fontweight="bold", color=col)
                if j == len(pairs) - 1:
                    cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, -1]))
                    cb.set_label(lab, fontsize=fs - 1)
                    cb.ax.tick_params(labelsize=fs - 2); cb.outline.set_visible(False)
    return draw


def _h4_scale(deriv, metrics, pairs, info, nat, R, fs):
    """beta0 (bars: Delta CoV at zero off-resonance) and beta1 (markers: change in Delta CoV per SD of |off|), both in
    CoV percentage points, per pair; columns voxel | ROI; beta1 fill = registered verdict."""
    def draw(fig, left, right, bottom, top):
        gs = GridSpec(1, 2, figure=fig, wspace=0.14, left=left, right=right, top=top, bottom=bottom)
        n, w = len(pairs), 0.15
        cols = []
        for c, level in enumerate(("voxel", "ROI")):
            ax = fig.add_subplot(gs[0, c], facecolor="white")
            for j, m in enumerate(metrics):
                for k, p in enumerate(pairs):
                    x = j + (k - (n - 1) / 2) * w
                    col, mk = _sc(info[p]["repeat_scanner"])
                    d = nat[(p, m, level)]
                    sd = d["cov_pooled_sd_pct"]
                    ax.bar(x, d["beta0_pp"], width=w * 0.85, color=mcolors.to_rgba(col, 0.35), edgecolor=col,
                           linewidth=1.0, zorder=2)
                    r = R[("H4", level, m, p)]
                    b1, lo, hi = (float(r[q]) * sd * d["off_hz_sd"] for q in ("mean", "ci_low", "ci_high"))   # per Hz -> per SD of |off|
                    fill = H4_MARK[r["verdict"]]
                    ax.plot([x, x], [lo, hi], color=col, linewidth=2, zorder=4)
                    ax.plot(x, b1, mk, color=col, markersize=9, markeredgewidth=1.6, markeredgecolor=col,
                            markerfacecolor=mcolors.to_rgba(col, fill) if fill else "white", linestyle="none", zorder=5)
                if j:
                    ax.axvline(j - 0.5, color=GRID, linewidth=1)
            ax.axhline(0, color=INK2, linewidth=0.9)
            ax.set_xlim(-0.5, len(metrics) - 0.5); ax.grid(axis="x", visible=False)
            ax.set_xticks(range(len(metrics)), metrics, fontsize=fs + 1, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Change in scan–rescan CoV\n(percentage points)")
            cols.append(ax)
        _col_titles(fig, [(cols[0], "Voxel-wise"), (cols[1], "ROI-wise")], fs, dy_in=0.1)
    return draw


def _h4_bins(deriv, metrics, pairs, info, nat, fs, nbins=10):
    """Voxel-wise (registered H4a units): mean Delta CoV (pp) in deciles of |off-resonance| per pair (markers + thin
    line) and the fitted line beta0 + beta1 * off (dashed), one column per metric."""
    import pandas as pd

    def draw(fig, left, right, bottom, top):
        gs = GridSpec(1, len(metrics), figure=fig, wspace=0.22, left=left, right=right, top=top, bottom=bottom)
        for j, m in enumerate(metrics):
            ax = fig.add_subplot(gs[0, j], facecolor="white")
            for p in pairs:
                col, mk = _sc(info[p]["repeat_scanner"])
                t = pd.read_csv(deriv / "analysis" / "inputs" / f"h4a_{p}_{m}.csv.gz",
                                usecols=["cov_nonden", "cov_den", "off_hz"])
                g = pd.DataFrame({"off": t.off_hz, "dc": 100 * (t.cov_den - t.cov_nonden),
                                  "b": pd.qcut(t.off_hz, nbins, labels=False, duplicates="drop")}).groupby("b").mean()
                ax.plot(g.off, g.dc, mk + "-", color=col, markersize=6, linewidth=1.2, zorder=3)
                d = nat[(p, m, "voxel")]
                xs = np.array([g.off.min(), g.off.max()])
                ax.plot(xs, d["beta0_pp"] + d["beta1_pp_per_10Hz"] * xs / 10, color=col,
                        linewidth=1.6, linestyle=(0, (3, 1.5)), zorder=2)
            ax.axhline(0, color=INK2, linewidth=0.9)
            ax.set_title(m, fontsize=fs + 1, fontweight="bold")
            ax.set_xlabel("|off-resonance| (Hz)")
            if j == 0:
                ax.set_ylabel("ΔCoV (percentage points)")
    return draw


def fig_h4(deriv: Path, metrics=("FA", "MD", "MK"), fs: float = 16, map_metric: str = "MK", z: float = 0,
           clip_pct: float = 98):
    """H4 (confirmatory): (A) |off-resonance| and Delta scan-rescan CoV maps per repeat pair; (B) beta0 and beta1 per
    pair in CoV percentage points, with the registered verdict on beta1."""
    R = results(deriv)
    info, pairs, nat = _h4_setup(deriv)
    v = [R[("H4", l, m, p)]["verdict"] for l in ("voxel", "ROI") for m in metrics for p in pairs]
    cnt = {k: v.count(k) for k in H4_MARK}
    cap = (
        "**Figure 5. Off-resonance and the effect of denoising on scan–rescan variability (H4).** Repeat pairs (repeat "
        "scanner) ordered by the scanner's CNR at b = 1000, lowest first. "
        f"**(A)** Mean absolute off-resonance (TOPUP field, Hz) over the six repeats (top) and ΔCoV = denoised − "
        f"non-denoised scan–rescan CoV of {map_metric} (bottom; percentage points, blue = lower CoV with denoising); all "
        f"white-matter voxels warped to MNI space for display (the analysis used the sparse checkerboard voxels in each subject's "
        f"template space); axial slice z = {z:+.0f} mm (MNI). "
        "**(B)** The regression ΔCoV_r = β0 + β1·off_r (ΔCoV_r = difference of z-standardised CoV; off_r = mean |off-resonance| "
        "in Hz), converted to CoV percentage points with each pair's SD of CoV. Bars: β0, the denoising effect at zero "
        "off-resonance (posterior mean; negative = denoising reduces scan–rescan CoV). Markers: β1, the change in that "
        "effect, shown per SD of the pair's |off-resonance| (posterior mean and 95% CrI; positive = smaller CoV reduction at higher "
        "off-resonance, as predicted by H4). "
        f"Marker fill: supports H4 (filled; BF₁₀ > 10, β1 > 0; {cnt['supports H4']} of {len(v)}), "
        f"supports the null (open; BF₁₀ < 1/10; {cnt['supports null']}), inconclusive (light; "
        f"{cnt['inconclusive'] + cnt['inconclusive (BF10 > 10, opposite sign)']}, of which "
        f"{cnt['inconclusive (BF10 > 10, opposite sign)']} with BF₁₀ > 10 and β1 < 0, opposite to the prediction). "
        "**(C)** Voxel-wise ΔCoV against off-resonance: mean ΔCoV (percentage points) in each decile of |off-resonance| "
        "per pair (markers), and the fitted line β0 + β1·off in the same units (dashed); registered H4a voxels.")
    sc = [info[p]["repeat_scanner"] for p in pairs]
    leg = _sc_handles(sc) + [
        Patch(facecolor=mcolors.to_rgba(INK, 0.35), edgecolor=INK, label="β0 (at 0 Hz)"),
        Line2D([], [], marker="o", color=INK, linestyle="none", markersize=9, label="β1 per SD of |off|"),
        Line2D([], [], marker="o", color=INK, linestyle="none", markersize=9, label="supports H4"),
        Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="none", markersize=9,
               label="supports null"),
        Line2D([], [], marker="o", color=INK, markerfacecolor=mcolors.to_rgba(INK, 0.35), linestyle="none",
               markersize=9, label="inconclusive")]
    with plt.rc_context(_style(fs)):
        fig = _compose([
            dict(letter="A", title=f"Off-Resonance and the Denoising Effect on {map_metric} Scan–Rescan CoV",
                 top_pad=0.5, body=6.6, bottom_pad=0.3, draw=_h4_maps(deriv, pairs, info, nat, map_metric, z, fs, clip_pct)),
            dict(letter="B", title="Denoising Effect (β0) vs Its Change with Off-Resonance (β1)", legend=leg,
                 legend_ncol=6, top_pad=0.6, body=4.6, bottom_pad=0.7,
                 draw=_h4_scale(deriv, metrics, pairs, info, nat, R, fs)),
            dict(letter="C", title="Denoising Effect Across Off-Resonance (Voxel-wise)",
                 legend=[Line2D([], [], color=INK, marker="o", markersize=6, linewidth=1.2,
                                label="mean per decile of |off|"),
                         Line2D([], [], color=INK, linewidth=1.6, linestyle=(0, (3, 1.5)), label="fitted β0 + β1·off")],
                 top_pad=0.55, body=4.0, bottom_pad=1.0, draw=_h4_bins(deriv, metrics, pairs, info, nat, fs))], fs)
    fig.caption = cap
    return fig


def figx_h4_variogram(deriv: Path, metrics=("FA", "MD", "MK"), example: str = "sub-16766", fs: float = 16,
                      slice_frac: float = 0.6, dmax: float = 80):
    """Exploratory H4, variogram-matched null (Burt et al., 2020): (A) real |off-resonance| field and surrogate fields;
    (B) autocorrelation of the real field and of the surrogates; (C) observed rho vs the surrogate null; (D) decile
    curves of Delta CoV for the real field and the surrogates."""
    import pandas as pd
    info, pairs, nat = _h4_setup(deriv)
    d = deriv / "analysis" / "exploratory" / "H4_variogram_null"
    summ = pd.read_csv(d / "summary.csv").set_index(["pair", "level", "metric"])
    acf = pd.read_csv(d / "acf.csv.gz")
    ex = np.load(d / f"examples_{example}.npz")
    real, nulls, und = ex["real"], ex["null"], ex["underlay"]
    zs = np.flatnonzero(np.isfinite(real).any(axis=(0, 1)))
    k = int(zs[0] + slice_frac * (zs[-1] - zs[0]))               # axial slice at this fraction of the white matter's height
    sl = lambda v: np.rot90(v[:, :, k])  # noqa: E731
    u = sl(und)
    cr = _crop((u > 0).astype(int))
    vmax = np.nanpercentile(real, 99)
    n_out = int(summ.outside_null_95.sum())
    n_pos = int((summ.outside_null_95 & (summ.observed_rho > 0)).sum())
    cap = (
        "**Figure 9. Variogram-matched null for the off-resonance association (exploratory).** **(A)** Mean "
        f"|off-resonance| (Hz) of one repeat pair ({_short(info[example]['repeat_scanner'])}) on its white-matter voxels, "
        "and four surrogate fields with the same values and a matched variogram (BrainSMASH; Burt et al., 2020). "
        "Checkerboard voxels, filled to all white-matter voxels for display. **(B)** Spatial autocorrelation of the "
        "rank-transformed field against distance, voxel-wise, per repeat pair: real field (coloured) and surrogates "
        "(grey line = median, band = central 95% of 200 surrogates). **(C)** Rank correlation (Spearman ρ) of ΔCoV with "
        "|off-resonance| per repeat pair: marker = observed ρ; grey bar = central 95% of ρ over 1,000 surrogate fields. "
        f"Filled = observed ρ outside that range ({n_out} of {len(summ)}, {n_pos} of them with ρ > 0). **(D)** Voxel-wise "
        "ΔCoV (percentage points) against |off-resonance| (Hz), as in Figure 5C: mean ΔCoV per decile of |off-resonance| "
        "for the real field (coloured) and the central 95% (band) and median (grey line) over the surrogate fields.")

    def draw_maps(fig, left, right, bottom, top):
        n = 1 + len(nulls)
        gs = GridSpec(1, n + 1, figure=fig, width_ratios=[1] * n + [0.05], wspace=0.05, left=left - 0.04,
                      right=right - 0.04, top=top, bottom=bottom)
        for j, v in enumerate([real] + list(nulls)):
            ax = fig.add_subplot(gs[0, j], facecolor="white")
            ax.imshow(np.ma.masked_where(u[cr] <= 0, u[cr]), cmap="gray", vmin=0, vmax=1.6 * np.percentile(u[u > 0], 99),
                      interpolation="nearest")
            im = ax.imshow(np.ma.masked_invalid(sl(v)[cr]), cmap=OFF_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title("real field" if j == 0 else f"surrogate {j}", fontsize=fs, fontweight="bold" if j == 0 else None)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]))
        cb.set_label("|off-resonance| (Hz)", fontsize=fs - 1); cb.ax.tick_params(labelsize=fs - 2)
        cb.outline.set_visible(False)

    def draw_acf(fig, left, right, bottom, top):
        gs = GridSpec(1, len(pairs), figure=fig, wspace=0.12, left=left, right=right, top=top, bottom=bottom)
        for j, p in enumerate(pairs):
            ax = fig.add_subplot(gs[0, j], facecolor="white")
            col, _ = _sc(info[p]["repeat_scanner"])
            g = acf[(acf.pair == p) & (acf.level == "voxel") & (acf.dist_mm <= dmax)].sort_values("dist_mm")
            ax.fill_between(g.dist_mm, g.surr_low, g.surr_high, color=GRID, edgecolor=INK3, linewidth=0.6, zorder=1)
            ax.plot(g.dist_mm, g.surr_median, color=INK3, linewidth=1.6, zorder=2)
            ax.plot(g.dist_mm, g.real, color=col, linewidth=2.4, zorder=3)
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.set_ylim(-0.4, 1.02); ax.set_xlim(0, dmax)
            ax.set_title(_short(info[p]["repeat_scanner"]), fontsize=fs, fontweight="bold", color=col)
            ax.set_xlabel("distance (mm)")
            if j == 0:
                ax.set_ylabel("autocorrelation")
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=fs - 3)

    def draw_null(fig, left, right, bottom, top):
        gs = GridSpec(1, 2, figure=fig, wspace=0.08, left=left, right=right, top=top, bottom=bottom)
        n, w = len(pairs), 0.15
        cols = []
        for c, level in enumerate(("voxel", "ROI")):
            ax = fig.add_subplot(gs[0, c], facecolor="white", sharey=cols[0] if cols else None)
            for j, m in enumerate(metrics):
                for kk, p in enumerate(pairs):
                    x = j + (kk - (n - 1) / 2) * w
                    col, mk = _sc(info[p]["repeat_scanner"])
                    r = summ.loc[(p, level, m)]
                    ax.bar(x, r.null_high - r.null_low, bottom=r.null_low, width=w * 0.75, color=GRID, edgecolor=INK3,
                           linewidth=0.8, zorder=2)
                    ax.plot(x, r.observed_rho, mk, color=col, markersize=10, markeredgewidth=1.8, markeredgecolor=col,
                            markerfacecolor=col if r.outside_null_95 else "white", linestyle="none", zorder=4)
                if j:
                    ax.axvline(j - 0.5, color=GRID, linewidth=1)
            ax.axhline(0, color=INK2, linewidth=0.9)
            ax.set_xlim(-0.5, len(metrics) - 0.5); ax.grid(axis="x", visible=False)
            ax.set_xticks(range(len(metrics)), metrics, fontsize=fs + 1, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Spearman ρ (ΔCoV, |off|)")
            else:
                ax.tick_params(labelleft=False)
            cols.append(ax)
        _col_titles(fig, [(cols[0], "Voxel-wise"), (cols[1], "ROI-wise")], fs, dy_in=0.1)

    def draw_curves(fig, left, right, bottom, top):
        gs = GridSpec(len(metrics), len(pairs), figure=fig, wspace=0.12, hspace=0.18, left=left, right=right, top=top,
                      bottom=bottom)
        for i, m in enumerate(metrics):
            row = []
            for j, p in enumerate(pairs):
                ax = fig.add_subplot(gs[i, j], facecolor="white", sharey=row[0] if row else None)
                row.append(ax)
                col, mk = _sc(info[p]["repeat_scanner"])
                dz = np.load(d / f"deciles_{p}.npz")[m]                     # (n_null, bins, [Hz, ΔCoV])
                hx = np.median(dz[:, :, 0], axis=0)
                lo, md, hi = np.percentile(dz[:, :, 1], [2.5, 50, 97.5], axis=0)
                ax.fill_between(hx, lo, hi, color=GRID, edgecolor=INK3, linewidth=0.6, zorder=1)
                ax.plot(hx, md, color=INK3, linewidth=1.6, zorder=2)
                t = pd.read_csv(deriv / "analysis" / "inputs" / f"h4a_{p}_{m}.csv.gz",
                                usecols=["cov_nonden", "cov_den", "off_hz"])
                v, dc = t.off_hz.to_numpy(), 100 * (t.cov_den - t.cov_nonden).to_numpy()
                b = np.minimum((np.argsort(np.argsort(v)) * dz.shape[1]) // len(v), dz.shape[1] - 1)
                nb_ = np.bincount(b)
                ax.plot(np.bincount(b, v) / nb_, np.bincount(b, dc) / nb_, mk + "-", color=col, markersize=6,
                        linewidth=1.8, zorder=3)
                if i == 0:
                    ax.set_title(_short(info[p]["repeat_scanner"]), fontsize=fs, fontweight="bold", color=col)
                if j == 0:
                    ax.set_ylabel(f"ΔCoV {m} (pp)")
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if i == len(metrics) - 1:
                    ax.set_xlabel("|off-resonance| (Hz)")
                ax.tick_params(labelsize=fs - 3)

    sur = [Patch(facecolor=GRID, edgecolor=INK3, label="surrogates: central 95%"),
           Line2D([], [], color=INK3, linewidth=1.6, label="surrogates: median")]
    leg = _sc_handles([info[p]["repeat_scanner"] for p in pairs]) + [
        Patch(facecolor=GRID, edgecolor=INK3, label="null: central 95%"),
        Line2D([], [], marker="o", color=INK, linestyle="none", markersize=10, label="outside null"),
        Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="none", markersize=10,
               label="inside null")]
    with plt.rc_context(_style(fs)):
        fig = _compose([
            dict(letter="A", title="Real Off-Resonance Field and Variogram-Matched Surrogates", top_pad=0.55,
                 body=3.4, bottom_pad=0.3, draw=draw_maps),
            dict(letter="B", title="Spatial Autocorrelation: Real Field vs Surrogates (Voxel-wise)",
                 legend=[Line2D([], [], color=INK, linewidth=2.4, label="real field")] + sur, top_pad=0.9, body=3.0,
                 bottom_pad=0.9, draw=draw_acf),
            dict(letter="C", title="Observed Association vs Variogram-Matched Null", legend=leg, legend_ncol=5,
                 top_pad=0.6, body=4.6, bottom_pad=0.7, draw=draw_null),
            dict(letter="D", title="ΔCoV Across Off-Resonance: Real Field vs Surrogates (Voxel-wise)",
                 legend=[Line2D([], [], marker="o", color=INK, markersize=6, linewidth=1.8,
                                label="real field: mean per decile of |off|")] + sur,
                 top_pad=0.55, body=2.5 * len(metrics), bottom_pad=0.9, draw=draw_curves)], fs)
    fig.caption = cap
    return fig


def figx_h4_beta1_null(deriv: Path, metrics=("FA", "MD", "MK"), fs: float = 16):
    """Exploratory H4: the slope beta1, per SD of the pair's |off-resonance|, against its surrogate null."""
    import pandas as pd
    info, pairs, _ = _h4_setup(deriv)
    R = results(deriv)
    t = pd.read_csv(deriv / "analysis" / "exploratory" / "H4_variogram_null" / "summary.csv").set_index(["pair", "level", "metric"])
    o = t.beta1_outside_null_95
    sup = {k for k in t.index if R[("H4", k[1], k[2], k[0])]["verdict"] == "supports H4"}
    cap = (
        "**Figure S16. β1 against variogram-matched surrogate fields (H4, exploratory).** β1 per SD of the pair's "
        "|off-resonance| (markers; equal to β1 of the model with z-scored |off-resonance|) for each repeat pair, metric and "
        "level. Grey bar = central 95% of β1 over 1,000 surrogate fields with the same values and a matched variogram. "
        f"Filled = outside that range ({int(o.sum())} of {len(t)}: {int((o & (t.beta1_per_hz > 0)).sum())} with β1 > 0, as "
        f"predicted by H4, and {int((o & (t.beta1_per_hz < 0)).sum())} with β1 < 0). Black outline = test that supports H4 in the "
        f"registered analysis ({len(sup)}, of which {sum(bool(o[k]) for k in sup)} are outside the range).")

    def draw(fig, left, right, bottom, top):
        gs = GridSpec(1, 2, figure=fig, wspace=0.08, left=left, right=right, top=top, bottom=bottom)
        n, w, cols = len(pairs), 0.15, []
        for c, level in enumerate(("voxel", "ROI")):
            ax = fig.add_subplot(gs[0, c], facecolor="white", sharey=cols[0] if cols else None)
            for j, m in enumerate(metrics):
                for k, p in enumerate(pairs):
                    x = j + (k - (n - 1) / 2) * w
                    col, mk = _sc(info[p]["repeat_scanner"])
                    r = t.loc[(p, level, m)]
                    lo, hi, b1 = (v * r.off_hz_sd for v in (r.beta1_null_low, r.beta1_null_high, r.beta1_per_hz))
                    ax.bar(x, hi - lo, bottom=lo, width=w * 0.75, color=GRID, edgecolor=INK3, linewidth=0.8, zorder=2)
                    ax.plot(x, b1, mk, color=col, markersize=11, markeredgewidth=1.8, linestyle="none", zorder=4,
                            markeredgecolor=INK if (p, level, m) in sup else col,
                            markerfacecolor=col if r.beta1_outside_null_95 else "white")
                if j:
                    ax.axvline(j - 0.5, color=GRID, linewidth=1)
            ax.axhline(0, color=INK2, linewidth=0.9)
            ax.set_xlim(-0.5, len(metrics) - 0.5); ax.grid(axis="x", visible=False)
            ax.set_xticks(range(len(metrics)), metrics, fontsize=fs + 1, fontweight="bold")
            if c == 0:
                ax.set_ylabel("β1 per SD of |off|")
            else:
                ax.tick_params(labelleft=False)
            cols.append(ax)
        _col_titles(fig, [(cols[0], "Voxel-wise"), (cols[1], "ROI-wise")], fs, dy_in=0.1)

    leg = _sc_handles([info[p]["repeat_scanner"] for p in pairs]) + [
        Patch(facecolor=GRID, edgecolor=INK3, label="null: central 95%"),
        Line2D([], [], marker="o", color=INK3, linestyle="none", markersize=10, label="outside null"),
        Line2D([], [], marker="o", color=INK3, markerfacecolor="white", linestyle="none", markersize=10, label="inside null"),
        Line2D([], [], marker="o", color=INK, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", markersize=11,
               label="supports H4 (registered)")]
    with plt.rc_context(_style(fs)):
        fig = _compose([dict(letter=None, title="H4: β1 vs Variogram-Matched Null", legend=leg, legend_ncol=5, top_pad=0.6,
                             body=4.8, bottom_pad=0.7, draw=draw)], fs)
    fig.caption = cap
    return fig


def figx_h4_variogram_fit(deriv: Path, fs: float = 16, dmax: float = 80):
    """Supplementary check of the variogram-matched surrogates: spatial autocorrelation of the real |off-resonance|
    field and of the surrogates, per repeat pair, voxel-wise (top) and ROI-wise (bottom)."""
    import pandas as pd
    info, pairs, nat = _h4_setup(deriv)
    acf = pd.read_csv(deriv / "analysis" / "exploratory" / "H4_variogram_null" / "acf.csv.gz")
    cap = (
        "**Supplementary Figure SX. Spatial autocorrelation of the off-resonance field and of its surrogates (H4, "
        "exploratory).** Autocorrelation of the rank-transformed |off-resonance| field against distance (4 mm classes) "
        "for each repeat pair, on the registered H4a voxels (top) and H4b ROIs (bottom): real field (coloured) and "
        "variogram-matched surrogates (grey: median and central 95% of 200 surrogates).")

    def draw(fig, left, right, bottom, top):
        gs = GridSpec(2, len(pairs), figure=fig, wspace=0.12, hspace=0.25, left=left, right=right, top=top, bottom=bottom)
        for i, level in enumerate(("voxel", "ROI")):
            for j, p in enumerate(pairs):
                ax = fig.add_subplot(gs[i, j], facecolor="white")
                col, _ = _sc(info[p]["repeat_scanner"])
                g = acf[(acf.pair == p) & (acf.level == level) & (acf.dist_mm <= dmax)].sort_values("dist_mm")
                ax.fill_between(g.dist_mm, g.surr_low, g.surr_high, color=GRID, edgecolor=INK3, linewidth=0.6, zorder=1)
                ax.plot(g.dist_mm, g.surr_median, color=INK3, linewidth=1.6, zorder=2)
                ax.plot(g.dist_mm, g.real, color=col, linewidth=2.4, zorder=3)
                ax.axhline(0, color=INK2, linewidth=0.8)
                ax.set_ylim(-0.4, 1.02); ax.set_xlim(0, dmax)
                if i == 0:
                    ax.set_title(_short(info[p]["repeat_scanner"]), fontsize=fs, fontweight="bold", color=col)
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("distance (mm)")
                if j == 0:
                    ax.set_ylabel(f"autocorrelation\n({'voxel' if level == 'voxel' else 'ROI'}-wise)")
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=fs - 3)

    with plt.rc_context(_style(fs)):
        fig = _compose([dict(letter="", title="Spatial Autocorrelation: Real Field vs Surrogates",
                             legend=[Line2D([], [], color=INK, linewidth=2.4, label="real field"),
                                     Patch(facecolor=GRID, edgecolor=INK3, label="surrogates: central 95%"),
                                     Line2D([], [], color=INK3, linewidth=1.6, label="surrogates: median")],
                             top_pad=0.9, body=5.6, bottom_pad=0.9, draw=draw)], fs)
    fig.caption = cap
    return fig
