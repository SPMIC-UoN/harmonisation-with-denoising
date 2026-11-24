#!/usr/bin/env python3
"""
run_models.py - Fit the confirmatory models (H1-H4) on the tables from build_inputs.py. Local replacement of the Stage 1 SLURM array scripts: one process per fit, 4 chains in parallel per fit,
several fits at once; job status files as in 03_brc.py (<out>/_jobs/<job>.json; done jobs skipped).

  python run_models.py --inputs <deriv>/analysis/inputs --out <deriv>/analysis/results [--only H1a_FA ...]
  python run_models.py --out ... --status
  python run_models.py --out ... --combine        # H3 Delta ICC + results table with the pre-registered criteria
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONTAINER_PY = "/opt/conda/envs/power-calc-env/bin/python"      # Stage 1 env.sif (env.def)
METRICS = ("FA", "MD", "MK")


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def seed_of(job: str) -> int:
    return int(hashlib.sha256(job.encode()).hexdigest()[:7], 16)


def read_table(p: Path) -> dict:
    with gzip.open(p, "rt") as fh:
        r = csv.reader(fh)
        head = next(r)
        cols = list(zip(*r))
    return {h: np.array(c) for h, c in zip(head, cols)}


def codes(x: np.ndarray) -> np.ndarray:
    labs = {k: i for i, k in enumerate(sorted(set(x.tolist())))}
    return np.array([labs[k] for k in x], dtype=int)


def jobs(inputs: Path) -> dict:
    """job name -> (model key, table path, arm or None)."""
    J = {}
    for m in METRICS:
        for lvl in "ab":
            J[f"H1{lvl}_{m}"] = (f"H1{lvl}", inputs / f"h1{lvl}_{m}.csv.gz", None)
            for arm in (0, 1):
                J[f"H3{lvl}_{m}_den{arm}"] = ("H3", inputs / f"h3{lvl}_{m}.csv.gz", arm)
    for p in sorted(inputs.glob("h2?_sub-*_*.csv.gz")) + sorted(inputs.glob("h4?_sub-*_*.csv.gz")):
        stem = p.name[:-7]                                       # h2a_sub-xxx_FA
        h, sub, m = stem.split("_")
        key = {"h2a": "H2a", "h2b": "H2b", "h4a": "H4", "h4b": "H4"}[h]
        J[f"H{h[1]}{h[2]}_{sub}_{m}"] = (key, p, None)
    return J


def data_for(key: str, t: dict, arm) -> dict:
    f = lambda c: t[c].astype(float)
    if key == "H1a":
        return dict(y=f("y"), subj=codes(t["subject"]), den=f("den"))
    if key == "H1b":
        return dict(y=f("y"), subj=codes(t["subject"]), den=f("den"), r=codes(t["unit"]))
    if key == "H2a":
        return dict(y=f("y"))
    if key == "H2b":
        return dict(y=f("y"), r=np.arange(t["y"].size))           # one Delta Bias per ROI
    if key == "H4":
        return dict(y=f("delta"), off_r=f(os.environ.get("H4_COL", "off_hz")))
    if key == "H3":
        m = t["den"].astype(int) == arm
        r = t["r"].astype(int)[m]
        return dict(y=f("y")[m], subj=codes(t["subject"][m]), scanner=codes(t["scanner"][m]),
                    r=codes(r), cond=np.zeros(int(m.sum()), dtype=int))
    raise KeyError(key)


# --------------------------------------------------------------------------------------------- one fit
def run_one(job: str, inputs: Path, out: Path, cores: int, draws: int, tune: int, h3_impl: str = "unchanged") -> None:
    import models as M
    key, table, arm = jobs(inputs)[job]
    data = data_for(key, read_table(table), arm)
    d = out / job
    d.mkdir(parents=True, exist_ok=True)
    seed = seed_of(job)
    t0 = time.time()
    if key == "H3":
        fit = M.fit_h3_arm if h3_impl == "unchanged" else M.fit_h3_arm_suffstats
        draws_d, res, idata = fit(data, seed=seed, cores=cores, draws=draws, tune=tune)
        res["h3_impl"] = h3_impl
        np.savez(d / "icc_draws.npz", **draws_d)
        names = ["sd_u", "sd_w", "sigma_cond"]
    else:
        if key == "H1a":                                    # exact per-cell likelihood, same model/priors/BF
            res, idata = M.fit_h1a_suffstats(data, seed=seed, cores=cores, draws=draws, tune=tune)
        else:
            res, idata = M.fit_bf(key, data, seed=seed, cores=cores, draws=draws, tune=tune)
        names = None
    res.update(job=job, model=key, table=str(table), table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
               n_obs=int(np.asarray(data["y"]).size), seed=seed, runtime_s=round(time.time() - t0),
               n_divergences=int(idata.sample_stats["diverging"].values.sum()))
    rows = M.summarise(idata, names)
    if key == "H3":
        rows = [r for r in rows if r["param"] != "sigma_cond[1]"]          # no data in a one-arm fit
    with open(d / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    infer = {"sd_u", "sd_w", "sigma_cond[0]"} if key == "H3" else {M.TARGET[key]}
    res["converged"] = bool(all(r["rhat"] < 1.01 and r["ess_bulk"] > 1000 for r in rows if r["param"] in infer))
    res["convergence_checked"] = sorted(infer)
    (d / "result.json").write_text(json.dumps(res, indent=1, default=float))
    idata.to_netcdf(str(d / "idata.nc"), engine="h5netcdf")


# --------------------------------------------------------------------------------------------- runner
def status_path(out: Path, job: str) -> Path:
    return out / "_jobs" / f"{job}.json"


def launch(job: str, a) -> dict:
    sp = status_path(a.out, job)
    st = json.loads(sp.read_text()) if sp.is_file() else {}
    if st.get("state") == "done" and (a.out / job / "result.json").is_file() and not a.force:
        return {"job": job, "state": "skipped"}
    log = a.out / "_jobs" / f"{job}.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTENSOR_FLAGS=f"compiledir={a.out / '_jobs' / 'pytensor' / job}")
    py = [sys.executable] if not a.container else \
        ["singularity", "exec", "--pwd", str(HERE)] + sum((["--bind", b] for b in a.bind), []) + \
        [str(a.container), CONTAINER_PY]
    cmd = py + ["-W", "ignore", str(Path(__file__)), "--one", job, "--inputs", str(a.inputs),
                "--out", str(a.out), "--cores", str(a.cores), "--draws", str(a.draws), "--tune", str(a.tune), "--h3-impl", a.h3_impl]
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        sp.write_text(json.dumps(dict(job=job, state="running", pid=proc.pid, host=platform.node(),
                                      started=utcnow(), cmd=cmd), indent=1))
        rc = proc.wait()
    st = dict(job=job, state="done" if rc == 0 else "failed", rc=rc, finished=utcnow(),
              runtime_s=round(time.time() - t0), log=str(log), cmd=cmd)
    sp.write_text(json.dumps(st, indent=1))
    return st


def combine(out: Path) -> None:
    import models as M
    rows = []
    for rj in sorted(out.glob("*/result.json")):
        r = json.loads(rj.read_text())
        if r["model"] == "H3":
            continue
        rows.append(dict(job=r["job"], model=r["model"], n_obs=r["n_obs"], target=r["target"],
                         mean=r[f"mean_{r['target']}"], sd=r[f"sd_{r['target']}"], BF10=r["BF10"],
                         converged=r["converged"], n_divergences=r["n_divergences"],
                         interpretation=M.interpret(r["model"], r)))
    for lvl in "ab":
        for m in METRICS:
            d0, d1 = out / f"H3{lvl}_{m}_den0", out / f"H3{lvl}_{m}_den1"
            if not ((d0 / "icc_draws.npz").is_file() and (d1 / "icc_draws.npz").is_file()):
                continue
            res = M.delta_icc(np.load(d1 / "icc_draws.npz")["icc"], np.load(d0 / "icc_draws.npz")["icc"])
            r0, r1 = (json.loads((x / "result.json").read_text()) for x in (d0, d1))
            res.update(converged=r0["converged"] and r1["converged"],
                       n_divergences=r0["n_divergences"] + r1["n_divergences"])
            (out / f"H3{lvl}_{m}_delta_icc.json").write_text(json.dumps(res, indent=1))
            rows.append(dict(job=f"H3{lvl}_{m}", model=f"H3{lvl}", n_obs=r0["n_obs"] + r1["n_obs"],
                             target="delta_icc", mean=res["mean_delta"], sd=None, BF10=None,
                             p_delta_gt0=res["p_delta_gt0"], q2_5=res["q2_5"], q97_5=res["q97_5"],
                             converged=res["converged"], n_divergences=res["n_divergences"],
                             interpretation=M.interpret(f"H3{lvl}", res)))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} results -> {out / 'results.csv'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fit the confirmatory models locally.")
    p.add_argument("--inputs", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+")
    p.add_argument("--max-jobs", type=int, default=24, help="fits at once (each uses --cores processes)")
    p.add_argument("--cores", type=int, default=4, help="chains in parallel per fit")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--container", type=Path, help="Stage 1 simulation image (env.sif); fits run inside it")
    p.add_argument("--bind", nargs="*", default=[], help="extra singularity binds, e.g. the derivatives dir")
    p.add_argument("--h3-impl", choices=("unchanged", "suffstats"), default="suffstats",
                   help="H3 likelihood: exact sufficient-statistics form (default) or the unchanged H3a.py code")
    p.add_argument("--h4-col", choices=("off_hz", "off_z"), default="off_hz",
                   help="H4 off-resonance covariate: |Hz| (default) or z-scored |Hz| (exploratory)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--one", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if not a.one:                                                # passed on to the fits (and into the container)
        os.environ["H4_COL"] = os.environ["SINGULARITYENV_H4_COL"] = a.h4_col
    if a.one:
        run_one(a.one, a.inputs, a.out, a.cores, a.draws, a.tune, a.h3_impl)
        return 0
    if a.status:
        for sp in sorted((a.out / "_jobs").glob("*.json")):
            s = json.loads(sp.read_text())
            print(f"{s['job']:28s} {s['state']:8s} {s.get('runtime_s', '')}")
        return 0
    if a.combine:
        combine(a.out)
        return 0
    if (a.draws, a.tune) != (2000, 2000):
        print(f"WARNING: draws/tune {a.draws}/{a.tune} differ from the pre-registered 2000/2000", flush=True)
    (a.out / "_jobs").mkdir(parents=True, exist_ok=True)
    J = jobs(a.inputs)
    todo = [j for j in J if (not a.only or j in a.only) and J[j][1].is_file()]
    print(f"{utcnow()} {len(todo)} jobs; {a.max_jobs} at once x {a.cores} chains", flush=True)
    rc = 0
    with ThreadPoolExecutor(max_workers=a.max_jobs) as ex:
        for f in as_completed([ex.submit(launch, j, a) for j in todo]):
            r = f.result()
            print(f"{utcnow()} {r['job']} {r['state']} {r.get('runtime_s', '')}", flush=True)
            rc |= r["state"] == "failed"
    return rc


if __name__ == "__main__":
    sys.exit(main())
