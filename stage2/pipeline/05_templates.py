#!/usr/bin/env python3
"""
05_templates.py - ANTs 2.4.4 subject templates (antsMultivariateTemplateConstruction2.sh).

Templates:
  sub-XXXXX/all                one per subject from all of its sessions      (H1-H3)
  sub-XXXXX/repeat-<scanner>   one per repeat subject from its 6 repeats     (H4)
Input per session: BRC struc_preproc bias-corrected, brain-extracted T1
(analysis/anatMRI/T1/processed/data/T1_unbiased_brain.nii.gz).

antsMultivariateTemplateConstruction2.sh -d 3 -i 4 -g 0.25 -t SyN -m CC -q 100x100x70x20
    -f 6x4x2x1 -s 3x2x1x0vox -r 1 -n 0 -c 2 -j <jobs> -o <prefix> <inputs>
(script defaults except -r 1: rigid initialisation to the inputs' average, and -n 0: no N4, the
inputs are already bias-corrected; -c/-j only control parallelism.)

Each template runs detached in <deriv>/templates/<id>/ with a generated script, log, status and
provenance (input SHA-256, ANTs version, command). Finished templates are skipped. No diffusion data
are read; T1 images are only passed to ANTs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_VERSION = "1.0.0"
ANTSPATH = Path("/usr/local/ants/ants-2.4.4/bin")
EXPECTED_ANTS = "2.4.4"
TEMPLATE_ARGS = ["-d", "3", "-i", "4", "-g", "0.25", "-t", "SyN", "-m", "CC", "-q", "100x100x70x20",
                 "-f", "6x4x2x1", "-s", "3x2x1x0vox", "-r", "1", "-n", "0"]
T1_REL = "analysis/anatMRI/T1/processed/data/T1_unbiased_brain.nii.gz"


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(meta_dir: Path, deriv: Path) -> list[dict]:
    rows = list(csv.DictReader(open(meta_dir / "sessions_manifest.csv", newline="")))
    by_sub = {}
    for r in rows:
        by_sub.setdefault(r["subject"], []).append(r)
    out = []
    for sub, rs in sorted(by_sub.items()):
        rs.sort(key=lambda r: r["session"])
        out.append({"id": f"{sub}/all", "subject": sub, "kind": "all", "sessions": [r["session"] for r in rs]})
        reps = [r for r in rs if r.get("n_sessions_on_scanner") == "6"]
        for lab in sorted({r["scanner_label"] for r in reps}):
            ses = [r["session"] for r in reps if r["scanner_label"] == lab]
            out.append({"id": f"{sub}/repeat-{lab}", "subject": sub, "kind": "repeat", "scanner": lab, "sessions": ses})
    for t in out:
        t["inputs"] = [str(deriv / "brc" / "anat" / f"{t['subject']}_{s}" / T1_REL) for s in t["sessions"]]
        t["dir"] = deriv / "templates" / t["id"]
    return out


def run_template(t: dict, jobs: int, threads: int, force: bool) -> dict:
    d: Path = t["dir"]
    d.mkdir(parents=True, exist_ok=True)
    status_p, prov_p = d / "status.json", d / "provenance.json"
    final = d / "T_template0.nii.gz"
    if not force and final.is_file() and status_p.is_file() and json.loads(status_p.read_text()).get("state") == "done":
        return {"id": t["id"], "state": "skipped (done)"}
    missing = [i for i in t["inputs"] if not Path(i).is_file()]
    if missing:
        return {"id": t["id"], "state": "failed", "errors": [f"missing input {m}" for m in missing]}
    # Copy inputs under short, unique names so ANTs output files are traceable to sessions.
    inp = d / "inputs"
    inp.mkdir(exist_ok=True)
    local = []
    for ses, src in zip(t["sessions"], t["inputs"]):
        dst = inp / f"{t['subject']}_{ses}_T1w.nii.gz"
        if not dst.is_file() or dst.stat().st_size != Path(src).stat().st_size:
            subprocess.run(["cp", src, str(dst)], check=True)
        local.append(str(dst))
    cmd = [str(ANTSPATH / "antsMultivariateTemplateConstruction2.sh")] + TEMPLATE_ARGS + \
          ["-c", "2", "-j", str(jobs), "-o", str(d) + "/T_"] + local
    script = d / "run_template.sh"
    script.write_text("#!/bin/bash\nset -e\n"
                      f"export ANTSPATH={ANTSPATH}/\nexport PATH={ANTSPATH}:$PATH\n"
                      f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}\n"
                      f"cd {shlex.quote(str(d))}\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    script.chmod(0o755)
    prov = {"template": t["id"], "kind": t["kind"], "sessions": t["sessions"],
            "inputs": {s: {"path": p, "sha256": sha256_of(Path(p))} for s, p in zip(t["sessions"], t["inputs"])},
            "command": cmd, "ants_version": ants_version(), "script": "05_templates.py v" + SCRIPT_VERSION,
            "itk_threads_per_registration": threads, "started": utcnow(), "host": platform.node()}
    prov_p.write_text(json.dumps(prov, indent=2))
    status_p.write_text(json.dumps({"state": "running", "started": utcnow()}))
    t0 = time.time()
    with open(d / "run_template.log", "w") as fh:
        rc = subprocess.run(["bash", str(script)], stdout=fh, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and final.is_file()
    st = {"state": "done" if ok else "failed", "rc": rc, "finished": utcnow(), "runtime_min": round((time.time() - t0) / 60)}
    status_p.write_text(json.dumps(st, indent=2))
    prov.update(finished=st["finished"], rc=rc,
                template_sha256=sha256_of(final) if final.is_file() else None)
    prov_p.write_text(json.dumps(prov, indent=2))
    return {"id": t["id"], "state": st["state"], "runtime_min": st["runtime_min"]}


def ants_version() -> str:
    out = subprocess.run([str(ANTSPATH / "antsRegistration"), "--version"], capture_output=True, text=True).stdout
    return out.split("\n")[0].replace("ANTs Version:", "").strip()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build ANTs subject templates (all sessions; repeat sessions).")
    p.add_argument("--deriv-dir", type=Path, required=True)
    p.add_argument("--meta-dir", type=Path, required=True)
    p.add_argument("--parallel-templates", type=int, default=5)
    p.add_argument("--jobs", type=int, default=6, help="Registrations in parallel per template (-j)")
    p.add_argument("--threads", type=int, default=2, help="ITK threads per registration")
    p.add_argument("--only", nargs="+", help="Template ids, e.g. sub-16981/all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    ver = ants_version()
    if not ver.startswith(EXPECTED_ANTS):
        print(f"ANTs is {ver}, expected {EXPECTED_ANTS}", file=sys.stderr)
        return 2
    templates = plan(a.meta_dir, a.deriv_dir)
    if a.only:
        templates = [t for t in templates if t["id"] in set(a.only)]
    log_dir = a.deriv_dir / "templates" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"05_templates_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log", "a")

    def say(msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    say(f"05_templates.py v{SCRIPT_VERSION} (sha256 {sha256_of(Path(__file__))}), ANTs {ver}, pid {os.getpid()}")
    for t in templates:
        say(f"PLAN {t['id']}: {len(t['sessions'])} sessions {t['sessions']}")
    if a.dry_run:
        return 0
    results = []
    with ThreadPoolExecutor(max_workers=a.parallel_templates) as ex:
        futs = {ex.submit(run_template, t, a.jobs, a.threads, a.force): t for t in templates}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001 - reported
                r = {"id": futs[f]["id"], "state": "failed", "errors": [str(exc)]}
            results.append(r)
            say(f"{r['state'].upper()} {r['id']} {r.get('runtime_min', '')} {'; '.join(r.get('errors', []))}")
    failed = [r for r in results if r["state"] == "failed"]
    say(f"Finished: {len(results)} templates, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
