'''
Unattended multi-experiment run: several independent training phases one after another, all starting from the same
model, each limited by --hours. At the end every result is benchmarked (evalModel.py, fixed seeds) against the start
model, and the table is written to models/night_benchmark.txt. A crash in one phase does not stop the next one.

    python nightRun.py                 3 phases x 3 hours (+ benchmark at the end, a few minutes)
    python nightRun.py --hours 2       shorter phases
    python nightRun.py --smoke         tiny version to check that everything works (~1 minute)

Phases (edit PHASES / START below to change the experiment):
  hardExplore  starts 20-30k only (maximum difficulty), big mutations   (sigma 0.05, mutRate 1.0)
  hardCareful  starts 20-30k only, careful mutations                    (sigma 0.02, mutRate 0.7)
  normalSafe   normal starts, careful mutations                         (the setting that worked at normal starts)
After the night, compare with:  models/night_benchmark.txt  (and the PNG/CSV next to every model).
'''
import argparse
import glob
import os
import subprocess
import sys
import time
import traceback

import proSprudler as ps

START = "models/sprudler_2026-10-02_15-27-39.pt"   # the strongest model so far (benchmark: 4,321 at start 0, 833 on 20-30k starts)

PHASES = [
    dict(name="hardExplore", minStart=20000, maxStart=30000, sigma=0.05, mutRate=1.0),
    dict(name="hardCareful", minStart=20000, maxStart=30000, sigma=0.02, mutRate=0.7),
    dict(name="normalSafe",  minStart=0,     maxStart=None,  sigma=0.02, mutRate=0.7),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unattended multi-phase training run")
    ap.add_argument("--hours", type=float, default=3.0, help="time limit per phase (default 3)")
    ap.add_argument("--smoke", action="store_true", help="tiny test version")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    log = os.path.join("models", "night_log.txt")
    os.makedirs("models", exist_ok=True)

    def note(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    workers, pop, games, stage1, gens = (4, 12, 6, 3, 50) if a.smoke else (24, 200, 200, 20, 4000)
    hours = 0.0004 if a.smoke else a.hours
    results = []
    note(f"night run start: {len(PHASES)} phases x {hours} h from {START}")
    for ph in PHASES:
        note(f"phase {ph['name']}: starts {ph['minStart']}-{ph['maxStart']}, sigma {ph['sigma']}, mutRate {ph['mutRate']}")
        try:
            ps.sprudlerTrainingLoop(workers, games, gens, pop, ph["mutRate"], ph["sigma"], 3 if not a.smoke else 1, 0.1 if not a.smoke else 0.25,
                                    startNetwork=ps.loadNetwork(START), networkName=ph["name"], livePlot=False,
                                    minStartHeight=ph["minStart"], maxStartHeight=ph["maxStart"],
                                    stage1Games=stage1, finalistFraction=0.3 if not a.smoke else 0.5, maxHours=hours)
            files = [f for f in glob.glob(f"models/{ph['name']}_*.pt") if not f.endswith("_pool.pt")]
            newest = max(files, key=os.path.getmtime)
            results.append(newest)
            note(f"phase {ph['name']} finished -> {newest}")
        except BaseException:                       # includes Ctrl-C inside a phase before its first generation ends
            note(f"phase {ph['name']} FAILED:\n{traceback.format_exc()}")

    if results:
        note("benchmarking all results against the start model ...")
        cmd = [sys.executable, "evalModel.py", START] + results + ["--min-start", "20000"]
        if a.smoke:
            cmd += ["--games", "50", "--workers", "4"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = out.stdout + (("\n" + out.stderr[-2000:]) if out.returncode else "")
        with open(os.path.join("models", "night_benchmark.txt"), "w") as f:
            f.write(text)
        note("benchmark written to models/night_benchmark.txt\n" + text)
    note("night run done")
