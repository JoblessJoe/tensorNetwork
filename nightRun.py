'''
Unattended multi-phase run: training phases one after another, each limited by generations and/or hours.
A phase starts from scratch, from a model file, or from the result of an earlier phase ('after').
At the end every result is benchmarked (evalModel.py, fixed seeds) and the table is written to
models/night_benchmark.txt. A crash in one phase does not stop the others (a phase whose 'after' phase failed is skipped).

    python nightRun.py                 all phases (see PHASES), then the benchmark
    python nightRun.py --scale 0.5     every phase's hours/generations x 0.5
    python nightRun.py --smoke         tiny version to check that everything works (~1 minute)

Current experiment (both from the 'target' pool, target-mode net: 25 inputs / 7 outputs, stable slots, steer smoothing 0.5):
  careful   careful mutations only (sigma 0.02, mutRate 0.7), 5 h - the polish step alone
  monster   the same + monster practice (30% of the games with 2x the monster spawn chance), 5 h
Benchmark columns: start 0 | hard 20-30k | mixed 0-30k | half at 0 + half hard | hard with 2x monsters; reference: the target model itself.
'''
import argparse
import glob
import os
import re
import subprocess
import sys
import time
import traceback

import proSprudler as ps

BASELINE = "models/sprudler_2026-10-02_17-00-59.pt"   # 23-input from-scratch model (1,000 gens, normal starts)

# start: model file | None (+ 'after': name of an earlier phase | None -> new network of 'inputs' inputs)
HARDEXPLORE_POOL = "models/hardExplore_2026-10-02_20-07-55_pool.pt"   # hard 2,339 / start 0 2,916; its pool keeps the diversity of the run
HARDEXPLORE = "models/hardExplore_2026-10-02_20-07-55.pt"

# start: model file | pool: pool file | after: earlier phase | none of them -> new network of 'inputs' inputs and 'hidden' layers
# zero: share of the games that always start at 0 (the rest draws from minStart..maxStart); snap: generations at which a snapshot is saved
TARGET_POOL = "models/target_2026-10-05_10-07-53_pool.pt"   # explore run: 6,326 / 2,760 / 3,966 / 4,812 (start 0 / hard / mixed / training mix)
TARGET = "models/target_2026-10-05_10-07-53.pt"
# smooth: steer smoothing, stable: stableSlots observation, monster / monsterMult: monster practice (share of games, spawn chance multiplier)
COMMON = dict(pool=TARGET_POOL, inputs=25, minStart=20000, maxStart=30000, zero=0.5, sigma=0.02, mutRate=0.7, smooth=0.5, stable=True, gens=100000, hours=5.0)
PHASES = [
    dict(COMMON, name="careful"),
    dict(COMMON, name="monster", monster=0.3, monsterMult=2.0),
    dict(COMMON, name="monsterExplore", monster=0.3, monsterMult=2.0, sigma=0.05, mutRate=1.0),   # monster practice alone (the explore settings of the target run)
]
REFERENCES = [TARGET]    # benchmarked as well, for comparison (all models here use stableSlots + smoothing 0.5, which evalModel applies to every model)
RANGES = ["20000:30000", "0:30000", "20000:30000:0.5"]    # benchmark columns after 'start 0': hard, mixed, and the zeroStart objective (half at 0, half hard)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unattended multi-phase training run")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every phase's hours and generations (default 1)")
    ap.add_argument("--phases", help="comma-separated names: run only these phases (the benchmark goes to night_benchmark_<names>.txt)")
    ap.add_argument("--no-plot", action="store_true", help="do not open the live learning-curve window")
    ap.add_argument("--smoke", action="store_true", help="tiny test version")
    a = ap.parse_args()
    if a.phases:
        names = a.phases.split(",")
        PHASES[:] = [ph for ph in PHASES if ph["name"] in names]
    elif not a.smoke:
        PHASES[:] = [ph for ph in PHASES if ph["name"] != "monsterExplore"]    # queued separately
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    log = os.path.join("models", "night_log.txt")
    os.makedirs("models", exist_ok=True)

    def note(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    workers, pop, games, stage1, elites, keep, frac = (4, 12, 6, 3, 1, 0.25, 0.5) if a.smoke else (24, 200, 200, 20, 3, 0.1, 0.3)
    outputs = {}     # phase name -> best-network file
    t0 = time.time()
    note(f"run start: {[p['name'] for p in PHASES]}")
    if not a.smoke and not a.no_plot:    # ONE live window for the whole night: follows the newest run, chained to its parent runs
        subprocess.Popen([sys.executable, "plotRun.py", "--live", "--chain", "--latest"])
    for ph in PHASES:
        gens = 12 if a.smoke else max(1, int(ph["gens"] * a.scale))
        hours = 0.0004 if a.smoke else ph["hours"] * a.scale
        note(f"phase {ph['name']}: {ph['inputs']} inputs, hidden {ph.get('hidden', [23, 23, 23])}, starts {ph['minStart']}-{ph['maxStart']}, sigma {ph['sigma']}, "
             f"mutRate {ph['mutRate']}, up to {gens} gens / {hours:.2f} h")
        try:
            kw = {}
            if ph.get("after"):
                if ph["after"] not in outputs:
                    raise RuntimeError(f"phase '{ph['after']}' has no result - skipping")
                kw["startNetwork"] = ps.loadNetwork(outputs[ph["after"]])
            elif ph.get("pool"):
                kw["startNetwork"] = ps.loadPool(ph["pool"])
            elif ph.get("start"):
                kw["startNetwork"] = ps.loadNetwork(ph["start"])
            else:
                kw.update(inputSize=ph["inputs"], hiddenSizes=ph.get("hidden", [23, 23, 23]), outputSize=2)
            ps.sprudlerTrainingLoop(workers, games, gens, pop, ph["mutRate"], ph["sigma"], elites, keep, networkName=ph["name"], livePlot=False,
                                    minStartHeight=ph["minStart"], maxStartHeight=ph["maxStart"], zeroFraction=ph.get("zero", 0.0),
                                    snapshotAt=(5,) if a.smoke and ph.get("snap") else ph.get("snap", ()),
                                    steerSmoothing=ph.get("smooth", 1.0), stableSlots=ph.get("stable", False),
                                    monsterFraction=ph.get("monster", 0.0), monsterMult=ph.get("monsterMult", 1.0),
                                    stage1Games=stage1, finalistFraction=frac, maxHours=hours, **kw)
            files = [f for f in glob.glob(f"models/{ph['name']}_*.pt") if not f.endswith("_pool.pt") and not re.search(r"_g\d+\.pt$", f)]
            outputs[ph["name"]] = max(files, key=os.path.getmtime)
            note(f"phase {ph['name']} finished -> {outputs[ph['name']]}")
        except BaseException:                       # includes Ctrl-C inside a phase before its first generation ends
            note(f"phase {ph['name']} FAILED:\n{traceback.format_exc()}")

    if outputs:
        note("benchmarking ...")
        snaps = sorted(f for name in outputs for f in glob.glob(f"models/{name}_*_g[0-9]*.pt") if os.path.getmtime(f) > t0)
        cmd = [sys.executable, "evalModel.py"] + REFERENCES + list(outputs.values()) + snaps + ["--stable", "--smoothing", "0.5", "--monster-column", "2", "--ranges"] + RANGES
        if a.smoke:
            cmd += ["--games", "50", "--workers", "4"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = out.stdout + (("\n" + out.stderr[-2000:]) if out.returncode else "")
        with open(os.path.join("models", f"night_benchmark_{a.phases.replace(',', '_')}.txt" if a.phases else "night_benchmark.txt"), "w") as f:
            f.write(text)
        note("benchmark written\n" + text)
    note("run done")
