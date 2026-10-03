'''
Unattended multi-phase run: training phases one after another, each limited by generations and/or hours.
A phase starts from scratch, from a model file, or from the result of an earlier phase ('after').
At the end every result is benchmarked (evalModel.py, fixed seeds) and the table is written to
models/night_benchmark.txt. A crash in one phase does not stop the others (a phase whose 'after' phase failed is skipped).

    python nightRun.py                 all phases (see PHASES), then the benchmark
    python nightRun.py --scale 0.5     every phase's hours/generations x 0.5
    python nightRun.py --smoke         tiny version to check that everything works (~1 minute)

Current experiment: does the SHAPE of the network matter? Four from-scratch runs with the recipe of the 23-input model 17-00-59
(normal starts, 1,000 gens): a replicate of [23,23,23] (noise check), [32,32,32], [48,48,48] and [23,23,23,23].
Compare their benchmark columns with 17-00-59 (2,084 / 328 / 600 on start 0 / hard 20-30k / mixed 0-30k); identical runs differ by ~15%.
'''
import argparse
import glob
import os
import subprocess
import sys
import time
import traceback

import proSprudler as ps

BASELINE = "models/sprudler_2026-10-02_17-00-59.pt"   # 23-input from-scratch model (1,000 gens, normal starts)

# start: model file | None (+ 'after': name of an earlier phase | None -> new network of 'inputs' inputs)
PHASES = [
    # same recipe as the 23-input model 17-00-59 (from scratch, normal starts, 1,000 gens, sigma 0.05, mutRate 1.0) - only the shape differs
    dict(name="shapeReplicate", start=None, after=None, inputs=23, hidden=[23, 23, 23],     gens=1000, hours=3.0, minStart=0, maxStart=None, sigma=0.05, mutRate=1.0),
    dict(name="shapeW32",       start=None, after=None, inputs=23, hidden=[32, 32, 32],     gens=1000, hours=3.0, minStart=0, maxStart=None, sigma=0.05, mutRate=1.0),
    dict(name="shapeW48",       start=None, after=None, inputs=23, hidden=[48, 48, 48],     gens=1000, hours=3.0, minStart=0, maxStart=None, sigma=0.05, mutRate=1.0),
    dict(name="shapeD4",        start=None, after=None, inputs=23, hidden=[23, 23, 23, 23], gens=1000, hours=3.0, minStart=0, maxStart=None, sigma=0.05, mutRate=1.0),
]
RANGES = ["20000:30000", "0:30000"]    # benchmark columns after 'start 0': hard starts, mixed starts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unattended multi-phase training run")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every phase's hours and generations (default 1)")
    ap.add_argument("--no-plot", action="store_true", help="do not open the live learning-curve window")
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

    workers, pop, games, stage1, elites, keep, frac = (4, 12, 6, 3, 1, 0.25, 0.5) if a.smoke else (24, 200, 200, 20, 3, 0.1, 0.3)
    outputs = {}     # phase name -> best-network file
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
            if ph["after"]:
                if ph["after"] not in outputs:
                    raise RuntimeError(f"phase '{ph['after']}' has no result - skipping")
                kw["startNetwork"] = ps.loadNetwork(outputs[ph["after"]])
            elif ph["start"]:
                kw["startNetwork"] = ps.loadNetwork(ph["start"])
            else:
                kw.update(inputSize=ph["inputs"], hiddenSizes=ph.get("hidden", [23, 23, 23]), outputSize=2)
            ps.sprudlerTrainingLoop(workers, games, gens, pop, ph["mutRate"], ph["sigma"], elites, keep, networkName=ph["name"], livePlot=False,
                                    minStartHeight=ph["minStart"], maxStartHeight=ph["maxStart"],
                                    stage1Games=stage1, finalistFraction=frac, maxHours=hours, **kw)
            files = [f for f in glob.glob(f"models/{ph['name']}_*.pt") if not f.endswith("_pool.pt")]
            outputs[ph["name"]] = max(files, key=os.path.getmtime)
            note(f"phase {ph['name']} finished -> {outputs[ph['name']]}")
        except BaseException:                       # includes Ctrl-C inside a phase before its first generation ends
            note(f"phase {ph['name']} FAILED:\n{traceback.format_exc()}")

    if outputs:
        note("benchmarking ...")
        cmd = [sys.executable, "evalModel.py", BASELINE] + list(outputs.values()) + ["--ranges"] + RANGES
        if a.smoke:
            cmd += ["--games", "50", "--workers", "4"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = out.stdout + (("\n" + out.stderr[-2000:]) if out.returncode else "")
        with open(os.path.join("models", "night_benchmark.txt"), "w") as f:
            f.write(text)
        note("benchmark written to models/night_benchmark.txt\n" + text)
    note("run done")
