'''
Unattended multi-phase run: training phases one after another, each limited by generations and/or hours.
At the end every result is benchmarked (evalModel.py, fixed seeds) and the table is written to
models/night_benchmark.txt. A crash in one phase does not stop the others.

    python nightRun.py                          all phases (see PHASES), then the benchmark
    python nightRun.py --phases a,b             only these phases (benchmark goes to night_benchmark_a_b.txt)
    python nightRun.py --scale 0.5              every phase's hours / generations x 0.5
    python nightRun.py --smoke                  tiny version to check that everything works (~1 minute)
    python nightRun.py --no-plot                no live learning-curve window

To plan a new experiment, edit section 2 (PHASES). Section 1 holds the building blocks, section 3 the machinery.

A phase is a dict (see RECIPE for all keys and their defaults). Where its network comes from:
    'start': model file | 'pool': pool file | 'after': name of an earlier phase | none of them: a new random network
    (inputs / hidden / outputs).
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

# =============================================================================================================
# 1. Building blocks
# =============================================================================================================

# The from-scratch recipe. Keys of a phase (every phase = RECIPE + its own changes):
#   inputs / hidden / outputs   network shape (inputs 25 + outputs 7 = target mode, 23 + 2 = plain)
#   minStart / maxStart / zero  start heights: random in minStart..maxStart (maxStart None = always 0); 'zero' = share of games that start at 0 anyway
#   sigma / mutRate             mutation size / share of mutated weights
#   smooth                      steer smoothing (1 = off)
#   stable                      stable slot observation: False = classic, True = 3/2/1/2 slots, or a tuple (platformsBelow, platformsAbove, monstersBelow, monstersAbove)
#   monster / monsterMult       monster practice: share of games with monsterMult x the normal monster spawn chance
#   gens / hours                limits of the phase (whichever comes first)
#   fast, repeat, snap          numba env, action repeat, generations at which snapshots are saved
RECIPE = dict(inputs=25, outputs=7, hidden=[23, 23, 23], minStart=0, maxStart=None, zero=0.0, sigma=0.05, mutRate=1.0,
              smooth=0.5, stable=True, gens=344, hours=2.0, fast=True)

# Population / evaluation settings: (full run, smoke test)
SETTINGS = dict(
    workers=(24, 4), population=(200, 12), games=(200, 6), stage1Games=(20, 3), elites=(3, 1), keepPart=(0.1, 0.25), finalistFraction=(0.3, 0.5))

# Models that are always benchmarked as a comparison (they use stableSlots + smoothing 0.5)
TARGET = "models/target_2026-10-05_10-07-53.pt"
CAREFUL = "models/careful_2026-10-05_18-43-51.pt"   # best so far: 7,485 / 3,106 / 4,502 / 4,968 / 2,188
REFERENCES = [TARGET, CAREFUL]
CONTROL_GLOBS = ["tgtScratchA_*", "tgtFast[BCD]_*", "noTarget[12]_*", "slotBase[12]_*"]    # replicates of the 344-gen control recipes

# Benchmark columns after 'start 0': hard starts, mixed starts, half at 0 + half hard
RANGES = ["20000:30000", "0:30000", "20000:30000:0.5"]


def weights(inputs, hidden, outputs):
    '''Number of weights + biases of a network.'''
    sizes = [inputs] + hidden + [outputs]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def scaledGens(inputs, hidden, outputs, controlWeights, baseGens=344):
    '''Generations for a network with more weights: baseGens x (its weights / the control's weights).'''
    return round(baseGens * weights(inputs, hidden, outputs) / controlWeights)


def slotPhase(name, layout, controlWeights):
    '''A no-target network for a slot layout (platformsBelow, platformsAbove, monstersBelow, monstersAbove).
    inputs = 2 + 3 * platforms + 2 * monsters; hidden width = input size, so the first layer is no bottleneck;
    generations are scaled with the weight count relative to the control.'''
    n = 2 + 3 * (layout[0] + layout[1]) + 2 * (layout[2] + layout[3])
    hidden = [n, n, n]
    return dict(RECIPE, name=name, stable=layout, inputs=n, outputs=2, hidden=hidden, gens=scaledGens(n, hidden, 2, controlWeights))


# =============================================================================================================
# 2. The experiment (edit this)
# =============================================================================================================
# Overnight (2026-10-07): ONE long polish run on a model trained FROM SCRATCH in the current setup (stable slots 3/2/1/2, no target mode, smoothing 0.5):
# the pool of slotBase2 (344 gens from scratch: 7,160 / 2,776, half-at-0 + half-hard 5,161 = the best of the from-scratch models on the training objective).
# Same polish recipe that gave `careful` its +29% hard / +9% start 0: 50/50 mix (half the games start at 0, half at 20-30k), careful mutations.
# Snapshots to see the progress. (Earlier screening rounds - slot layouts, hidden widths - found nothing beating the default; see AGENTS.md 2p-2r.)
PHASES = [
    dict(RECIPE, name="polishBase", inputs=23, outputs=2, pool="models/slotBase2_2026-10-06_18-36-05_pool.pt", minStart=20000, maxStart=30000, zero=0.5,
         sigma=0.02, mutRate=0.7, gens=100000, hours=8.0, snap=(1000, 2500, 5000)),
]


# =============================================================================================================
# 3. Machinery
# =============================================================================================================

def isFinalModel(path):
    '''A phase's result file, not its pool (_pool.pt) or a snapshot (_g<gen>.pt).'''
    return not path.endswith("_pool.pt") and not re.search(r"_g\d+\.pt$", path)


def startNetworkArgs(ph, results):
    '''How the phase gets its network: keyword arguments for sprudlerTrainingLoop.'''
    if ph.get("after"):
        if ph["after"] not in results:
            raise RuntimeError(f"phase '{ph['after']}' has no result - skipping")
        return dict(startNetwork=ps.loadNetwork(results[ph["after"]]))
    if ph.get("pool"):
        return dict(startNetwork=ps.loadPool(ph["pool"]))
    if ph.get("start"):
        return dict(startNetwork=ps.loadNetwork(ph["start"]))
    return dict(inputSize=ph["inputs"], hiddenSizes=ph["hidden"], outputSize=ph["outputs"])


def runPhase(ph, cfg, a, results, note):
    '''Trains one phase; on success stores its result file in results[name].'''
    gens = 12 if a.smoke else max(1, int(ph["gens"] * a.scale))
    hours = 0.0004 if a.smoke else ph["hours"] * a.scale
    note(f"phase {ph['name']}: {ph['inputs']} inputs, hidden {ph['hidden']}, starts {ph['minStart']}-{ph['maxStart']}, sigma {ph['sigma']}, "
         f"mutRate {ph['mutRate']}, up to {gens} gens / {hours:.2f} h")
    try:
        ps.sprudlerTrainingLoop(
            cfg["workers"], cfg["games"], gens, cfg["population"], ph["mutRate"], ph["sigma"], cfg["elites"], cfg["keepPart"],
            networkName=ph["name"], livePlot=False,
            minStartHeight=ph["minStart"], maxStartHeight=ph["maxStart"], zeroFraction=ph.get("zero", 0.0),
            snapshotAt=(5,) if a.smoke and ph.get("snap") else ph.get("snap", ()),
            steerSmoothing=ph.get("smooth", 1.0), stableSlots=ph.get("stable", False),
            monsterFraction=ph.get("monster", 0.0), monsterMult=ph.get("monsterMult", 1.0),
            fast=ph.get("fast", False), actionRepeat=ph.get("repeat", 1),
            stage1Games=cfg["stage1Games"], finalistFraction=cfg["finalistFraction"], maxHours=hours,
            **startNetworkArgs(ph, results))
        files = [f for f in glob.glob(f"models/{ph['name']}_*.pt") if isFinalModel(f)]
        results[ph["name"]] = max(files, key=os.path.getmtime)
        note(f"phase {ph['name']} finished -> {results[ph['name']]}")
    except BaseException:    # includes Ctrl-C inside a phase before its first generation ends
        note(f"phase {ph['name']} FAILED:\n{traceback.format_exc()}")


def slotsArgs(stable):
    '''evalModel arguments for a 'stable' setting.'''
    if isinstance(stable, tuple):
        return ["--slots", ",".join(map(str, stable))]
    return ["--stable"] if stable else []


def benchmark(phases, results, startTime, smoke):
    '''Benchmarks references, control models and all phase results; returns the text.
    evalModel applies ONE smoothing/slots setting to all its models, so models are benchmarked in groups of equal (smoothing, stable).'''
    snapshots = sorted(f for name in results for f in glob.glob(f"models/{name}_*_g[0-9]*.pt") if os.path.getmtime(f) > startTime)
    controls = sorted(f for pattern in CONTROL_GLOBS for f in glob.glob(f"models/{pattern}.pt") if isFinalModel(f))
    groups = {(0.5, True): REFERENCES + controls + snapshots}
    for ph in phases:
        if ph["name"] in results:
            groups.setdefault((ph.get("smooth", 1.0), ph.get("stable", False)), []).append(results[ph["name"]])
    text = ""
    for (smooth, stable), models in groups.items():
        cmd = [sys.executable, "evalModel.py"] + models + ["--smoothing", str(smooth), "--monster-column", "2", "--fast", "--ranges"] + RANGES + slotsArgs(stable)
        if smoke:
            cmd += ["--games", "50", "--workers", "4"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text += f"\n=== smoothing {smooth}, stableSlots {stable} ===" + out.stdout + (("\n" + out.stderr[-2000:]) if out.returncode else "")
    return text


def main():
    ap = argparse.ArgumentParser(description="Unattended multi-phase training run")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every phase's hours and generations (default 1)")
    ap.add_argument("--phases", help="comma-separated names: run only these phases (the benchmark goes to night_benchmark_<names>.txt)")
    ap.add_argument("--no-plot", action="store_true", help="do not open the live learning-curve window")
    ap.add_argument("--smoke", action="store_true", help="tiny test version")
    a = ap.parse_args()

    phases = [ph for ph in PHASES if not a.phases or ph["name"] in a.phases.split(",")]
    cfg = {key: values[1 if a.smoke else 0] for key, values in SETTINGS.items()}
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("models", exist_ok=True)

    def note(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join("models", "night_log.txt"), "a") as f:
            f.write(line + "\n")

    results = {}     # phase name -> best-network file
    startTime = time.time()
    note(f"run start: {[p['name'] for p in phases]}")
    if not a.smoke and not a.no_plot:    # ONE live window for the whole night: follows the newest run, chained to its parent runs
        subprocess.Popen([sys.executable, "plotRun.py", "--live", "--chain", "--latest"])
    for ph in phases:
        runPhase(ph, cfg, a, results, note)

    if results:
        note("benchmarking ...")
        text = benchmark(phases, results, startTime, a.smoke)
        fileName = "night_benchmark_smoke.txt" if a.smoke else (f"night_benchmark_{a.phases.replace(',', '_')}.txt" if a.phases else "night_benchmark.txt")    # a smoke test must never overwrite real results
        with open(os.path.join("models", fileName), "w") as f:
            f.write(text)
        note("benchmark written\n" + text)
    note("run done")


if __name__ == "__main__":
    main()
