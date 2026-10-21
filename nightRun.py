'''
Unattended multi-phase run: training phases one after another, each limited by generations and/or hours.
At the end every result is benchmarked (evalModel.py, fixed seeds) and the table is written to
models/night_benchmark.txt. A crash in one phase does not stop the others.

    python nightRun.py                          all phases of the default experiment (see EXPERIMENTS), then the benchmark
    python nightRun.py --experiment features    another experiment
    python nightRun.py --phases a,b             only these phases (benchmark goes to night_benchmark_a_b.txt)
    python nightRun.py --scale 0.5              every phase's hours / generations x 0.5
    python nightRun.py --smoke                  tiny version to check that everything works (~1 minute)
    python nightRun.py --no-plot                no live learning-curve window

To plan a new experiment, edit section 2 (the phase lists in EXPERIMENTS). Section 1 holds the building blocks, section 3 the machinery.

A phase is a dict (see RECIPE for all keys and their defaults). Where its network comes from:
    'start': model file | 'pool': pool file | 'after': name of an earlier phase (its best network) | 'afterPool': name of an earlier phase (its whole pool)
    | none of them: a new random network
    (inputs / hidden / outputs).
'''
# One worker process per hardware thread already uses every core: numpy's BLAS must NOT start helper threads of its own (they spin and oversubscribe
# the machine: 31 workers x ~2.5 busy threads each, load average ~96 on 32 threads, 4x slower generations). Must be set before numpy is imported.
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
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
#   replace                     share of weights replaced by a fresh Kaiming-range value on top of the noise (default 0 = off)
#   smooth                      steer smoothing (1 = off)
#   stable                      stable slot observation: False = classic, True = 3/2/1/2 slots, or a tuple (platformsBelow, platformsAbove, monstersBelow, monstersAbove)
#   monster / monsterMult       monster practice: share of games with monsterMult x the normal monster spawn chance
#   gens / hours                limits of the phase (whichever comes first)
#   fast, repeat, snap          numba env, action repeat, generations at which snapshots are saved
#   games                       games per network per generation (default: SETTINGS)
#   es / lr                     es=True: train with Evolution Strategies (proSprudler.trainingLoopES) instead of the GA; lr = its learning rate.
#                               ES uses sigma, games, population; mutRate / replace / elites / keepPart / stage1Games are GA-only
RECIPE = dict(inputs=25, outputs=7, hidden=[23, 23, 23], minStart=0, maxStart=None, zero=0.0, sigma=0.05, mutRate=1.0,
              smooth=0.5, stable=True, gens=344, hours=2.0, fast=True)

# Population / evaluation settings: (full run, smoke test)
SETTINGS = dict(
    workers=(31, 4), population=(200, 12), games=(200, 6), stage1Games=(20, 3), elites=(3, 1), keepPart=(0.1, 0.25), finalistFraction=(0.3, 0.5))

# Models that are always benchmarked as a comparison (they use stableSlots + smoothing 0.5)
TARGET = "models/target_2026-10-05_10-07-53.pt"
CAREFUL = "models/careful_2026-10-05_18-43-51.pt"   # best so far: 7,485 / 3,106 / 4,502 / 4,968 / 2,188
REFERENCES = [TARGET, CAREFUL]
CONTROL_GLOBS = ["tgtScratchA_*", "tgtFast[BCD]_*", "noTarget[12]_*", "slotBase[12]_*", "polishBase_*", "polishBase2_*", "polishBase3_*", "base0_*", "polishA_*", "polishB_*", "polishC_*"]    # replicates of the 344-gen control recipes

# Extra comparison models for a slot layout other than the default (benchmarked together with the phases that use the same layout)
REFERENCE_GROUPS = {(0.5, (3, 2, 1, 2, 0, 1)): ["featLand_[12]_*", "polishLand_2026-10-08_13-46-00", "ctrlA_2026-10-19_18-45-03", "ctrlB_2026-10-19_22-02-08", "scrCtrl[AB]_*"], (0.5, (3, 2, 1, 2, 1, 1)): ["featBoth_[12]_*"], (0.5, (3, 2, 1, 2, 1, 0)): ["featOcc_[12]_*"]}

# Benchmark columns after 'start 0': hard starts, mixed starts, half at 0 + half hard
RANGES = ["20000:30000", "0:30000", "20000:30000:0.5"]


def weights(inputs, hidden, outputs):
    '''Number of weights + biases of a network.'''
    sizes = [inputs] + hidden + [outputs]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def scaledGens(inputs, hidden, outputs, controlWeights, baseGens=344):
    '''Generations for a network with more weights: baseGens x (its weights / the control's weights).'''
    return round(baseGens * weights(inputs, hidden, outputs) / controlWeights)


def obsSize(layout):
    '''Number of network inputs for a stable-slot layout (platformsBelow, platformsAbove, monstersBelow, monstersAbove[, occupied, landing]).'''
    platforms = layout[0] + layout[1]
    extras = (1 if len(layout) > 4 and layout[4] else 0) + (2 if len(layout) > 5 and layout[5] else 0)
    return 2 + 3 * platforms + 2 * (layout[2] + layout[3]) + platforms * extras


def slotPhase(name, layout, controlWeights):
    '''A no-target network for a slot layout (platformsBelow, platformsAbove, monstersBelow, monstersAbove).
    inputs = 2 + 3 * platforms + 2 * monsters; hidden width = input size, so the first layer is no bottleneck;
    generations are scaled with the weight count relative to the control.'''
    n = obsSize(layout)
    hidden = [n, n, n]
    return dict(RECIPE, name=name, stable=layout, inputs=n, outputs=2, hidden=hidden, gens=scaledGens(n, hidden, 2, controlWeights))


# =============================================================================================================
# 2. The experiment (edit this)
# =============================================================================================================
# Three polish levers, compared on ONE shared from-scratch base (2026-10-07; the continuation of the plateaued polishBase lineage lost start-0 skill):
#   base0      from scratch, normal starts, 344 gens (random nets die instantly on hard starts, so the 50/50 mix cannot be used from gen 0)
#   polishA    baseline polish: 200 games per network, half the games start at 0, half at 20-30k
#   polishB    more games per network (800): less noisy selection
#   polishC    mix shifted toward the early game: 70% of the games start at 0, 30% at 20-30k
# All three branch off the SAME pool (afterPool), same sigma / mutRate, and get the same wall-clock time (3 h; B plays ~3.4x more games per generation,
# so it gets fewer generations). Snapshots at 500 / 1,000 gens to compare at equal generations too.
POLISH = dict(RECIPE, inputs=23, outputs=2, afterPool="base0", minStart=20000, maxStart=30000, zero=0.5, sigma=0.02, mutRate=0.7, gens=100000, hours=3.0, snap=(500, 1000))
POLISH_EXPERIMENT = [
    dict(RECIPE, name="base0", inputs=23, outputs=2, gens=344, hours=1.0),
    dict(POLISH, name="polishA"),
    dict(POLISH, name="polishB", games=800),
    dict(POLISH, name="polishC", zero=0.7),
]


# New observation features (2026-10-07), screened from scratch against the 344-gen controls slotBase1/2 + noTarget1/2 (23 in / 2 out / [23]x3):
#   occupied: a monster stands on that platform (1 float per platform slot)        -> 28 inputs
#   landing:  [T, r] = frames until the feet land on that platform (vertical motion is independent of steering) and how much of the available
#             steering is needed to get there (2 floats per platform slot)           -> 33 inputs
#   both                                                                              -> 38 inputs
# generations scaled with the weight count relative to the control, 2 replicates each, ~1.5 h on the 7950X.
CONTROL_SLOTS = weights(23, [23] * 3, 2)    # 1,704 weights
FEATURES_EXPERIMENT = [slotPhase(f"{name}_{r}", layout, CONTROL_SLOTS)
                       for name, layout in (("featOcc", (3, 2, 1, 2, 1, 0)), ("featLand", (3, 2, 1, 2, 0, 1)), ("featBoth", (3, 2, 1, 2, 1, 1))) for r in "12"]

# The polish branches again, but off the already finished base0 pool (no second base0 run): used by the combined experiment
BASE0_POOL = "models/base0_2026-10-07_20-46-50_pool.pt"    # from-scratch base, 344 gens, normal starts
POLISH_BRANCHES = [dict(ph, pool=BASE0_POOL, afterPool=None) for ph in POLISH_EXPERIMENT if ph["name"].startswith("polish")]

# Next round (2026-10-08), all 3 h, 50/50 mix unless noted, sigma 0.02 / mutRate 0.7, compared with polishA (200 games) / polishB (800) / polishC (70% at 0):
#   polishD     800 games + 70/30 mix          (do the two levers add up?)
#   polishLand  800 games, from the featLand_2 pool (landing prediction, 33 inputs), now WITH hard starts in the mix  (fair test of the feature; compare polishB)
#   polishBoth  800 games, from the featBoth_2 pool (occupied flag + landing prediction, 38 inputs)                    (compare polishB)
#   polishE     1,600 games                    (do more games keep helping? compare polishB 800 and polishA 200)
FEAT_POOLS = dict(land="models/featLand_2_2026-10-07_22-12-42_pool.pt", both="models/featBoth_2_2026-10-07_23-30-00_pool.pt")
NEXT_EXPERIMENT = [
    dict(POLISH, name="polishD", afterPool=None, pool=BASE0_POOL, games=800, zero=0.7),
    dict(POLISH, name="polishLand", afterPool=None, pool=FEAT_POOLS["land"], games=800, inputs=obsSize((3, 2, 1, 2, 0, 1)), hidden=[obsSize((3, 2, 1, 2, 0, 1))] * 3, stable=(3, 2, 1, 2, 0, 1)),
    dict(POLISH, name="polishBoth", afterPool=None, pool=FEAT_POOLS["both"], games=800, inputs=obsSize((3, 2, 1, 2, 1, 1)), hidden=[obsSize((3, 2, 1, 2, 1, 1))] * 3, stable=(3, 2, 1, 2, 1, 1)),
    dict(POLISH, name="polishE", afterPool=None, pool=BASE0_POOL, games=1600),
]


# Replace mutation (2026-10-19): on top of the gaussian noise, a share 'replace' of the weights gets a fresh random value (Kaiming range).
# Screening off the polishLand pool (current best), polishLand recipe (800 games, 50/50 mix, sigma 0.02 / mutRate 0.7), 1.5 h each,
# 2 replicates with replace 0.001 vs 2 same-recipe controls (replace 0), interleaved so a slow machine hits both arms alike.
LAND_LAYOUT = (3, 2, 1, 2, 0, 1)
POLISH_LAND_POOL = "models/polishLand_2026-10-08_13-46-00_pool.pt"
REPLACE_BASE = dict(POLISH, afterPool=None, pool=POLISH_LAND_POOL, games=800, inputs=obsSize(LAND_LAYOUT), hidden=[obsSize(LAND_LAYOUT)] * 3,
                    stable=LAND_LAYOUT, hours=1.5)
REPLACE_EXPERIMENT = [dict(REPLACE_BASE, name=f"{kind}{r}", replace=rate) for r in "AB" for kind, rate in (("repl", 0.001), ("ctrl", 0.0))]

# The same question from scratch (2026-10-20): the screening above only fine-tuned a finished pool, where fresh random weights have little to add.
# Landing layout (33 in / [33]x3 / 2 out), from-scratch recipe (344 gens scaled by weight count, normal starts), 2 replicates each:
# control (replace 0), replace 0.001 (the rate tested above) and a 10x higher rate 0.01. Interleaved so a slow machine hits all arms alike.
SCRATCH_REPLACE_EXPERIMENT = [dict(slotPhase(f"{kind}{r}", LAND_LAYOUT, CONTROL_SLOTS), replace=rate)
                              for r in "AB" for kind, rate in (("scrCtrl", 0.0), ("scrRepl", 0.001), ("scrHi", 0.01))]

# Evolution Strategies (2026-10-21): one parent, 200 children = parent + sigma * noise, every child plays all its games, the parent moves
# along the rank-weighted average noise (learning rate lr). First test off the polishLand pool (its best network = the parent), 1.5 h each,
# 2 replicates; the GA controls are ctrlA / ctrlB of the 'replace' experiment (same pool, same 1.5 h, benchmark 8,419 / 4,001 / 5,137 / 6,266 / 2,907 avg).
# 200 games per child (not 800): ES has no two-stage shortcut, so 800 would cost ~3x the GA's time per generation; 200 is about the same.
# lr 0.5: with N = 200 and weights +-0.5 the average vote is ~0.02 per weight, so each generation moves a weight ~0.01 (half of sigma 0.02).
ES_BASE = dict(REPLACE_BASE, es=True, lr=0.5, games=200, snap=())
ES_EXPERIMENT = [dict(ES_BASE, name=f"es{r}") for r in "AB"]

# ES long runs (2026-10-22, ~12 h): does ES break the GA plateau (~8,000-8,400 at start 0 / ~4,000 hard, more GA hours bought nothing)?
# Not a speed race: each run gets long enough to reach its OWN plateau; compared ceiling vs ceiling (one run per setting, so only a clear gain counts).
#   esLong05 / esLong025    polishLand pool (the GA's best), lr 0.5 / 0.25, 4 h each                                  -> vs polishLand, ctrlA / ctrlB
#   esScr05 -> esScrPolish05 from scratch, the GA lineage's two steps: 1 h normal starts, then 3 h 50/50 mix off its pool  -> does ES find its own way up?
# ~5-7 s per generation (200 children x 200 games), so 4 h ~ 2,000-2,700 gens; snapshots show where the climb stops.
ES_LONG_EXPERIMENT = [
    dict(ES_BASE, name="esLong05", lr=0.5, hours=4.0, snap=(500, 1000, 2000)),
    dict(slotPhase("esScr05", LAND_LAYOUT, CONTROL_SLOTS), es=True, lr=0.5, gens=100000, hours=1.0),
    dict(ES_BASE, name="esScrPolish05", lr=0.5, hours=3.0, pool=None, afterPool="esScr05", snap=(500, 1000)),
    dict(ES_BASE, name="esLong025", lr=0.25, hours=4.0, snap=(500, 1000, 2000)),
]


# 'all': the cheap feature screening first (~1.5 h; may change what is worth polishing), then the three polish branches (3 h each)
EXPERIMENTS = {"esLong": ES_LONG_EXPERIMENT, "es": ES_EXPERIMENT, "scratchReplace": SCRATCH_REPLACE_EXPERIMENT, "replace": REPLACE_EXPERIMENT, "next": NEXT_EXPERIMENT, "all": FEATURES_EXPERIMENT + POLISH_BRANCHES, "polish": POLISH_EXPERIMENT, "features": FEATURES_EXPERIMENT}    # python nightRun.py --experiment features


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
    if ph.get("afterPool"):    # the whole gene pool of an earlier phase (keeps the diversity; several phases can branch off the same pool)
        if ph["afterPool"] not in results:
            raise RuntimeError(f"phase '{ph['afterPool']}' has no result - skipping")
        return dict(startNetwork=ps.loadPool(results[ph["afterPool"]].replace(".pt", "_pool.pt")))
    if ph.get("pool"):
        return dict(startNetwork=ps.loadPool(ph["pool"]))
    if ph.get("start"):
        return dict(startNetwork=ps.loadNetwork(ph["start"]))
    return dict(inputSize=ph["inputs"], hiddenSizes=ph["hidden"], outputSize=ph["outputs"])


def runPhase(ph, cfg, a, results, note):
    '''Trains one phase; on success stores its result file in results[name].'''
    gens = 12 if a.smoke else max(1, int(ph["gens"] * a.scale))
    hours = 0.0004 if a.smoke else ph["hours"] * a.scale
    method = f"ES, lr {ph['lr']}" if ph.get("es") else f"GA, mutRate {ph['mutRate']}, replace {ph.get('replace', 0.0)}"
    note(f"phase {ph['name']}: {ph['inputs']} inputs, hidden {ph['hidden']}, starts {ph['minStart']}-{ph['maxStart']}, sigma {ph['sigma']}, "
         f"{method}, up to {gens} gens / {hours:.2f} h")
    try:
        games = cfg["games"] if a.smoke else ph.get("games", cfg["games"])
        common = dict(networkName=ph["name"], livePlot=False,
                      minStartHeight=ph["minStart"], maxStartHeight=ph["maxStart"], zeroFraction=ph.get("zero", 0.0),
                      steerSmoothing=ph.get("smooth", 1.0), stableSlots=ph.get("stable", False),
                      monsterFraction=ph.get("monster", 0.0), monsterMult=ph.get("monsterMult", 1.0),
                      fast=ph.get("fast", False), actionRepeat=ph.get("repeat", 1), maxHours=hours,
                      **startNetworkArgs(ph, results))
        if ph.get("es"):
            ps.trainingLoopES(cfg["workers"], games, gens, cfg["population"], ph["mutRate"], ph["sigma"], cfg["elites"], ph["lr"], cfg["keepPart"],
                              snapshotAt=(5,) if a.smoke and ph.get("snap") else ph.get("snap", ()), **common)
        else:
            ps.sprudlerTrainingLoop(
                cfg["workers"], games, gens, cfg["population"], ph["mutRate"], ph["sigma"], cfg["elites"], cfg["keepPart"],
                snapshotAt=(5,) if a.smoke and ph.get("snap") else ph.get("snap", ()), replaceRate=ph.get("replace", 0.0),
                stage1Games=cfg["stage1Games"], finalistFraction=cfg["finalistFraction"], **common)
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
    groups = {(0.5, True): REFERENCES + controls}
    for f in snapshots:    # a snapshot belongs to the slot layout of its own phase (otherwise it is evaluated with the wrong observation)
        owner = next((ph for ph in phases if os.path.basename(f).startswith(ph["name"] + "_")), None)
        key = (owner.get("smooth", 1.0), owner.get("stable", False)) if owner else (0.5, True)
        groups.setdefault(key, []).append(f)
    for key, patterns in REFERENCE_GROUPS.items():
        if any(ph.get("stable") == key[1] and ph.get("smooth", 1.0) == key[0] for ph in phases):    # only if a phase of this layout is benchmarked
            groups.setdefault(key, []).extend(sorted(f for pattern in patterns for f in glob.glob(f"models/{pattern}.pt") if isFinalModel(f)))
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
    ap.add_argument("--experiment", default="esLong", choices=sorted(EXPERIMENTS), help="which list of phases (see EXPERIMENTS in section 2; default esLong)")
    ap.add_argument("--phases", help="comma-separated names: run only these phases (the benchmark goes to night_benchmark_<names>.txt)")
    ap.add_argument("--no-plot", action="store_true", help="do not open the live learning-curve window")
    ap.add_argument("--smoke", action="store_true", help="tiny test version")
    a = ap.parse_args()

    phases = [ph for ph in EXPERIMENTS[a.experiment] if not a.phases or ph["name"] in a.phases.split(",")]
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
