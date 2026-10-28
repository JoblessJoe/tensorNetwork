'''
Unattended multi-phase run: training phases one after another, each limited by generations and/or hours.
At the end every result is benchmarked (evalModel.py, fixed seeds) and the table is written to
models/night_benchmark_<experiment>.txt. A crash in one phase does not stop the others.

    python nightRun.py                          all phases of the default experiment (see EXPERIMENTS), then the benchmark
    python nightRun.py --experiment features    another experiment
    python nightRun.py --phases a,b             only these phases (benchmark goes to night_benchmark_<experiment>_a_b.txt)
    python nightRun.py --scale 0.5              every phase's hours / generations x 0.5
    python nightRun.py --workers 19             other number of worker processes (default SETTINGS: 31 = 32-thread PC; 20-thread server: 19)
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
from proSprudler import OBSERVATION_LENGTH

# =============================================================================================================
# 1. Building blocks
# =============================================================================================================

# The from-scratch recipe. Keys of a phase (every phase = RECIPE + its own changes):
#   inputs / hidden / outputs   network shape (the observation has OBSERVATION_LENGTH = 33 floats, 2 outputs: steer, shoot)
#   minStart / maxStart / zero  start heights: random in minStart..maxStart (maxStart None = always 0); 'zero' = share of games that start at 0 anyway
#   sigma / mutRate             mutation size / share of mutated weights
#   replace                     share of weights replaced by a fresh Kaiming-range value on top of the noise (default 0 = off)
#   smooth                      steer smoothing (1 = off)
#   monster / monsterMult       monster practice: share of games with monsterMult x the normal monster spawn chance
#   gens / hours                limits of the phase (whichever comes first)
#   fast, repeat, snap          numba env, action repeat, generations at which snapshots are saved
#   games                       games per network per generation (default: SETTINGS)
#   population                  networks per generation (default: SETTINGS). A multiple of the workers does NOT save time worth mentioning
#                               (measured ~1%: games differ in length, so the workers finish at different times anyway; AGENTS.md 2zc)
#   es / lr                     es=True: train with Evolution Strategies (proSprudler.trainingLoopES) instead of the GA; lr = its learning rate.
#   penalties                   (ES + fast only, optional) (shotFree, shotAll, steer[, shotMiss[, detour]]): points subtracted per game for bullets fired with no monster on screen,
#                               for every bullet, per unit of summed steer change between decisions, for every bullet that hit nothing, and per px of detour (sideways distance travelled beyond the net displacement of a jump) (proSprudler.behaviourPenalty); default (0, 0, 0)
#   sigmaEnd / lrEnd            (ES only, optional) sigma / lr move geometrically from `sigma` / `lr` to this value over the phase (by the clock, via `hours`)
#                               ES uses sigma, games, population; mutRate / replace / elites / keepPart / stage1Games are GA-only
RECIPE = dict(inputs=OBSERVATION_LENGTH, outputs=2, hidden=[OBSERVATION_LENGTH] * 3, minStart=0, maxStart=None, zero=0.0, sigma=0.05, mutRate=1.0,
              smooth=0.5, gens=344, hours=2.0, fast=True)

# Population / evaluation settings: (full run, smoke test)
SETTINGS = dict(
    workers=(31, 4), population=(200, 12), games=(200, 6), stage1Games=(20, 3), elites=(3, 1), keepPart=(0.1, 0.25), finalistFraction=(0.3, 0.5))

# Benchmark columns after 'start 0': hard starts, mixed starts, half at 0 + half hard
RANGES = ["20000:30000", "0:30000", "20000:30000:0.5"]


def weights(inputs, hidden, outputs):
    '''Number of weights + biases of a network.'''
    sizes = [inputs] + hidden + [outputs]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def scaledGens(inputs, hidden, outputs, controlWeights, baseGens=344):
    '''Generations for a network with more weights: baseGens x (its weights / the control's weights).'''
    return round(baseGens * weights(inputs, hidden, outputs) / controlWeights)


def slotPhase(name, controlWeights):
    '''A network phase with the fixed observation (33 inputs); hidden width = input size, so the first layer is no bottleneck;
    generations are scaled with the weight count relative to the control.'''
    n = OBSERVATION_LENGTH
    hidden = [n, n, n]
    return dict(RECIPE, name=name, inputs=n, outputs=2, hidden=hidden, gens=scaledGens(n, hidden, 2, controlWeights))


# =============================================================================================================
# 2. The experiment (edit this)
# =============================================================================================================
# The GA polish recipe (sigma 0.02, mutRate 0.7, hard starts in the mix, 3 h); REPLACE_BASE builds on it. (The earlier polish / feature experiments need observation
# layouts that no longer exist and were removed; their results are in AGENTS.md.)
POLISH = dict(RECIPE, afterPool=None, minStart=20000, maxStart=30000, zero=0.5, sigma=0.02, mutRate=0.7, gens=100000, hours=3.0, snap=(500, 1000))
CONTROL_SLOTS = weights(23, [23] * 3, 2)    # 1,704 weights: the control of the early 23-input screens; generations of a phase scale with weights / this

# Replace mutation (2026-10-19): on top of the gaussian noise, a share 'replace' of the weights gets a fresh random value (Kaiming range).
# Screening off the polishLand pool (current best), polishLand recipe (800 games, 50/50 mix, sigma 0.02 / mutRate 0.7), 1.5 h each,
# 2 replicates with replace 0.001 vs 2 same-recipe controls (replace 0), interleaved so a slow machine hits both arms alike.
POLISH_LAND_POOL = "models/polishLand_2026-10-08_13-46-00_pool.pt"
REPLACE_BASE = dict(POLISH, afterPool=None, pool=POLISH_LAND_POOL, games=800, hours=1.5)
REPLACE_EXPERIMENT = [dict(REPLACE_BASE, name=f"{kind}{r}", replace=rate) for r in "AB" for kind, rate in (("repl", 0.001), ("ctrl", 0.0))]

# The same question from scratch (2026-10-20): the screening above only fine-tuned a finished pool, where fresh random weights have little to add.
# Landing layout (33 in / [33]x3 / 2 out), from-scratch recipe (344 gens scaled by weight count, normal starts), 2 replicates each:
# control (replace 0), replace 0.001 (the rate tested above) and a 10x higher rate 0.01. Interleaved so a slow machine hits all arms alike.
SCRATCH_REPLACE_EXPERIMENT = [dict(slotPhase(f"{kind}{r}", CONTROL_SLOTS), replace=rate)
                              for r in "AB" for kind, rate in (("scrCtrl", 0.0), ("scrRepl", 0.001), ("scrHi", 0.01))]

# Evolution Strategies (2026-10-21): one parent, 200 children = parent + sigma * noise, every child plays all its games, the parent moves
# along the rank-weighted average noise (learning rate lr). First test off the polishLand pool (its best network = the parent), 1.5 h each,
# 2 replicates; the GA controls are ctrlA / ctrlB of the 'replace' experiment (same pool, same 1.5 h, benchmark 8,419 / 4,001 / 5,137 / 6,266 / 2,907 avg).
# 200 games per child (not 800): ES has no two-stage shortcut, so 800 would cost ~3x the GA's time per generation; 200 is about the same.
# lr 0.5: with N = 200 and weights +-0.5 the average vote is ~0.02 per weight, so each generation moves a weight ~0.01 (half of sigma 0.02).
ES_BASE = dict(REPLACE_BASE, es=True, lr=0.5, games=200, snap=())
ES_EXPERIMENT = [dict(ES_BASE, name=f"es{r}") for r in "AB"]

# ES long runs (2026-10-21, ~12 h): does ES break the GA plateau (~8,000-8,400 at start 0 / ~4,000 hard, more GA hours bought nothing)?
# Not a speed race: each run gets long enough to reach its OWN plateau; compared ceiling vs ceiling (one run per setting, so only a clear gain counts).
#   esLong05 / esLong025    polishLand pool (the GA's best), lr 0.5 / 0.25, 4 h each                                  -> vs polishLand, ctrlA / ctrlB
#   esScr05 -> esScrPolish05 from scratch, the GA lineage's two steps: 1 h normal starts, then 3 h 50/50 mix off its pool  -> does ES find its own way up?
# ~5-7 s per generation (200 children x 200 games), so 4 h ~ 2,000-2,700 gens; snapshots show where the climb stops.
ES_LONG_EXPERIMENT = [
    dict(ES_BASE, name="esLong05", lr=0.5, hours=4.0, snap=(500, 1000, 2000)),
    dict(slotPhase("esScr05", CONTROL_SLOTS), es=True, lr=0.5, gens=100000, hours=1.0),
    dict(ES_BASE, name="esScrPolish05", lr=0.5, hours=3.0, pool=None, afterPool="esScr05", snap=(500, 1000)),
    dict(ES_BASE, name="esLong025", lr=0.25, hours=4.0, snap=(500, 1000, 2000)),
]

# ES long runs, round 2 (2026-10-22): round 1 gave esLong025 9,642 / 5,423 vs esLong05 9,115 / 4,934 (one run each, lr 0.25 looked better).
#   esLong025b   replicate of esLong025 (same recipe)    -> is the lr 0.25 result luck?
#   esLong01     lr 0.1, everything else equal           -> does the trend "smaller lr is better" continue?
ES_LONG2_EXPERIMENT = [
    dict(ES_BASE, name="esLong025b", lr=0.25, hours=4.0, snap=(500, 1000, 2000)),
    dict(ES_BASE, name="esLong01", lr=0.1, hours=4.0, snap=(500, 1000, 2000)),
]

# Action repeat from scratch (2026-10-22): the network decides only every k-th frame and its action is repeated in between, so a generation
# costs ~1/k of the network calls (repeat 2 ~1.7x faster per point climbed, AGENTS.md 2zb). Never trained with before; nets trained at 1 lose
# 18% at 2. Question: which k gives the best model after the same TIME (more generations per hour vs coarser control)?
# esScr05 recipe (landing layout, ES lr 0.5, normal starts), 1 h each, k = 1 (control) .. 5, 2 replicates, interleaved.
# Smoothing per decision 1 - 0.5^k: the steering moves as fast per game frame as with 0.5 at every frame, so only the decision rate differs.
# Snapshots at equal generations answer "per generation" as well. Benchmarked at each model's own repeat.
REPEAT_EXPERIMENT = [dict(slotPhase(f"rep{k}{r}", CONTROL_SLOTS), es=True, lr=0.5, gens=100000, hours=1.0,
                          repeat=k, smooth=1 - 0.5 ** k, snap=(250, 500, 1000))
                     for r in "AB" for k in range(1, 6)]

# Action repeat, round 2 (2026-10-23): round 1 (AGENTS.md 2zd) had repeat 5 best per hour (10,906 start 0) and not saturated. Arms 6 / 8 / 10,
# 2 replicates, plus one repeat-5 control (rep5C): the fused loop (7ef9bd4) makes generations ~6-11% faster than in round 1, so the control
# measures that edge for the per-hour comparison (per generation the snapshots are unaffected). Same recipe, 1 h each, smoothing 1 - 0.5^k.
REPEAT2_EXPERIMENT = ([dict(slotPhase("rep5C", CONTROL_SLOTS), es=True, lr=0.5, gens=100000, hours=1.0, repeat=5, smooth=1 - 0.5 ** 5,
                            snap=(250, 500, 1000, 2000, 3000))]
                      + [dict(slotPhase(f"rep{k}{r}", CONTROL_SLOTS), es=True, lr=0.5, gens=100000, hours=1.0,
                              repeat=k, smooth=1 - 0.5 ** k, snap=(250, 500, 1000, 2000, 3000))
                         for r in "AB" for k in (6, 8, 10)])

# ES learning rate from scratch (2026-10-22, server, 19 workers): lr was only ever compared off a trained pool (round 1: 0.25 9,642 / 5,423 vs
# 0.5 9,115 / 4,934, one run each). Here: which lr climbs best from a random network in the same time? esScr05 recipe (= rep1 of 'repeat':
# landing layout, sigma 0.05, 200 children x 200 games, normal starts), lr 0.1 / 0.25 / 0.5 (control) / 1.0, 1 h each, 2 replicates, interleaved.
# The server is ~5x slower than the PC (~250-400 gens per hour), so this sees the early climb only, not the plateau.
ES_LR_EXPERIMENT = [dict(slotPhase(f"lr{str(lr).replace('.', '')}{r}", CONTROL_SLOTS), es=True, lr=lr, gens=100000, hours=1.0,
                         snap=(100, 250))
                    for r in "AB" for lr in (0.1, 0.25, 0.5, 1.0)]

# ES sigma from scratch (planned for the night after ES_LR): how far from the parent should the children be? ES so far used sigma 0.05
# (scratch) and 0.02 (polish) only. Same recipe, lr 0.5, sigma 0.02 / 0.05 (control) / 0.1, 1.5 h each, 2 replicates, interleaved.
ES_SIGMA_EXPERIMENT = [dict(slotPhase(f"sig{str(sg).replace('.', '')}{r}", CONTROL_SLOTS), es=True, lr=0.5, sigma=sg,
                            gens=100000, hours=1.5, snap=(100, 250, 500))
                       for r in "AB" for sg in (0.02, 0.05, 0.1)]


# Sigma x repeat, and a repeat-10 replicate (2026-10-23, after AGENTS.md 2ze): sigma 0.1 won on the server (repeat 1, from scratch, 8.0-8.2k vs 4-6k at
# 0.05); all repeat runs used 0.05. Here sigma 0.1 / 0.15 on repeat 5, 2 replicates, sigma 0.2 once, against rep5C (sigma 0.05, 11,590 / 4,701) and a
# second control rep5D, plus a third repeat-10 run (sigma 0.05; rep10A/B had ~6.1k on hard starts, unreplicated). Same recipe as REPEAT2, 1 h each,
# interleaved; last a 4 h repeat-10 run rep10L (the 1 h runs were not saturated: where is the ceiling?).
SIGMA_REPEAT_EXPERIMENT = [dict(slotPhase(name, CONTROL_SLOTS), es=True, lr=0.5, sigma=sg, gens=100000, hours=hrs,
                                repeat=k, smooth=1 - 0.5 ** k, snap=snap)
                           for name, sg, k, hrs, snap in (("s10r5A", 0.1, 5, 1.0, (250, 500, 1000, 2000, 3000)), ("s15r5A", 0.15, 5, 1.0, (250, 500, 1000, 2000, 3000)),
                                                          ("rep10C", 0.05, 10, 1.0, (250, 500, 1000, 2000, 3000)), ("s10r5B", 0.1, 5, 1.0, (250, 500, 1000, 2000, 3000)),
                                                          ("s15r5B", 0.15, 5, 1.0, (250, 500, 1000, 2000, 3000)), ("rep5D", 0.05, 5, 1.0, (250, 500, 1000, 2000, 3000)),
                                                          ("s20r5A", 0.2, 5, 1.0, (250, 500, 1000, 2000, 3000)), ("rep10L", 0.05, 10, 4.0, (1000, 3000, 6000, 10000)))]

# Long replicates (2026-10-24): rep10L (repeat 10, sigma 0.05, 4 h from scratch) reached 16,535 / 10,589 (start 0 / hard) in ONE run, while the 1 h repeat-10
# replicate rep10C collapsed (3.6k). Which repeat is the most reliable over a long run? 4 h each, sigma 0.05, lr 0.5, repeat 5 / 7 / 10, smoothing 1 - 0.5^k.
# Order = one replicate of each arm first (12 h), then the second ones; rep10L is the first repeat-10 replicate, so the new repeat-10 runs are B and C.
REPEAT_LONG_EXPERIMENT = [dict(slotPhase(name, CONTROL_SLOTS), es=True, lr=0.5, sigma=0.05, gens=100000, hours=4.0,
                               repeat=k, smooth=1 - 0.5 ** k, snap=(1000, 3000, 6000, 10000))
                          for name, k in (("rl5A", 5), ("rl7A", 7), ("rl10B", 10), ("rl5B", 5), ("rl7B", 7), ("rl10C", 10))]

# Games per child, continued from the best net (2026-10-24, server): all ES runs so far used 200 children x 200 games. Noise in the ranking matters most
# near the ceiling (GA: 800 games helped in the polish stage), so continue the 4 h repeat-10 run (rep10L, 16.5k start 0) with the SAME recipe
# (repeat 10, sigma 0.05, lr 0.5, start 0 only) and only change the games per child: 200 (control) vs 800, 4 h each. Equal wall-clock, so the
# 800-game arm gets ~1/4 of the generations.
REP10_BEST_POOL = "models/rep10L_2026-10-24_02-52-46_pool.pt"
GAMES_PER_CHILD_EXPERIMENT = [dict(slotPhase(f"continueBest_{g}games", CONTROL_SLOTS), es=True, lr=0.5, sigma=0.05, pool=REP10_BEST_POOL,
                                   games=g, gens=100000, hours=4.0, repeat=10, smooth=1 - 0.5 ** 10, snap=(250, 500, 1000, 2000))
                              for g in (200, 800)]

# Fewer games per child (2026-10-24 evening): 800 games lost to 200 at equal wall-clock (server benchmark 17.3k / 11.9k vs 18.3k / 12.6k from the 16.5k / 10.6k net).
# ES averages the ranking noise over 200 children, so fewer games = more generations may win. Same continuation, 100 and 50 games per child, 4 h each.
# More children instead of more games (2026-10-25): same cost per generation as 200 children x 200 games (40,000 games), but 800 children x 50 games. lr stays 0.5: the useful part of the
# averaged vote does NOT shrink with N (only its noise does), so lr 1.0 doubled the step and the first update dropped the children from 15.7k to 12.2k. 2 h; control = the 200 x 200 arm.
CHILDREN_EXPERIMENT = [dict(GAMES_PER_CHILD_EXPERIMENT[0], name="continueBest_800children_50games", games=50, population=800, lr=0.5, hours=2.0)]
# Population scan (2026-10-25, PC): 800 children x 50 games matched the 200 x 200 control after half the time on the server (AGENTS.md 2zh). Same budget per
# generation (~40,000 games), children vs games per child, 1 h each, from the best 4 h repeat-5 net (30.0k start 0) at repeat 5: the 200 x 200 control, 800 x 50,
# 1600 x 25, 3200 x 12, and 800 x 50 with lr 0.25. lr 0.5 for the others (the useful part of the vote does not shrink with N, so no rescaling).
REP5_BEST_POOL = "models/rl5A_2026-10-24_10-51-51_pool.pt"
POPULATION_SCAN_EXPERIMENT = [dict(slotPhase(name, CONTROL_SLOTS), es=True, lr=lr, sigma=0.05, pool=REP5_BEST_POOL, games=g, population=n, gens=100000,
                                   hours=1.0, repeat=5, smooth=1 - 0.5 ** 5, snap=(250, 500, 1000, 2000))
                              for name, n, g, lr in (("continueRepeat5_200children_200games", 200, 200, 0.5), ("continueRepeat5_800children_50games", 800, 50, 0.5),
                                                     ("continueRepeat5_1600children_25games", 1600, 25, 0.5), ("continueRepeat5_3200children_12games", 3200, 12, 0.5),
                                                     ("continueRepeat5_800children_50games_lr025", 800, 50, 0.25))]

# Long run (2026-10-25 evening, PC): one net for up to 24 h. Start = final model of the 800 x 50 arm of the population scan (36.1k start 0 / 32.2k hard, the best arm),
# the proven recipe (800 children x 50 games, repeat 5, start 0 only) with sigma falling 0.05 -> 0.03 and lr 0.5 -> 0.2 over the 24 h (geometric, by the clock).
# lr must stay near 0.5 at the start for 800 children (lr 1.0 overshot); sigma 0.02 pool runs went flat, so the end values stay moderate.
# Snapshots every few thousand generations (~800 gens/h): pick the best by benchmark afterwards, the end of a run can drift.
LONG_RUN_EXPERIMENT = [dict(slotPhase("longRepeat5_800children_50games_sigmaLrDecay", CONTROL_SLOTS), es=True, lr=0.5, lrEnd=0.2, sigma=0.05, sigmaEnd=0.03,
                            pool="models/continueRepeat5_800children_50games_2026-10-25_13-35-30_pool.pt", games=50, population=800, gens=1000000,
                            hours=24.0, repeat=5, smooth=1 - 0.5 ** 5, snap=(2000, 5000, 10000, 15000))]

# Next server screening batch (2026-10-26): continue the 30.0k repeat-5 net (200 x 200, repeat 5, sigma 0.05, lr 0.5 as in sigmaContinue) and change ONE thing per arm:
# a smaller fixed lr (0.25, 0.15), half the games from hard starts (20-30k), half the games with 2x monsters. FIXED GENERATIONS (350 = what the sigma 0.05 control arms
# reached in 1.25 h: 349 / 347), not fixed time, so CPU contention on the shared server cannot change how much training an arm gets (the 'hours' cap is only a safety limit).
# The control is the sigma 0.05 pair of sigmaContinue (same machine, same start, same settings). Two replicates (A, B), same order in both rounds.
CONTINUE_BASE = dict(slotPhase("base", CONTROL_SLOTS), es=True, lr=0.5, sigma=0.05, pool=REP5_BEST_POOL, games=200, population=200, gens=350, hours=3.0,
                     repeat=5, smooth=1 - 0.5 ** 5, snap=(100, 200))
SCREEN_ARMS = (("lr025", dict(lr=0.25)), ("lr015", dict(lr=0.15)),
               ("hardStarts", dict(zero=0.5, minStart=20000, maxStart=30000)), ("monsters", dict(monster=0.5, monsterMult=2.0)))
SCREEN_EXPERIMENT = [dict(CONTINUE_BASE, name=f"continueRepeat5_{arm}_{r}", **changes) for r in "AB" for arm, changes in SCREEN_ARMS]

# Long run part 2 (2026-10-26 evening, PC, NEW fast code: FastBatch.run + grouped evaluation, +55% generations per hour at 800 x 50 on the server, bit-identical):
# continues the final net of the 24 h run (stopped by hand after 8,792 generations / 20.9 h, generation mean ~80k, schedule had reached sigma ~0.032 / lr ~0.225).
# Same recipe (800 x 50, repeat 5, start 0 only); sigma 0.03 -> 0.025 and lr 0.2 -> 0.12 over 18 h (both showed a wide plateau, so this is a gentle continuation of the
# decay). Progress of the schedules counts the process clock (perf_counter): if the PC sleeps meanwhile the run lasts longer in wall time.
LONG_RUN_PART1_POOL = "models/longRepeat5_800children_50games_sigmaLrDecay_2026-10-25_21-32-23_pool.pt"
LONG_RUN2_EXPERIMENT = [dict(slotPhase("longRepeat5_800children_50games_continued", CONTROL_SLOTS), es=True, lr=0.2, lrEnd=0.12, sigma=0.03, sigmaEnd=0.025,
                             pool=LONG_RUN_PART1_POOL, games=50, population=800, gens=1000000, hours=18.0, repeat=5, smooth=1 - 0.5 ** 5,
                             snap=(1000, 2500, 5000, 7500, 10000))]

# longRun2 was cut by an accidental PC shutdown after 502 generations (73 min, 2026-10-26 21:56): resume from its last checkpoint pool with the same
# recipe (the schedules had decayed only ~6%, so they restart at the same values) for the remaining ~16.8 h.
LONG_RUN2_RESUME_POOL = "models/longRepeat5_800children_50games_continued_2026-10-26_20-36-32_pool.pt"
LONG_RUN2B_EXPERIMENT = [dict(LONG_RUN2_EXPERIMENT[0], name="longRepeat5_800children_50games_continued_resumed", pool=LONG_RUN2_RESUME_POOL, hours=16.8)]

# Behaviour fine-tuning (2026-10-27): continue the final net of longRun2b with a fitness that punishes shooting at nothing and wiggling (proSprudler.behaviourPenalty).
# Scale: the net scores ~3 points per frame, a 110k game has ~7k decisions, ~3.6k bullets (73% with no monster on screen) and a summed steer change of ~2k;
# a kill is worth 100. Penalties (shotFree, shotAll, steer) in points: light (2, 0.2, 2) ~ 5% of a game, medium (5, 0.5, 6) ~ 15%, strong (10, 1, 15) ~ 30% of the score
# at the current behaviour. Same recipe as longRun2 (800 x 50, repeat 5, lr 0.2, sigma 0.03, fixed), 1 h per arm; the control has no penalty.
# Judge by the benchmark (start 0 / hard within noise of 110.7k / 101.0k) and by the behaviour counters, not by the CSV (penalised fitness for the arms).
BEHAVIOUR_POOL = "models/longRepeat5_800children_50games_continued_resumed_2026-10-26_22-01-30_pool.pt"
BEHAVIOUR_ARMS = (("control", (0.0, 0.0, 0.0)), ("light", (2.0, 0.2, 2.0)), ("medium", (5.0, 0.5, 6.0)), ("strong", (10.0, 1.0, 15.0)))
BEHAVIOUR_EXPERIMENT = [dict(slotPhase(f"behaviour_{arm}", CONTROL_SLOTS), es=True, lr=0.2, sigma=0.03, penalties=pen, pool=BEHAVIOUR_POOL,
                             games=50, population=800, gens=1000000, hours=1.0, repeat=5, smooth=1 - 0.5 ** 5, snap=(100, 250))
                        for arm, pen in BEHAVIOUR_ARMS]

# Steering-only penalty under the new bullet rule (2026-10-27): the nets learned under the old rule (control 124.3k -> 103.4k start 0 with the new one), so every
# arm continues the control pool of the behaviour experiment. The shot penalties did nothing (saturated shoot output), so only the steer penalty is varied:
# control (0), 0.5, 1, 2 (the light arm used 2 together with shot terms and lost a third of its score). 1 h each, same recipe as BEHAVIOUR_EXPERIMENT.
# Judge by the benchmark (against the control arm) and behaviour.py (steer change per decision).
STEER_POOL = "models/behaviour_control_2026-10-27_08-22-28_pool.pt"
STEER_ARMS = (("control", 0.0), ("steer05", 0.5), ("steer1", 1.0), ("steer2", 2.0))
STEER_EXPERIMENT = [dict(slotPhase(f"newRule_{arm}", CONTROL_SLOTS), es=True, lr=0.2, sigma=0.03, penalties=(0.0, 0.0, steer), pool=STEER_POOL,
                         games=50, population=800, gens=1000000, hours=1.0, repeat=5, smooth=1 - 0.5 ** 5, snap=(100, 250))
                    for arm, steer in STEER_ARMS]

# Shoot-head reset (2026-10-27 evening): the control net's shoot output is saturated (logit ~ +25 in every situation), so ES noise never flips a shot and a shot
# penalty gets no signal. resetShootHead.py zeroes only that output's weights / bias (steering and everything else stays), the shoot logit starts near 0.
# Reset net: 10.8k points (30 games, start 0; was 132.7k), shoots in 37% of the frames instead of 100%.
# Goal: aimed shots, not spray and pray. The control net kills ~250 monsters per game (150k-point games) with ~4,700 bullets (accuracy ~5%), so the penalty counts the bullets that hit
# NOTHING (shotMiss, points per missed bullet; a kill pays +100, so a bullet is worth firing when it hits with a probability above miss / (100 + miss)).
# Six arms, 2.2 h each (~13 h in total, the night plus the morning), same recipe as the other screens (800 x 50, repeat 5, lr 0.2, sigma 0.03), penalties (shotFree, shotAll, steer, shotMiss):
# no penalty (how fast does it relearn?), miss 3 / 10 / 30 / 100, and miss 10 with 5 per bullet fired while no monster is on screen.
HEAD_RESET_POOL = "models/newRule_control_headReset_pool.pt"
HEAD_RESET_ARMS = (("noPenalty", (0.0, 0.0, 0.0, 0.0)), ("miss10", (0.0, 0.0, 0.0, 10.0)), ("miss30", (0.0, 0.0, 0.0, 30.0)),
                   ("miss3", (0.0, 0.0, 0.0, 3.0)), ("miss100", (0.0, 0.0, 0.0, 100.0)), ("free5_miss10", (5.0, 0.0, 0.0, 10.0)))
HEAD_RESET_EXPERIMENT = [dict(slotPhase(f"headReset_{arm}", CONTROL_SLOTS), es=True, lr=0.2, sigma=0.03, penalties=pen, pool=HEAD_RESET_POOL,
                              games=50, population=800, gens=1000000, hours=2.2, repeat=5, smooth=1 - 0.5 ** 5, snap=(250, 500, 1000))
                         for arm, pen in HEAD_RESET_ARMS]

# Dead zone 0.2 + detour penalty (2026-10-28): a steer output within 0.2 of 0.5 now means "do not move" (env.STEER_DEAD_ZONE), so the net can rest instead of
# swinging out and back. The detour penalty charges the sideways distance travelled beyond the net displacement of every jump (px of back and forth).
# Calibration (miss-10 net under the dead zone): ~2,500 px of detour per 1,000 frames, the score is ~3 points per frame, so weight w costs about 0.9 w of the score
# at today's behaviour. Four arms from the miss-10 net (aimed shooting, accuracy 44%), all with miss 10: detour 0 (adapts to the dead zone, the control),
# 0.03, 0.1, 0.3. 2.2 h each, same recipe as the other screens.
DEAD_ZONE_POOL = "models/headReset_miss10_2026-10-27_23-08-55_pool.pt"
DEAD_ZONE_ARMS = (("detour0", 0.0), ("detour01", 0.1), ("detour003", 0.03), ("detour03", 0.3))
DEAD_ZONE_EXPERIMENT = [dict(slotPhase(f"deadZone_{arm}", CONTROL_SLOTS), es=True, lr=0.2, sigma=0.03, penalties=(0.0, 0.0, 0.0, 10.0, w), pool=DEAD_ZONE_POOL,
                             games=50, population=800, gens=1000000, hours=2.2, repeat=5, smooth=1 - 0.5 ** 5, snap=(250, 500, 1000))
                        for arm, w in DEAD_ZONE_ARMS]

GAMES_FEWER_EXPERIMENT =[dict(GAMES_PER_CHILD_EXPERIMENT[0], name=f"continueBest_{g}games", games=g) for g in (100, 50)]

# Sigma on a strong net (2026-10-25, server, 12 workers): every sigma test so far was 1-1.5 h from scratch (0.05 best at repeat 5), but the best step size
# probably shrinks as the net gets good. Continue the best 4 h repeat-5 net (30.0k start 0) with the proven recipe (200 x 200, lr 0.5, repeat 5) and only
# change sigma: 0.05 = control, 0.03, 0.07, 0.02, 0.1. Two full rounds (A, B), 1.25 h each, same order in both so round A is a complete scan if the run is cut
# short; the control appears twice to measure run-to-run noise from the same start net.
SIGMA_CONTINUE_EXPERIMENT = [dict(slotPhase(f"continueRepeat5_sigma{str(sg).replace('.', '')}_{r}", CONTROL_SLOTS), es=True, lr=0.5, sigma=sg,
                                  pool=REP5_BEST_POOL, games=200, population=200, gens=100000, hours=1.25, repeat=5, smooth=1 - 0.5 ** 5,
                                  snap=(100, 200, 400, 600))
                             for r in "AB" for sg in (0.05, 0.03, 0.07, 0.02, 0.1)]

# ES sigma from scratch, round 2 (server, 2026-10-23): extends ES_SIGMA (0.02 / 0.05 / 0.1; 0.1 best, 8.0-8.2k) upwards. Same recipe, 1.5 h, 2 replicates.
ES_SIGMA2_EXPERIMENT = [dict(slotPhase(f"sig{str(sg).replace('.', '')}{r}", CONTROL_SLOTS), es=True, lr=0.5, sigma=sg,
                             gens=100000, hours=1.5, snap=(100, 250, 500))
                        for r in "AB" for sg in (0.15, 0.2)]

# 'all': the cheap feature screening first (~1.5 h; may change what is worth polishing), then the three polish branches (3 h each)
EXPERIMENTS = {"deadZone": DEAD_ZONE_EXPERIMENT, "headReset": HEAD_RESET_EXPERIMENT, "steerScreen": STEER_EXPERIMENT, "behaviour": BEHAVIOUR_EXPERIMENT, "longRun2b": LONG_RUN2B_EXPERIMENT, "longRun2": LONG_RUN2_EXPERIMENT, "screenLrStartsMonsters": SCREEN_EXPERIMENT, "longRun": LONG_RUN_EXPERIMENT, "sigmaContinue": SIGMA_CONTINUE_EXPERIMENT, "populationScan": POPULATION_SCAN_EXPERIMENT, "moreChildren": CHILDREN_EXPERIMENT, "gamesFewer": GAMES_FEWER_EXPERIMENT, "gamesPerChild": GAMES_PER_CHILD_EXPERIMENT, "repLong": REPEAT_LONG_EXPERIMENT, "esSigma2": ES_SIGMA2_EXPERIMENT, "sigmaRep": SIGMA_REPEAT_EXPERIMENT, "repeat2": REPEAT2_EXPERIMENT, "esLr": ES_LR_EXPERIMENT, "esSigma": ES_SIGMA_EXPERIMENT, "repeat": REPEAT_EXPERIMENT, "esLong2": ES_LONG2_EXPERIMENT, "esLong": ES_LONG_EXPERIMENT,"es": ES_EXPERIMENT, "scratchReplace": SCRATCH_REPLACE_EXPERIMENT, "replace": REPLACE_EXPERIMENT, }    # python nightRun.py --experiment features


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
    population = cfg["population"] if a.smoke else ph.get("population", cfg["population"])
    method = (f"ES, lr {ph['lr']}" if ph.get("es") else f"GA, mutRate {ph['mutRate']}, replace {ph.get('replace', 0.0)}") + f", repeat {ph.get('repeat', 1)}, smooth {ph.get('smooth', 1.0):g}"
    note(f"phase {ph['name']}: {ph['inputs']} inputs, hidden {ph['hidden']}, starts {ph['minStart']}-{ph['maxStart']}, sigma {ph['sigma']}, "
         f"{method}, population {population} on {cfg['workers']} workers, up to {gens} gens / {hours:.2f} h")
    try:
        games = cfg["games"] if a.smoke else ph.get("games", cfg["games"])
        common = dict(networkName=ph["name"], livePlot=False,
                      minStartHeight=ph["minStart"], maxStartHeight=ph["maxStart"], zeroFraction=ph.get("zero", 0.0),
                      steerSmoothing=ph.get("smooth", 1.0),
                      monsterFraction=ph.get("monster", 0.0), monsterMult=ph.get("monsterMult", 1.0),
                      fast=ph.get("fast", False), actionRepeat=ph.get("repeat", 1), maxHours=hours,
                      **startNetworkArgs(ph, results))
        if ph.get("es"):
            ps.trainingLoopES(cfg["workers"], games, gens, population, ph["mutRate"], ph["sigma"], cfg["elites"], ph["lr"], cfg["keepPart"],
                              snapshotAt=(5,) if a.smoke and ph.get("snap") else ph.get("snap", ()),
                              sigmaEnd=ph.get("sigmaEnd"), learningRateEnd=ph.get("lrEnd"), penalties=ph.get("penalties", (0.0, 0.0, 0.0)), **common)
        else:
            ps.sprudlerTrainingLoop(
                cfg["workers"], games, gens, population, ph["mutRate"], ph["sigma"], cfg["elites"], cfg["keepPart"],
                snapshotAt=(5,) if a.smoke and ph.get("snap") else ph.get("snap", ()), replaceRate=ph.get("replace", 0.0),
                stage1Games=cfg["stage1Games"], finalistFraction=cfg["finalistFraction"], **common)
        files = [f for f in glob.glob(f"models/{ph['name']}_*.pt") if isFinalModel(f)]
        results[ph["name"]] = max(files, key=os.path.getmtime)
        note(f"phase {ph['name']} finished -> {results[ph['name']]}")
    except BaseException:    # includes Ctrl-C inside a phase before its first generation ends
        note(f"phase {ph['name']} FAILED:\n{traceback.format_exc()}")


def benchmark(phases, results, startTime, smoke, workers):
    '''Benchmarks references, control models and all phase results; returns the text.
    evalModel applies ONE smoothing/repeat setting to all its models, so models are benchmarked in groups of equal (smoothing, repeat);
    every model at the action repeat it was trained with (references: 1).'''
    snapshots = sorted(f for name in results for f in glob.glob(f"models/{name}_*_g[0-9]*.pt") if os.path.getmtime(f) > startTime)
    groups = {}    # only this run's own phases and snapshots: old reference models are no longer benchmarked along
    for f in snapshots:    # a snapshot belongs to the slot layout of its own phase (otherwise it is evaluated with the wrong observation)
        owner = next((ph for ph in phases if os.path.basename(f).startswith(ph["name"] + "_")), None)
        key = (owner.get("smooth", 1.0), owner.get("repeat", 1)) if owner else (0.5, 1)
        groups.setdefault(key, []).append(f)
    for ph in phases:
        if ph["name"] in results:
            groups.setdefault((ph.get("smooth", 1.0), ph.get("repeat", 1)), []).append(results[ph["name"]])
    text = ""
    for (smooth, repeat), models in groups.items():
        cmd = [sys.executable, "evalModel.py"] + models + ["--smoothing", str(smooth), "--monster-column", "2", "--fast", "--repeat", str(repeat), "--ranges"] + RANGES
        cmd += ["--games", "50", "--workers", "4"] if smoke else ["--workers", str(workers)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text += f"\n=== smoothing {smooth:g}, repeat {repeat} ===" + out.stdout + (("\n" + out.stderr[-2000:]) if out.returncode else "")
    return text


def main():
    ap = argparse.ArgumentParser(description="Unattended multi-phase training run")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every phase's hours and generations (default 1)")
    ap.add_argument("--experiment", default="repeat", choices=sorted(EXPERIMENTS), help="which list of phases (see EXPERIMENTS in section 2; default repeat)")
    ap.add_argument("--phases", help="comma-separated names: run only these phases (the benchmark goes to night_benchmark_<names>.txt)")
    ap.add_argument("--no-plot", action="store_true", help="do not open the live learning-curve window")
    ap.add_argument("--smoke", action="store_true", help="tiny test version")
    ap.add_argument("--workers", type=int, help="worker processes (default SETTINGS: 31; use hardware threads - 1)")
    a = ap.parse_args()

    phases = [ph for ph in EXPERIMENTS[a.experiment] if not a.phases or ph["name"] in a.phases.split(",")]
    cfg = {key: values[1 if a.smoke else 0] for key, values in SETTINGS.items()}
    if a.workers and not a.smoke:
        cfg["workers"] = a.workers
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
        text = benchmark(phases, results, startTime, a.smoke, cfg["workers"])
        fileName = "night_benchmark_smoke.txt" if a.smoke else (f"night_benchmark_{a.experiment}_{a.phases.replace(',', '_')}.txt" if a.phases else f"night_benchmark_{a.experiment}.txt")    # a smoke test must never overwrite real results, one experiment never another's
        with open(os.path.join("models", fileName), "w") as f:
            f.write(text)
        note("benchmark written\n" + text)
    note("run done")


if __name__ == "__main__":
    main()
