# Handoff (2026-10-08) - read CLAUDE.md, learningContract.md first; AGENTS.md 2f-2x is the detailed log (grep it)

**Rules of engagement (binding):** learning project, the user writes the code themselves. *Ask before any big edit*, even tooling. Short answers, small chunks; let them explain first, validate/nudge. They are in charge of training (edit `__main__` of `proSprudler.py`, start runs); you check logs, benchmark, advise. Small requested fixes / config edits are fine. They offered to be walked through `evaluateNetworkFast` (proSprudler.py); `../sprudelJump/fastEnv.py` is a black box (FastBatch, observe, decide, step, done/scores).

**Goal:** ~100k points per game (needs ~770 consecutive landings, ~80% at max difficulty). Current best start-0 score ~8.4k, hard ~4.1k: still far away.

**Setup now:** Ryzen 9 7950X (31 workers, ~2.4 s/gen at 200 games), numba env (`fast=True`), remote via sshd + Tailscale (`training-pc`), run in tmux with `--no-plot`. Curve: `python plotRun.py --chain --save curve.png`.

**Models** (benchmark = `evalModel.py`, fixed 200 games; columns start 0 / hard 20-30k / mixed 0-30k / half-0+half-hard / hard monsters x2):
- Standard base (23 in / 2 out / [23]x3, stable slots 3/2/1,2, smoothing 0.5, from scratch): **`polishBase_2026-10-07_01-24-15.pt`** 8,354 / 3,564 / 4,602 / 5,771 / 2,687 = best start 0. Continuations polishBase2/3 gave nothing (plateau, start 0 even dropped).
- **`polishB_2026-10-08_03-35-09.pt`** (800 games/net, 50/50 mix, sigma 0.02 / mutRate 0.7): 7,441 / 4,074 / 4,090 / 5,805 / 2,779 = best hard. Beat polishA (200 games) clearly; polishC (70% start-0 mix) 7,849 / 3,508 -> best start 0 of that branch.
- Feature models (landing prediction: `--slots 3,2,1,2,0,1`, 33 inputs; both flags 38): from scratch better at start 0 (+18%) but worse on unseen hard starts; polished with hard starts + 800 games, `polishLand` was reported at ~8,512 / 3,409 / half+half 6,591 (from memory notes; not yet in `night_benchmark.txt` - verify).
- Needs flags: `--stable --smoothing 0.5` (+ `--slots ...` for feature layouts). Old `careful` (target mode) 7,485 / 3,106 is superseded.

**Night run in progress (started 2026-10-08 10:29, `python nightRun.py` = experiment `next`):** `polishD` (800 games + 70/30 mix; did not add up in 3 h), `polishLand`, `polishBoth` (stopped by Ctrl-C at ~19:00 after ~2 h), `polishE` (1,600 games; first start died with the Ctrl-C). Log shows a restart at 19:04 with `['polishBoth', 'polishE']` still running at 20:40 -> check `models/night_log.txt`, then read `night_benchmark.txt` when it finishes (columns for polishD/Land/Both/E not there yet).

**What worked / what did not:** observation fixes and noise reduction >> hyperparameters. Worked: stable slots (the whole effect of the older stable+target+smoothing bundle; target and smoothing add no score), careful mutations as polish, more games per network (800 > 200), common random numbers, rank selection, numba env. Did not help: more slots/width, previous-action input, difficulty input, monster-heavy practice, occupied flag alone, 800 games + 70/30 mix combined. Trust only the benchmark (noise ~+-5% start 0, ~+-13% hard); training means/records mislead.

**Roadmap** (also in auto-memory `project_improvement_roadmap`): (1) monster timing/reach inputs (monster deaths are 40-50% of games; the net steers toward the killer in >90% of those), (2) more games per network (polishE pending), (3) shrinking sigma schedule, (4) selection settings (keepPart 0.05), (5) Evolution Strategies, (6) platform width as input. Method: short weight-scaled screening runs with 2 replicates vs same-recipe controls, long run only for a clear winner.

**Commands**
- Run: `source venv/bin/activate; python proSprudler.py` or `python nightRun.py [--experiment polish|features|next] [--phases a,b] [--smoke] [--no-plot]`.
- Benchmark: `OPENBLAS_NUM_THREADS=1 python evalModel.py models/x.pt --stable --smoothing 0.5 --fast --monster-column 2 --ranges 20000:30000 0:30000 20000:30000:0.5`.
- Watch a model: `cd ../sprudelJump; python demo_render.py --model ../tensorNetwork/models/<x>.pt --stable --smoothing 0.5`.
- Other tools: `stressCheck.py` (silent-error check for the undervolt), `monsterDeaths.py`.

**Gotchas:** `pkill -f <script>` kills your own bash command if it contains the script name; the Bash tool blocks `sleep`; numba installed with `pip install --no-deps numba llvmlite`; the user's Ctrl-C in `nightRun` ends the current phase and moves on (second one interrupts the benchmark); `--smoke` now writes `night_benchmark_smoke.txt` (no longer overwrites the real one); both repos must be pulled on the server.

**Uncommitted at handoff:** lots of new `models/*` files, `AGENTS.md`, `nightRun.py`, `proSprudler.py`, `CLAUDE.md`, this file. Ask the user before committing/pushing.
