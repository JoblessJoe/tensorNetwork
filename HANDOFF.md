# Handoff (2026-10-06) - read AGENTS.md and learningContract.md first

**Rules of engagement (binding):** learning project, the user writes the code themselves. *Ask before any big edit*, even tooling (they were unhappy about the numba port being built unasked). Short answers, small chunks; let them explain first, validate/nudge. They are in charge of training: they edit `__main__` of `proSprudler.py` and start runs themselves (live plot window); you check logs, benchmark, advise. Small requested fixes / config edits are fine. The user explicitly offered to be walked through `evaluateNetworkFast` (proSprudler.py) - treat `../sprudelJump/fastEnv.py` as a black box (5 calls: FastBatch, observe, decide, step, done/scores).

**Goal:** a model scoring ~100k points per game (needs ~770 consecutive landings; ~80% of such a game is at max difficulty). Current hard-start score ~3k, so the goal is still very far; 10-20k would already be a stretch.

**State of the models** (benchmark = `evalModel.py`, fixed 200 games; columns start 0 / hard 20-30k / mixed 0-30k / half-at-0+half-hard / hard with 2x monsters):
- Best model: `models/careful_2026-10-05_18-43-51.pt` (+ `_pool.pt`): 7,485 / 3,106 / 4,502 / 4,968 / 2,188. Target-mode net (25 inputs, 7 outputs), needs `--stable --smoothing 0.5` everywhere.
- 4 from-scratch replicates (~344 gens, 15 min each on the fast env) land at ~6.4-7.1k start 0 and 2.2-2.8k hard (A: `tgtScratchA_*`, B/C/D: `tgtFastB/C/D_*`): the recipe is reliable. Best of them: C (`models/tgtFastC_2026-10-06_14-22-58.pt`).
- Older chain: `sprudler_2026-10-04_08-28-10.pt` (classic 23-input obs, no flags needed) 6,898 / 2,401 / 3,704 / 5,044.

**What worked / what did not** (details: AGENTS.md 2f-2n): observation fixes and noise reduction >> hyperparameters; stable platform slots + network-chosen committed target + steer smoothing 0.5 (the three were introduced together, which one matters is unknown); careful mutations (sigma 0.02, mutRate 0.7) as a polish step after an explore run (+29% on hard); numba env (identical results, 7-14x faster). Did not help: previous-action inputs, monster-heavy practice, difficulty input, env-chosen targets (user rejected as hard-coding). Training curves and 'best ever' records mislead (winner's curse, population converging) - trust the benchmark only.

**Open decision / suggested next step:** (1) ablations from scratch at 344 gens, 2 replicates each (without target: inputs=23/outputs=2; without smoothing; classic slots), each ~15 min = `nightRun.py` phases (config only) -> learn which change does the work; (2) continue the best base with the 50/50 start mix, explore then careful; (3) maybe 1,000 games per network now that evaluation is cheap (less noisy selection). Ideas not built: imitation learning from a lookahead planner (also a feasibility check for 100k), Evolution Strategies, recurrent memory, platform width as an input (the state has no width).

**Commands**
- Run: `source venv/bin/activate; python proSprudler.py` (settings in `__main__`: `fast=True`, `actionRepeat=1`, `stableSlots`, `steerSmoothing`, ...) or `python nightRun.py [--phases a,b] [--smoke] [--no-plot]`.
- Benchmark: `python evalModel.py models/x.pt --stable --smoothing 0.5 --fast --monster-column 2 --ranges 20000:30000 0:30000 20000:30000:0.5` (`OPENBLAS_NUM_THREADS=1` for bit-exact comparisons with old numbers).
- Watch a model: `cd ../sprudelJump; python demo_render.py --model ../tensorNetwork/models/careful_2026-10-05_18-43-51.pt --stable --smoothing 0.5`.

**Gotchas:** `pkill -f <script>` kills your own bash command if its text contains the script name; the Bash tool blocks `sleep`; numba was installed with `pip install --no-deps numba llvmlite` (keeps numpy); both repos (`tensorNetwork`, `../sprudelJump`) must be pulled on the server; a Ryzen 9 7950X is on its way (then use ~31 workers: `concurrent`, `workers` in nightRun, `--workers` in evalModel); the user's Ctrl-C in `nightRun` ends the current phase and moves on (a second one interrupts the benchmark); smoke runs (`--smoke`) overwrite `models/night_*.txt` (restore with `git checkout`).

**Uncommitted at handoff:** `nightRun.py` (gens=344 and the tgtFast phases), `AGENTS.md` (2n), `HANDOFF.md`, models `tgtFast*`, `tgtScratch*`, `careful2`-era files; ask the user before committing/pushing.
