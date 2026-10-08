# CLAUDE.md — tensorNetwork

Learning project: the user (5th-semester CS student) builds small AIs from first principles and enjoys it. Started as a tensor/GPU port of a hand-made neural net trained with a **genetic algorithm** (no backprop, on purpose); now mainly "Sprudler": GA-trained nets that play a headless Doodle-Jump clone (`../sprudelJump`), goal ~100k points per game.

## Binding rules (full text: `learningContract.md` — read it at session start)
1. **The user writes the code.** For their project code, don't hand over copy-paste fixes. Explain the concept (analogy from an unrelated domain), say *that* something is a bug and *why*, let them write the line.
2. **Let them explain first**, then confirm what is right and nudge only the gap. No extra caveats.
3. **Fine to do directly:** explain language/library rules, give exact syntax for unfamiliar APIs upfront, worked examples when asked outside "fix my code", small requested mechanical edits and config changes.
4. **Short answers**, small chunks, split across messages.
5. **Ask before any big edit** (new module, rewrite, multi-file change, even tooling). Say what you'd build and why; offer: user writes it / skeleton / delegate. After a delegated big edit, offer a walkthrough in small chunks. One past permission is not standing permission.
6. The user runs and tunes training themselves (edits `__main__` of `proSprudler.py`, starts runs); you check logs, benchmark, advise.
7. **Always use the Vibe Wise plugin for coding** (installed 2026-10-08, `vibe-wise@anthropic-plugin-directory`): it exists so the user does the thinking and actually learns. Whenever coding/debugging/design work comes up, use its skills/flow first and follow its guidance together with this contract. If it seems inactive, tell the user (a restart/reload may be needed) instead of silently skipping it.
8. New teaching-style feedback from the user -> update `learningContract.md`.

## Where the knowledge lives (read on demand, don't load all of it)
- `AGENTS.md` (~100 KB): detailed project memory — architecture, environment quirks, diagnosed pitfalls, and a dated experiment log (sections 2f–2x for Sprudler). `grep` it for a topic instead of reading it whole. Keep it current when something meaningful is learned.
- `HANDOFF.md`: older handoff (2026-10-06), partly stale.
- Auto-memory (`~/.claude/projects/-home-jtebbert-projects-tensorNetwork/memory/`, shared with the VS Code plugin): roadmap, screening-then-long-run method, feedback rules.
- Key files: `network.py` (Network, batched forward), `training.py` (GA engine), `inCircleNN.py` (toy problem), `proSprudler.py` (Sprudler training + `evaluateNetworkFast`), `nightRun.py` (experiment phases), `evalModel.py` (benchmark), `plotRun.py` (learning curves), `stressCheck.py`, `monsterDeaths.py`; env in `../sprudelJump` (`env.py`, numba `fastEnv.py`, treat as black box).

## Hard-won lessons (short version)
- **Observation design beats hyperparameters.** Biggest wins: platforms below the player visible; stable slot layout (3 below, 2 above, nearest-first); screen-wrap-aware x. Landing-prediction features help in-distribution.
- **Trust only the benchmark** (`evalModel.py`, fixed 200 games, columns start 0 / hard 20-30k / mixed / half+half / monsters x2). Training means and "best ever" mislead (winner's curse). Noise: ~±5% start 0, ~±13% hard; from-scratch single runs can vary a lot, use replicates.
- **Experiments are screening rounds** (short, weight-scaled, 2 replicates vs a same-recipe control); only a clear winner gets a long run.
- Current bests: polishBase (start 0 ~8.4k), polishB (hard ~4.1k, 800 games/net); more games per network helps; slots/width/prev-action/difficulty inputs did not.
- Main death cause: jumping into a monster from below (30-50% of games).
- Exact comparisons need `OPENBLAS_NUM_THREADS=1`. Server/WSL specifics (cuda indices, CPU governor, venvs, sshd/tailscale) are in AGENTS.md "Environment".
- `pkill -f <script>` kills your own bash command if it contains the script name.
