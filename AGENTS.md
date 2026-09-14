# AGENTS.md — tensorNetwork

Read this file (and `learningContract.md`, which it summarizes) at the start of every session on this repo. Keep both up to date — see "Keeping this file current" at the bottom.

## What this project is

A learning project, not a production one. The user is a 5th-semester CS student, learning neural networks from first principles. They previously built a working network using plain Python classes (`Neuron`, `Edge`, `Network`) and are now rebuilding it with PyTorch tensors, specifically to learn GPU-friendly, matricized/batched thinking instead of per-object loops. The current task is a toy binary classifier: given `(x, y)`, predict whether the point lies inside a circle (center `(2,1)`, radius `0.4`), trained with a **genetic algorithm** (not backprop/autograd — that's a deliberate choice, see Open items).

## Ground rules — read `learningContract.md` in full

`learningContract.md` is the authoritative, user-maintained contract for how an agent should teach in this project. Summary (do not treat this summary as a substitute for reading the actual file — it gets edited over time):

1. **Teach transfer, don't hand over code.** For the user's actual project code (`network.py`/`training.py`), don't give copy-paste-ready fixes. Point at the concept (often via an unrelated analogy — cars, recipes, etc.), let the user write the actual line.
2. **Let the user explain first, then validate/nudge.** Their learning method is explaining their own understanding back, often self-correcting mid-explanation. Confirm what's right, nudge only the specific gap — don't re-lecture the whole concept, don't pile on caveats they didn't ask for.
3. **What's fine to do directly (no analogy needed):** pointing out *that* something is a bug and *why*, conceptually; explaining language/library rules, APIs, and conventions directly (e.g. what `.squeeze()` does, what `torch.rand` returns); fully worked examples when explicitly requested outside a "fix my code" context; purely mechanical, non-learning-bearing edits (e.g. a find-and-replace refactor, once the user understands *why*) — use judgment, but default to teaching over doing.
4. **Answer length: short.** Small chunks, not essays. Split a long answer across multiple messages rather than one wall of text.
5. **Exceptions to rule 1 require explicit negotiation, not assumption.** When the user asks for a full rewrite of real project code (this has happened once — the initial tensor-port of `training.py`), don't just do it silently. Surface the tension against the contract and ask (e.g. via a clarifying question) whether they want a one-time exception, heavier scaffolding instead, or a standing contract change. One granted exception does not imply standing permission for future asks.

If the user gives new teaching-style feedback mid-conversation ("I don't want X", "do more of Y"), update `learningContract.md` to capture it — don't just apply it silently for the rest of the session and forget it next time.

## Architecture snapshot

**`network.py`**
- `Network` — `@dataclass`: `layers: list[tuple[weights, bias]]` (one tuple per layer, weight shape `[out_features, in_features]`), `loss`, `accuracy` (both `Optional[float]`, set externally after evaluation), `fitness` property (`1/(1+loss)`).
- `forwardPass(inputs)` — batched: `inputs` is `[batch_size, in_features]`, not a single sample. Loops over layers (`ReLU` for all hidden layers), final layer uses `torch.sigmoid` (needed for `crossEntropyLoss` to get a valid probability — the network originally used `ReLU` everywhere, which caused a real, diagnosed failure mode, see Known pitfalls). Matmul form is `outputPrev @ weights.T + bias` (transposed weights, because of the batch dimension — `weights` alone is `[out,in]`, not compatible with a `[batch,in]` input without transposing).
- `buildNetwork(inputSize, hiddenSizes, outputSize)` — builds layer list. Weight init: `kaiming_uniform_` (ReLU hidden layers), `xavier_uniform_` (sigmoid output layer). Biases: zero-init. All tensors created with `device=DEVICE`.
- `DEVICE` — module-level constant, currently `"cuda:1"` (see Environment/hardware below — index is *not* the same across every environment).

**`training.py`** — genetic algorithm training pipeline, fully batched (no per-sample or per-training-point Python loops; there *is* still a Python loop over the population of networks each generation — see Open items).
- `generateTrainingData(n)` → `(pointsTensor [n,2], targetsTensor [n])`.
- `crossEntropyLoss(target, prediction)`, `getAccuracy(predictions, target)` — both fully vectorized, both return plain Python `float`s (not tensors — `.item()` matters, was a recurring bug source).
- `evaluateNetwork(network, points, targets)` → `(loss, accuracy)`.
- `crossoverTensor`, `mutateTensor` — elementwise tensor ops (`torch.where`, `torch.rand_like`/`randn_like`) replacing what would otherwise be per-neuron/per-edge loops.
- `breed`, `buildPopulation`, `selection`, `pickRandomParents`, `trainingLoop` — standard GA loop: evaluate population, keep elites + top `keepPartSelection`, breed+mutate to refill population, repeat for `iterations` generations.
- `gpu` — module-level constant, same value/caveats as `network.py`'s `DEVICE` (kept as two separate constants, not shared, to avoid a circular import between the two files).

## Environment / hardware — non-obvious, easy to get wrong again

- **Local dev machine has no GPU.** Testing `forwardPass`/`buildNetwork` logic locally means bypassing `DEVICE` — construct a `Network` directly with plain CPU tensors rather than going through `buildNetwork`.
- **Server has two GPUs**: Tesla P40 and GTX 1660. `nvidia-smi`'s own GPU numbering does **not** match PyTorch's `cuda:N` indices in `venv-p40` — `nvidia-smi` labels the P40 as its GPU "0" (PCI-bus-ID order), but PyTorch enumerates fastest-first: **`cuda:0` = GTX 1660, `cuda:1` = Tesla P40**. Confirmed directly via `torch.cuda.get_device_name(0/1)` (not just inferred from `nvidia-smi`'s process table) — this is settled, not a guess. If it's ever in doubt again, re-run that exact check rather than trusting `nvidia-smi`'s display.
- **`venv-p40`** (Python 3.11, `uv`, `torch==2.4.1+cu121`) is the environment that actually supports the P40 — its Pascal architecture (`sm_61`) was dropped from newer official PyTorch builds, so this specific torch version is a hard requirement, not a preference. Shell aliases `tnvenv`/`tnp40` jump into the project and activate the right venv (`tnp40` = P40/CUDA env, `tnvenv` = the other one).
- **The GTX 1660 can be passed through to a Windows VM** (`vmstart`/`vmstop` aliases in `.zshrc`). If that VM is running, the 1660 (and whatever CUDA index it held) becomes invisible to the training process — device indices can shift as a result. Don't assume `cuda:1` is stable across sessions without checking.

## Known pitfalls already diagnosed once (don't rediscover from scratch)

- **`ReLU` on the output layer** produces unbounded, non-probability output — broke `crossEntropyLoss` numerically (needed a clamp hack) and, more importantly, caused genuinely flat fitness landscapes (near-zero weights → near-zero pre-activation → `ReLU` output stuck near 0 for everyone → no selection signal → GA just drifts). Fixed via `sigmoid` on the output layer + proper (Kaiming/Xavier) weight init instead of `torch.rand`. Both networks converging to *exactly* `-log(0.5)` or the "always-near-zero" loss value was the diagnostic tell — if loss values look suspiciously like a clean closed-form constant, suspect a degenerate constant-output failure mode, not "just needs more training."
- **Broadcasting silently does the wrong thing**, doesn't error. `[1000]` vs `[1000,1]` combined via `*` broadcasts into a `[1000,1000]` outer-product-like mess instead of pairing elementwise — this happened between `forwardPass`'s output (`[batch,1]`, from `outputSize=1`) and `target` (`[batch]`). Fixed with `.squeeze(1)`. Any time a "loss looks weirdly wrong but doesn't crash," check tensor shapes before logic.
- **Depth without width doesn't help.** Empirically tested: `hiddenSizes=[2,2]` performed identically to `hiddenSizes=[2]` (both ~75% accuracy) — a narrow bottleneck caps representational capacity regardless of depth stacked after it. `hiddenSizes=[5]` alone clearly beat both (~97-98%).

## Open items discussed but not yet done

- **No train/test split exists yet.** All accuracy/loss numbers so far are in-sample (evaluated on the same data trained on). Flagged as the highest-priority next step before further tuning — current numbers (~98% for `[5]`) haven't been checked for generalization.
- **Population loop is still a Python `for` loop** (one `evaluateNetwork` call per network per generation) — this is the current GPU-utilization bottleneck (~13% observed). Discussed but deliberately deferred: true population-level batching (stack all networks' weights into one 3D tensor, `torch.bmm`) is a real architecture change, not a quick fix. Smaller intermediate options discussed: cut unnecessary `.item()` sync points (currently up to 2× population-size forced syncs per generation), or CUDA streams to overlap population members without a full restructure.
- **Running multiple independent `trainingLoop` calls in parallel** (e.g. testing several `hiddenSizes` configs at once) — recommended approach is `multiprocessing`/`ProcessPoolExecutor` with the **`'spawn'` start method**, not the Linux default `'fork'` (forking after CUDA is initialized in the parent process crashes child processes). Not yet built. VRAM is a real constraint on the 1660 (6GB) for many concurrent processes — the P40 (23GB) is the better target for this.
- Untried architecture variants worth testing given the width>depth finding: `[5,5]`, `[8,4]`, etc.

## Keeping this file current

This file is a working memory aid, not a one-time snapshot. When you (the agent) reach a meaningful checkpoint in a session — a real bug diagnosed and fixed, a design decision made, an experiment result that changes what's believed about the model, a new environment quirk discovered — update the relevant section above before the session ends. Prefer editing existing sections over appending a growing changelog; this file should always reflect *current* understanding, not a history of how it got there. If `learningContract.md` changes, make sure the summary in "Ground rules" above still matches it.
