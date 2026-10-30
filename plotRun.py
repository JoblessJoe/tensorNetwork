'''
Learning curve for a GA training run, read from the CSV that sprudlerTrainingLoop writes
(one line per generation, next to the saved model: models/<name>_<timestamp>.csv).
Runs in its own process, so it never slows the training down.

    python plotRun.py                      newest CSV in models/, one window
    python plotRun.py --chain              join all runs of this model's lineage (follows 'startFrom' links)
    python plotRun.py a.csv b.csv          join several runs by hand, oldest first
    python plotRun.py --live               same, redrawn every 5 s while the run is going
    python plotRun.py --live --running --chain   one window with a panel for EVERY run that is going right now (2 runs in parallel = 2 panels side by side)
    python plotRun.py models/x.csv --save curve.png
    python plotRun.py --last 200           zoom into the last 200 generations
    python plotRun.py --full               x axis over the planned run length (default: follows the data)
    python plotRun.py --log                logarithmic y axis (slow starts, huge late scores)
    python plotRun.py --light              light theme (default: dark)

The y axis rescales itself on every redraw, so it zooms out as the scores grow.
'''
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# validated categorical slots 1-3 (blue, orange, aqua), stepped per theme
THEMES = {
    "dark": dict(surface="#1a1a19", ink="#f2f1ec", ink2="#c3c2b7", muted="#8a8980", grid="#2e2d2a",
                 blue="#3987e5", orange="#d95926", aqua="#199e70"),
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8980", grid="#e8e7e2",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a"),
}


def loadRun(path):
    '''Returns (meta dict, columns dict of float arrays). Tolerates a half-written last line (live mode).'''
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                meta = dict(kv.split("=", 1) for kv in line[1:].split() if "=" in kv)
            elif line and not line[0].isalpha():
                parts = line.split(",")
                if len(parts) == 8:
                    rows.append([float(x) for x in parts])
    names = ["gen", "elapsed_s", "mean", "median", "p10", "p90", "best", "chosen"]
    data = np.array(rows) if rows else np.empty((0, 8))
    return meta, {n: data[:, k] for k, n in enumerate(names)}


_warned = set()


def loadMeta(path):
    if not os.path.exists(path):          # live mode starts before the first generation has been written
        return {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                return dict(kv.split("=", 1) for kv in line[1:].split() if "=" in kv)
    return {}


def resolveChain(path):
    '''Follows the 'startFrom' links in the CSV headers back to the first run. Returns CSV paths, oldest first.'''
    base = os.path.dirname(os.path.abspath(__file__))
    chain, seen = [path], {os.path.abspath(path)}
    while True:
        src = loadMeta(chain[0]).get("startFrom", "None")
        if src in ("None", ""):
            break
        csv = src[:-8] + ".csv" if src.endswith("_pool.pt") else src[:-3] + ".csv" if src.endswith(".pt") else src
        csv = csv if os.path.isabs(csv) else os.path.join(base, csv)
        if not os.path.exists(csv):
            if csv not in _warned:
                _warned.add(csv)
                print(f"chain: no CSV for {src} (renamed, or trained before logging existed) - chain starts after it")
            break
        if os.path.abspath(csv) in seen:
            break
        seen.add(os.path.abspath(csv))
        chain.insert(0, csv)
    return chain


def loadRuns(paths):
    '''Joins several run CSVs (oldest first) into one long run: generation and time continue where the previous run ended.
    Returns (meta of the last run, columns incl. per-generation seconds 'dt', [(first generation, label) of every run after the first], planned total generations).'''
    cols = ["gen", "elapsed_s", "mean", "median", "p10", "p90", "best", "chosen", "dt"]
    parts = {c: [] for c in cols}
    bounds, genOff, timeOff, meta, total = [], 0, 0.0, {}, 0
    for k, path in enumerate(paths):
        m, d = loadRun(path)
        if len(d["gen"]) == 0:
            continue
        d["dt"] = np.diff(d["elapsed_s"], prepend=0.0)
        if parts["gen"]:
            hi, lo = m.get("maxStartHeight", "None"), int(m.get("minStartHeight", 0))
            bounds.append((genOff + 1, f"run {len(bounds) + 2}: sigma {m.get('sigma', '?')}, " + (f"starts {lo // 1000}-{int(hi) // 1000}k" + (f" + {int(float(m.get('zeroFraction', 0)) * 100)}% at 0" if float(m.get("zeroFraction", 0)) else "") if hi != "None" else "normal starts")))
        for c in cols:
            parts[c].append(d[c] + (genOff if c == "gen" else timeOff if c == "elapsed_s" else 0))
        genOff += int(d["gen"][-1])
        timeOff += d["elapsed_s"][-1]
        meta = m
        total = genOff - int(d["gen"][-1]) + int(m.get("generations", 0))
    merged = {c: (np.concatenate(v) if v else np.empty(0)) for c, v in parts.items()}
    return meta, merged, bounds, total


def rolling(x, window):
    '''Centered-ish trailing moving average that is defined from the first point on.'''
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


_shapeCache = {}


def networkInfo(csvPath, meta):
    '''(layer sizes [inputs, hidden..., outputs], number of weights + biases) of the run's network: from the CSV header (newer runs) or, for older
    runs, from the model file saved next to the CSV (same name, .pt). None if neither is available yet.'''
    if "shape" in meta:
        sizes = [int(n) for n in meta["shape"].split("-")]
        return sizes, sum(a * b + b for a, b in zip(sizes, sizes[1:]))
    pt = csvPath[:-4] + ".pt"
    if pt not in _shapeCache and os.path.exists(pt):
        try:
            import torch
            layers = torch.load(pt, map_location="cpu", weights_only=True)["layers"]
            sizes = [int(layers[0][0].shape[1])] + [int(w.shape[0]) for w, _ in layers]
            _shapeCache[pt] = (sizes, sum(int(w.numel() + b.numel()) for w, b in layers))
        except Exception:
            return None
    return _shapeCache.get(pt)


def describeSlots(text):
    '''Human-readable observation layout from the CSV value of stableSlots.'''
    if text is None:
        return "slots 3/2/1/2 + landing prediction (fixed layout)"    # runs since the layout became fixed no longer log stableSlots
    if text == "False":
        return "classic observation (5 platforms + 3 monsters sorted by distance)"
    if text == "True":
        return "slots 3/2/1/2 (platforms below/above, monsters below/above)"
    nums = [int(n) for n in text.strip("()").split(",") if n.strip()]
    out = "slots " + "/".join(str(n) for n in nums[:4])
    if len(nums) > 4 and nums[4]:
        out += " + occupied flag"
    if len(nums) > 5 and nums[5]:
        out += " + landing prediction"
    return out


def fmtTime(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if h else f"{m}m {seconds % 60:02d}s"


def spread(positions, lo, hi, minGap):
    '''Nudges label y positions apart (sorted order kept) so end labels never overlap.'''
    order = np.argsort(positions)
    out = np.array(positions, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < minGap:
            out[b] = out[a] + minGap
    if out[order[-1]] > hi:                       # pushed off the top: shift the whole stack down
        out -= out[order[-1]] - hi
    return out


def runInfo(csvPath, meta):
    '''(time cap in hours or None, device 'GPU' / 'CPU' or None) of a run. New runs write both into the CSV header (maxHours, device). Older runs: the cap from
    the phase line in models/night_log.txt ('up to N gens / H h', the last one of that name), the device from an optional sidecar file '<csv>.device'.'''
    cap, device = None, meta.get("device")
    if meta.get("maxHours") not in (None, "None", ""):
        cap = float(meta["maxHours"])
    elif meta.get("name"):
        log = os.path.join(os.path.dirname(os.path.abspath(csvPath)), "night_log.txt")
        try:
            with open(log, errors="replace") as f:
                for line in f:
                    m = re.search(r"phase (\S+): .* up to \d+ gens / ([\d.]+) h", line)
                    if m and m.group(1) == meta["name"]:
                        cap = float(m.group(2))
        except OSError:
            pass
    if device is None and os.path.exists(csvPath + ".device"):
        with open(csvPath + ".device") as f:
            device = f.read().strip()
    return cap, (device.upper() if device else None)


def draw(fig, paths, theme, last=None, log=False, full=False, chain=False):
    T = THEMES[theme]
    if chain and len(paths) == 1:
        paths = resolveChain(paths[0])          # no-op while the CSV does not exist yet
    if os.path.exists(paths[-1]):
        meta, d, bounds, totalGens = loadRuns(paths)
    else:
        meta, d, bounds, totalGens = {}, {"gen": np.empty(0)}, [], 0
    fig.clear()
    fig.patch.set_facecolor(T["surface"])
    wIn = fig.bbox.width / fig.dpi
    k = min(1.0, wIn / 12.5)                 # panels are narrower than a single window: shrink the texts with them
    left = max(0.075, 1.0 / wIn)             # room for the y tick labels and the axis label
    if len(d["gen"]) == 0:
        fig.text(0.5, 0.5, "waiting for the first generation...", color=T["ink2"], ha="center", va="center", fontsize=14)
        return None

    total = totalGens or None
    gen = d["gen"]
    cap, device = runInfo(paths[-1], meta)
    ownElapsed = loadRun(paths[-1])[1]["elapsed_s"][-1]          # clock of the newest run alone (the time cap counts that, not the chain)
    perGen = rolling(d["dt"], 20)[-1]
    live = time.time() - os.path.getmtime(paths[-1]) < max(90.0, 8 * perGen)    # the CSV is still being written
    timeLeft = max(0.0, cap * 3600 - ownElapsed) if cap else None
    sel = slice(-last, None) if last else slice(None)
    bestEver = np.maximum.accumulate(d["best"])
    window = max(5, len(gen) // 12)
    smooth = rolling(d["mean"], window)

    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1], hspace=0.08, left=left, right=0.82, top=0.725, bottom=0.09)
    ax = fig.add_subplot(gs[0])
    axT = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, axT):
        a.set_facecolor(T["surface"])
        a.grid(True, color=T["grid"], linewidth=1, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(T["grid"])
        a.tick_params(which="both", colors=T["ink2"], length=0, labelsize=10)
    ax.tick_params(labelbottom=False)

    # population: 10th-90th percentile band + raw mean (faint) + smoothed mean (the trend)
    ax.fill_between(gen, d["p10"], d["p90"], color=T["blue"], alpha=0.18, linewidth=0, zorder=2)
    ax.plot(gen, d["mean"], color=T["blue"], alpha=0.35, linewidth=1, zorder=3)
    ax.plot(gen, smooth, color=T["blue"], linewidth=2, solid_capstyle="round", zorder=5)
    ax.plot(gen, d["best"], color=T["orange"], alpha=0.75, linewidth=1, zorder=4)
    ax.step(gen, bestEver, where="post", color=T["aqua"], linewidth=2, zorder=6)
    # dots where the all-time best was beaten
    rec = np.flatnonzero(np.diff(bestEver, prepend=-np.inf) > 0)
    ax.scatter(gen[rec], bestEver[rec], s=34, color=T["aqua"], edgecolor=T["surface"], linewidth=2, zorder=7)

    for g, label in bounds:
        ax.axvline(g - 0.5, color=T["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(label, xy=(g - 0.5, 0), xycoords=("data", "axes fraction"), xytext=(4, 6), textcoords="offset points",
                    color=T["muted"], fontsize=8.5, va="bottom", ha="left", rotation=90, annotation_clip=False)
    # y scale: zooms to what is visible; linear starts at 0 unless zoomed into a late window
    vis = lambda k: d[k][sel]
    top = max(vis("best").max(), bestEver[sel].max())
    if log:
        ax.set_yscale("log")
        low = max(1.0, min(np.min(vis("p10")[vis("p10") > 0], initial=top), top) * 0.8)
        ax.set_ylim(low, top * 1.35)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))   # plain numbers on minor ticks too
    else:
        low = 0 if not last else max(0, vis("p10").min() * 0.85)
        ax.set_ylim(low, top * 1.10 if top > 0 else 1)
    x0 = gen[sel][0]
    xr = total if (full and total and not last) else gen[-1]   # follows the data; --full shows the planned run length
    if full and cap and live and not last and timeLeft:
        xr = gen[-1] + timeLeft / max(perGen, 1e-9)                 # time-capped: where the cap will be reached at the current speed
    ax.set_xlim(x0 - 0.5, max(xr, x0 + 1) + 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("score per game (points)", color=T["ink2"], fontsize=10)
    unit, perUnit = ("minutes", 60.0) if d["elapsed_s"][-1] < 5400 else ("hours", 3600.0)
    hours = d["elapsed_s"] / perUnit
    secAx = ax.secondary_xaxis("top", functions=(lambda g: np.interp(g, gen, hours), lambda h: np.interp(h, hours, gen)))
    secAx.set_xlabel(f"elapsed {unit}", color=T["muted"], fontsize=9 * k)
    secAx.tick_params(colors=T["ink2"], length=0, labelsize=9 * k)
    secAx.spines["top"].set_visible(False)

    # end labels (text stays in ink; the dot carries the series color)
    y0, y1 = ax.get_ylim()
    ends = [("best ever", bestEver[-1], T["aqua"]), ("best of generation", d["best"][-1], T["orange"]),
            ("population mean", smooth[-1], T["blue"])]
    if log:
        ypos = [np.log10(v) for _, v, _ in ends]
        gap = 0.085 * (np.log10(y1) - np.log10(y0)); lo_, hi_ = np.log10(y0), np.log10(y1)
        ypos = [10 ** v for v in spread(ypos, lo_, hi_, gap)]
    else:
        ypos = spread([v for _, v, _ in ends], y0, y1, 0.08 * (y1 - y0))
    for (label, v, color), yp in zip(ends, ypos):
        ax.annotate(f"{label}\n{v:,.0f}", xy=(gen[-1], v), xytext=(1.01, yp), textcoords=("axes fraction", "data"),
                    color=T["ink"], fontsize=9.5, va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=T["muted"], linewidth=0.8, shrinkA=0, shrinkB=4))
        ax.scatter([gen[-1]], [v], s=40, color=color, edgecolor=T["surface"], linewidth=2, zorder=8, clip_on=False)

    # legend (always present for >= 2 series)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color=T["blue"], lw=2), Patch(facecolor=T["blue"], alpha=0.25),
               Line2D([0], [0], color=T["orange"], lw=1, alpha=0.75), Line2D([0], [0], color=T["aqua"], lw=2)]
    ax.legend(handles, ["population mean (trend)", "population 10-90%", "best of generation", "best ever"],
              loc="upper left", frameon=True, facecolor=T["surface"], framealpha=0.92, edgecolor="none", fontsize=9, labelcolor=T["ink2"], ncol=2, handlelength=1.6).set_zorder(20)

    # seconds per generation
    dt = d["dt"]
    axT.bar(gen, dt, width=1.0, color=T["muted"], alpha=0.55, linewidth=0, zorder=2)
    axT.set_ylabel("sec / gen", color=T["ink2"], fontsize=10)
    axT.set_xlabel("generation", color=T["ink2"], fontsize=10)
    axT.set_ylim(0, max(dt[sel].max(), 1) * 1.15)

    # headline + facts
    elapsed = d["elapsed_s"][-1]
    done = int(gen[-1])
    if cap:      # time-capped run: the cap, not the generation count, ends it
        if timeLeft <= 0:
            eta = f"   |   time cap {cap:g} h reached"
        elif live:
            eta = f"   |   {fmtTime(timeLeft)} left of {cap:g} h, ends ~{time.strftime('%a %H:%M', time.localtime(time.time() + timeLeft))}"
        else:
            eta = f"   |   stopped at {fmtTime(ownElapsed)} of {cap:g} h"
    else:
        eta = f"   |   ETA {fmtTime((total - done) * perGen)}" if total and done < total else ("   |   finished" if total else "")
    prog = f"generation {done:,}" + (f" / {total:,}" if total and not cap else "")
    speed = f"{perGen:.2f}" if perGen < 1 else f"{perGen:.1f}"
    name = meta.get("name", os.path.basename(paths[-1]))
    fig.text(left, 0.965, name, color=T["ink"], fontsize=17 * k, fontweight="bold", va="top")
    if device:
        fig.text(0.82, 0.962, f" {device} ", color=T["ink"], fontsize=11 * k, fontweight="bold", va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc=T["aqua"] if device == "GPU" else T["blue"], ec="none", alpha=0.85))
    # time progress bar (time cap) - or generation progress for a run with a sensible planned length
    frac = min(1.0, ownElapsed / (cap * 3600)) if cap else (min(1.0, done / total) if total and total <= 100000 else None)
    if frac is not None:
        from matplotlib.patches import Rectangle
        bar = fig.add_axes([left, 0.917, 0.82 - left, 0.010])
        bar.set_xlim(0, 1); bar.set_ylim(0, 1); bar.axis("off")
        bar.add_patch(Rectangle((0, 0), 1, 1, color=T["grid"], linewidth=0))
        bar.add_patch(Rectangle((0, 0), frac, 1, color=T["blue"], linewidth=0))
    chained = f"   |   {len(bounds) + 1} runs chained" if bounds else ""
    fig.text(left, 0.903, f"{prog}   |   elapsed {fmtTime(elapsed)}{eta}   |   {speed} s/gen ({3600 / max(perGen, 1e-9):,.0f} gens/h){chained}", color=T["ink2"], fontsize=10.5 * k, va="top")

    sinceRecord = done - int(gen[rec[-1]]) if len(rec) else done
    facts = f"trend now {smooth[-1]:,.0f}"
    if len(gen) > 1:                                  # change over the last (up to) 1,000 generations needs a point to compare with
        recent = min(1000, len(gen) - 1)
        gain = smooth[-1] - smooth[-1 - recent]
        facts += f"   ({gain:+,.0f} / {recent} gens, {100 * gain / max(smooth[-1 - recent], 1e-9):+.0f}%)"
    facts += f"   |   best ever {bestEver[-1]:,.0f}, set {sinceRecord} gens ago   |   {len(rec)} records"
    fig.text(left, 0.866, facts, color=T["ink2"], fontsize=10.5 * k, va="top")
    cfg = (f"population {meta.get('population', '?')}  games/network {meta.get('games', '?')}  mutation {meta.get('mutationRate', '?')}"
           f" / sigma {meta.get('sigma', '?')}  elites {meta.get('elites', '?')}  keep {meta.get('keepPart', '?')}"
           f"  start heights {meta.get('minStartHeight', 0)}-{meta.get('maxStartHeight')}" if meta.get("maxStartHeight", "None") != "None" else
           f"population {meta.get('population', '?')}  games/network {meta.get('games', '?')}  mutation {meta.get('mutationRate', '?')}"
           f" / sigma {meta.get('sigma', '?')}  elites {meta.get('elites', '?')}  keep {meta.get('keepPart', '?')}  normal start")
    fig.text(left, 0.830, cfg, color=T["muted"], fontsize=9.5 * k, va="top")
    info = networkInfo(paths[-1], meta)
    parts = []
    if info:
        sizes, nWeights = info
        kind = "  (target mode)" if sizes[0] == 25 and sizes[-1] == 7 and meta.get("stableSlots") == "True" else ""
        parts.append(f"network {' \u2192 '.join(str(n) for n in sizes)}{kind}   {nWeights:,} weights")
    if meta.get("stableSlots") is not None:           # older runs: their observation layout (the current one is fixed)
        parts.append(describeSlots(meta.get("stableSlots")))
    if meta.get("steerSmoothing") not in (None, "1.0", "1"):
        parts.append(f"steer smoothing {meta['steerSmoothing']}")
    if meta.get("zeroFraction") not in (None, "0.0", "0"):
        parts.append(f"{100 * float(meta['zeroFraction']):.0f}% of the games start at 0")
    if meta.get("startFrom") not in (None, "None"):
        parts.append("continues " + os.path.basename(meta["startFrom"]).replace(".pt", "").replace("_pool", "")[:34])
    else:
        parts.append("from scratch")
    fig.text(left, 0.796, "   |   ".join(parts), color=T["muted"], fontsize=9.5 * k, va="top")

    return dict(ax=ax, gen=gen, d=d, smooth=smooth, bestEver=bestEver)


def addHover(fig, state, theme):
    '''Crosshair + readout while the mouse is over the plot (interactive windows only).'''
    T = THEMES[theme]
    ax = state["ax"]
    vline = ax.axvline(0, color=T["muted"], linewidth=1, visible=False, zorder=1)
    box = ax.text(0.5, 0.97, "", transform=ax.transAxes, color=T["ink"], fontsize=9.5, va="top", ha="center",
                  bbox=dict(boxstyle="round,pad=0.4", fc=T["surface"], ec=T["grid"]), visible=False, zorder=20)

    def onMove(ev):
        if ev.inaxes is not ax:
            if vline.get_visible():
                vline.set_visible(False); box.set_visible(False); fig.canvas.draw_idle()
            return
        g = state["gen"]
        k = int(np.clip(np.searchsorted(g, ev.xdata), 0, len(g) - 1))
        d = state["d"]
        vline.set_xdata([g[k], g[k]]); vline.set_visible(True)
        box.set_text(f"gen {int(g[k])}   mean {d['mean'][k]:,.0f}   median {d['median'][k]:,.0f}   "
                     f"p10-p90 {d['p10'][k]:,.0f}-{d['p90'][k]:,.0f}   best {d['best'][k]:,.0f}   best ever {state['bestEver'][k]:,.0f}")
        box.set_visible(True)
        fig.canvas.draw_idle()
    if getattr(fig, "_hoverCid", None) is not None:
        fig.canvas.mpl_disconnect(fig._hoverCid)
    fig._hoverCid = fig.canvas.mpl_connect("motion_notify_event", onMove)


def newestCsv():
    files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "*.csv"))
    if not files:
        raise SystemExit("no models/*.csv found - start a training run first (it writes one line per generation)")
    return max(files, key=os.path.getmtime)


def activeCsvs(maxAge=180.0):
    '''The run CSVs written to within the last maxAge seconds = the runs that are going right now, oldest start first.'''
    now = time.time()
    files = [f for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "*.csv"))
             if now - os.path.getmtime(f) < maxAge and not os.path.basename(f).startswith("stress")]
    return sorted(files, key=lambda f: os.path.basename(f)[-23:])          # '<name>_YYYY-MM-DD_HH-MM-SS.csv'


MAX_PANELS = 4


class Panels:
    '''One window with a panel per running run (sub-figures of one figure). A panel keeps following its run; when that run ends (the next phase of a
    nightRun starts a new CSV) it takes over a running CSV nobody shows yet, so two parallel runs never swap places.'''

    def __init__(self, fig):
        self.fig, self.paths, self.subs = fig, [], []

    def update(self):
        active = activeCsvs()
        claimed = {p for p in self.paths if p in active}
        for k, p in enumerate(self.paths):
            if p not in active:
                free = [c for c in active if c not in claimed]
                if free:
                    self.paths[k] = free[0]
                    claimed.add(free[0])
        for c in active:
            if c not in claimed and len(self.paths) < MAX_PANELS:
                self.paths.append(c)
                claimed.add(c)
        self.paths = [p for p in self.paths if p in active]          # a run that ended and was not replaced loses its panel
        if not self.paths:
            self.paths = [newestCsv()]                  # nothing is running: show the newest run
        if len(self.subs) != len(self.paths):
            n = len(self.paths)
            cols = n if n <= 2 else 2
            rows = -(-n // cols)
            self.fig.clear()
            self.fig.set_size_inches(12.5 if cols == 1 else 10.2 * cols, 7.6 * rows, forward=True)
            self.subs = list(self.fig.subfigures(rows, cols, wspace=0.01, hspace=0.01, squeeze=False).ravel())[:n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Learning curve of a GA training run")
    ap.add_argument("csv", nargs="*", help="run CSV(s), oldest first (default: newest in models/)")
    ap.add_argument("--chain", action="store_true", help="follow the startFrom links back and join all runs of this model lineage")
    ap.add_argument("--live", nargs="?", const=5.0, type=float, metavar="SECONDS", help="redraw every N seconds (default 5)")
    ap.add_argument("--save", metavar="PNG", help="write a PNG instead of opening a window")
    ap.add_argument("--last", type=int, metavar="N", help="zoom into the last N generations")
    ap.add_argument("--latest", action="store_true", help="with --live: always follow the newest CSV in models/ (switches by itself when a new run/phase starts)")
    ap.add_argument("--running", action="store_true", help="one panel per run that is going right now (CSV written in the last 3 minutes), up to 4; with --live they follow their runs")
    ap.add_argument("--full", action="store_true", help="x axis spans the whole planned run, not just the finished generations (time-capped: up to where the cap is reached)")
    ap.add_argument("--log", action="store_true", help="logarithmic y axis")
    ap.add_argument("--light", action="store_true", help="light theme")
    a = ap.parse_args()
    theme = "light" if a.light else "dark"
    paths = a.csv or [newestCsv()]

    if a.save:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() == "agg":
        raise SystemExit(f"matplotlib has no window backend in this Python ({sys.executable}).\n"
                         "Use the project venv (source venv/bin/activate, or ./venv/bin/python plotRun.py ...), "
                         "which has PyQt6, or write a PNG with --save out.png.")
    fig = plt.figure(figsize=(12.5, 7.6), dpi=100)
    panels = Panels(fig) if a.running else None

    def redraw():
        '''Draws everything once; returns the hover states.'''
        if panels:
            panels.update()
            return [draw(sub, [p], theme, a.last, a.log, a.full, a.chain) for sub, p in zip(panels.subs, panels.paths)]
        return [draw(fig, paths, theme, a.last, a.log, a.full, a.chain)]

    def hover(states):
        for target, st in zip(panels.subs if panels else [fig], states):
            if st:
                addHover(target, st, theme)

    states = redraw()
    if a.save:
        fig.savefig(a.save, facecolor=fig.get_facecolor())
        print("saved", a.save)
    else:
        hover(states)
        plt.show(block=not a.live)
        while a.live and plt.fignum_exists(fig.number):
            # wait inside the window's own event loop (keeps it responsive). NOT plt.pause(): that calls
            # show() every cycle, which raises the window to the front and steals focus on every refresh.
            fig.canvas.start_event_loop(a.live)
            if not plt.fignum_exists(fig.number):
                break
            if a.latest and not panels:
                try:
                    newest = newestCsv()
                    if newest != paths[-1]:
                        print("following", newest, flush=True)
                        paths = [newest]
                except SystemExit:
                    pass
            try:
                hover(redraw())
                fig.canvas.draw_idle()
            except Exception:                         # a bad frame (e.g. half-written CSV line) must never close the window
                import traceback
                traceback.print_exc()
