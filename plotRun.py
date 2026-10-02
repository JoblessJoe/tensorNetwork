'''
Learning curve for a GA training run, read from the CSV that sprudlerTrainingLoop writes
(one line per generation, next to the saved model: models/<name>_<timestamp>.csv).
Runs in its own process, so it never slows the training down.

    python plotRun.py                      newest CSV in models/, one window
    python plotRun.py --live               same, redrawn every 5 s while the run is going
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


def rolling(x, window):
    '''Centered-ish trailing moving average that is defined from the first point on.'''
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
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


def draw(fig, path, theme, last=None, log=False, full=False):
    T = THEMES[theme]
    meta, d = loadRun(path) if os.path.exists(path) else ({}, {k: np.empty(0) for k in ("gen",)})
    fig.clear()
    fig.set_facecolor(T["surface"])
    if len(d["gen"]) == 0:
        fig.text(0.5, 0.5, "waiting for the first generation...", color=T["ink2"], ha="center", va="center", fontsize=14)
        return None

    total = int(meta.get("generations", 0)) or None
    gen = d["gen"]
    sel = slice(-last, None) if last else slice(None)
    bestEver = np.maximum.accumulate(d["best"])
    window = max(5, len(gen) // 12)
    smooth = rolling(d["mean"], window)

    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1], hspace=0.08, left=0.075, right=0.82, top=0.80, bottom=0.09)
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
    ax.set_xlim(x0 - 0.5, max(xr, x0 + 1) + 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("score per game (points)", color=T["ink2"], fontsize=10)

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
    dt = np.diff(d["elapsed_s"], prepend=0.0)
    axT.bar(gen, dt, width=1.0, color=T["muted"], alpha=0.55, linewidth=0, zorder=2)
    axT.set_ylabel("sec / gen", color=T["ink2"], fontsize=10)
    axT.set_xlabel("generation", color=T["ink2"], fontsize=10)
    axT.set_ylim(0, max(dt[sel].max(), 1) * 1.15)

    # headline + facts
    elapsed = d["elapsed_s"][-1]
    perGen = rolling(dt, 20)[-1]
    done = int(gen[-1])
    eta = f"   ETA {fmtTime((total - done) * perGen)}" if total and done < total else ("   finished" if total else "")
    prog = f"generation {done}" + (f" / {total}" if total else "")
    name = meta.get("name", os.path.basename(path))
    fig.text(0.075, 0.955, name, color=T["ink"], fontsize=17, fontweight="bold", va="top")
    fig.text(0.075, 0.915, f"{prog}   |   elapsed {fmtTime(elapsed)}{eta}   |   {perGen:.1f} s/gen", color=T["ink2"], fontsize=10.5, va="top")

    sinceRecord = done - int(gen[rec[-1]]) if len(rec) else done
    facts = f"trend now {smooth[-1]:,.0f}"
    if len(gen) > 1:                                  # change over the last (up to) 100 generations needs a point to compare with
        recent = min(100, len(gen) - 1)
        gain = smooth[-1] - smooth[-1 - recent]
        facts += f"   ({gain:+,.0f} / {recent} gens, {100 * gain / max(smooth[-1 - recent], 1e-9):+.0f}%)"
    facts += f"   |   best ever {bestEver[-1]:,.0f}, set {sinceRecord} gens ago   |   {len(rec)} records"
    fig.text(0.075, 0.875, facts, color=T["ink2"], fontsize=10.5, va="top")
    cfg = (f"population {meta.get('population', '?')}  games/network {meta.get('games', '?')}  mutation {meta.get('mutationRate', '?')}"
           f" / sigma {meta.get('sigma', '?')}  elites {meta.get('elites', '?')}  keep {meta.get('keepPart', '?')}"
           f"  start heights 0-{meta.get('maxStartHeight')}" if meta.get("maxStartHeight", "None") != "None" else
           f"population {meta.get('population', '?')}  games/network {meta.get('games', '?')}  mutation {meta.get('mutationRate', '?')}"
           f" / sigma {meta.get('sigma', '?')}  elites {meta.get('elites', '?')}  keep {meta.get('keepPart', '?')}  normal start")
    fig.text(0.075, 0.838, cfg, color=T["muted"], fontsize=9.5, va="top")

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Learning curve of a GA training run")
    ap.add_argument("csv", nargs="?", help="run CSV (default: newest in models/)")
    ap.add_argument("--live", nargs="?", const=5.0, type=float, metavar="SECONDS", help="redraw every N seconds (default 5)")
    ap.add_argument("--save", metavar="PNG", help="write a PNG instead of opening a window")
    ap.add_argument("--last", type=int, metavar="N", help="zoom into the last N generations")
    ap.add_argument("--full", action="store_true", help="x axis spans the whole planned run, not just the finished generations")
    ap.add_argument("--log", action="store_true", help="logarithmic y axis")
    ap.add_argument("--light", action="store_true", help="light theme")
    a = ap.parse_args()
    theme = "light" if a.light else "dark"
    path = a.csv or newestCsv()

    if a.save:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() == "agg":
        raise SystemExit(f"matplotlib has no window backend in this Python ({sys.executable}).\n"
                         "Use the project venv (source venv/bin/activate, or ./venv/bin/python plotRun.py ...), "
                         "which has PyQt6, or write a PNG with --save out.png.")
    fig = plt.figure(figsize=(12.5, 7.6), dpi=110)
    state = draw(fig, path, theme, a.last, a.log, a.full)
    if a.save:
        fig.savefig(a.save, facecolor=fig.get_facecolor())
        print("saved", a.save)
    else:
        if state:
            addHover(fig, state, theme)
        plt.show(block=not a.live)
        while a.live and plt.fignum_exists(fig.number):
            # wait inside the window's own event loop (keeps it responsive). NOT plt.pause(): that calls
            # show() every cycle, which raises the window to the front and steals focus on every refresh.
            fig.canvas.start_event_loop(a.live)
            if not plt.fignum_exists(fig.number):
                break
            try:
                state = draw(fig, path, theme, a.last, a.log, a.full)
                if state:
                    addHover(fig, state, theme)
                fig.canvas.draw_idle()
            except Exception:                         # a bad frame (e.g. half-written CSV line) must never close the window
                import traceback
                traceback.print_exc()
